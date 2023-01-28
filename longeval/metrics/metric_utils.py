from sacrerouge.metrics import Rouge, SentBleu
from bert_score import BERTScorer
import functools
import tqdm
import random
import numpy as np
from longeval.linkage.summac.model_summac import SummaCZS, SummaCConv
from bleurt import score

from longeval.metrics.bart_score import BARTScorer

from scipy.stats import kendalltau, pearsonr


def get_metrics_list(compare_to):
    if compare_to == "source":
        metric_list = ['rouge-1_precision', 'rouge-2_precision', 'rouge-l_precision', 'bert_score_precision', 'summac', 'summac_conv']
    else:
        metric_list = ['rouge-1_f1', 'rouge-2_f1', 'rouge-l_f1', 'bart_score', 'bart_sc_pb', 'sent_bleu', 'bert_score', 'bleurt']
    return metric_list

def get_correlation_intervals(human_ratings, metric_ratings, num_samples, corr_type, confidence=95.0, verbose=False):
    all_corr_vals = []
    for _ in tqdm.tqdm(range(num_samples), disable=not verbose):
        sampled_human_scores = []
        for sample_list in human_ratings:
            sample = random.choices(sample_list, k=len(sample_list))
            sampled_human_scores.append(np.mean(sample))
        assert len(sampled_human_scores) == len(metric_ratings)
        if corr_type == "kendall":
            corr_val = kendalltau(sampled_human_scores, metric_ratings).correlation
        elif corr_type == "pearson":
            corr_val = pearsonr(sampled_human_scores, metric_ratings)[0]
        else:
            raise ValueError("Wrong correlation type")
        all_corr_vals.append(corr_val)

    lower_interval = (100.0 - confidence) / 2
    lower_percentile = np.percentile(all_corr_vals, lower_interval)
    upper_percentile = np.percentile(all_corr_vals, 100.0 - lower_interval)
    return np.array([x for x in all_corr_vals if lower_percentile <= x and x <= upper_percentile])

def get_metric(metric_name):
    if metric_name.startswith("rouge"):
        rouge = Rouge(max_ngram=2, compute_rouge_l=True)
        rouge_type, prec_rec_f1 =metric_name.split("_")
        metric_fn = functools.partial(sacrerouge_fn_builder, scorer=rouge.score, score_type=rouge_type, score_type2=prec_rec_f1)
    elif metric_name == "bert_score":
        metric_fn = bert_score_setup()
    elif metric_name == "bert_score_precision":
        metric_fn = bert_score_setup(score_type="precision")
    elif metric_name == "bert_score_recall":
        metric_fn = bert_score_setup(score_type="recall")
    elif metric_name == "sent_bleu":
        sent_bleu = SentBleu()
        metric_fn = lambda x, y: sent_bleu.score(x, y)['sent-bleu']
    elif metric_name == "bleurt":
        checkpoint = "/data/kalpesh/bleurt/BLEURT-20"
        scorer = score.BleurtScorer(checkpoint)
        metric_fn = functools.partial(bleurt_fn, scorer=scorer)
    elif metric_name == "summac":
        model = SummaCZS(granularity="sentence", model_name="vitc")
        metric_fn = functools.partial(summac_fn, scorer=model)
    elif metric_name == "summac_conv":
        model = SummaCConv(granularity="sentence", models=["vitc"], bins="percentile", start_file="eval_summary/summac/summac_conv_vitc_sent_perc_e.bin")
        metric_fn = functools.partial(summac_fn, scorer=model)
    elif metric_name == "bart_score":
        bart_scorer = BARTScorer(device='cuda:0', checkpoint='facebook/bart-large-cnn')
        metric_fn = functools.partial(bart_score_fn, scorer=bart_scorer)
    elif metric_name == "bart_sc_pb":
        bart_scorer = BARTScorer(device='cuda:0', checkpoint='facebook/bart-large-cnn')
        metric_fn = functools.partial(bart_score_fn, scorer=bart_scorer)
        bart_scorer.load(path='/data/kalpesh/bart.pth')
    else:
        raise ValueError("Invalid --metric_name value")
    return metric_fn

def bleurt_fn(generation, refs, scorer):
    scores = scorer.score(references=refs, candidates=[generation for _ in range(len(refs))])
    return max(scores)

def summac_fn(generation, document, scorer):
    if len(document) > 1:
        document = document[:1]
    return scorer.score(document, [generation])["scores"][0]

def bart_score_fn(generation, refs, scorer):
    score = scorer.multi_ref_score([generation], [refs], agg="max", batch_size=4)
    return score[0]

def sacrerouge_fn_builder(generation, references, scorer, score_type, score_type2):
    score_all = scorer(generation, references)
    return score_all[score_type][score_type2]

def bert_score_setup(score_type="f1"):
    scorer = BERTScorer(model_type="microsoft/deberta-xlarge-mnli", lang="en", rescale_with_baseline=True)
    return functools.partial(bert_score_fn, scorer=scorer, score_type=score_type)

def bert_score_fn(generation, references, scorer, score_type):
    scores = scorer.score([generation], [references])
    if score_type == "precision":
        return scores[0].item()
    elif score_type == "recall":
        return scores[1].item()
    else:
        return scores[2].item()

def get_summary_level_corr(story_list, list1, list2, corr_type="KT"):
    passage_ids = list(set([x.split("_")[0] for x in story_list]))
    corrs = []
    for pid in passage_ids:
        relevant_list1 = [x for x, y in zip(list1, story_list) if pid in y]
        relevant_list2 = [x for x, y in zip(list2, story_list) if pid in y]
        if corr_type == "KT":
            score = kendalltau(relevant_list1, relevant_list2).correlation
        elif corr_type == "pearson":
            score = pearsonr(relevant_list1, relevant_list2)[0]
        if np.isnan(score):
            score = 0.0
        corrs.append(score)
    return np.mean(corrs)

def get_system_level_corr(story_list, list1, list2):
    model_ids = list(set(["_".join(x.split("_")[1:]) for x in story_list]))
    model_scores1 = []
    model_scores2 = []
    for pid in model_ids:
        relevant_list1 = [x for x, y in zip(list1, story_list) if pid == "_".join(y.split("_")[1:])]
        relevant_list2 = [x for x, y in zip(list2, story_list) if pid == "_".join(y.split("_")[1:])]
        model_scores1.append(np.mean(relevant_list1))
        model_scores2.append(np.mean(relevant_list2))
    corr = kendalltau(model_scores1, model_scores2).correlation
    if np.isnan(corr):
        corr = 0.0
    return corr
