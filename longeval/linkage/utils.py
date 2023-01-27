import collections as cll
import functools
import glob
import json
import pandas as pd
import re
import string
import torch
from bert_score import BERTScorer
from longeval.linkage.multivers_utils import multivers_setup
from longeval.linkage.summac_utils import summac_zero_setup
from longeval.linkage.superpal_utils import superpal_setup
from longeval.linkage.similarity.test_sim import find_similarity_matrix
from rank_bm25 import BM25Okapi
from rouge_score import rouge_scorer
from functools import partial

from transformers import AutoTokenizer, DPRContextEncoder


dataset_list = [
    {"name": "SQuALITY-Alignments", "source_file": "data/squality_alignment/data.json", "data_file": None, "cache": False, "label_studio": True, "source_dataset": "squality"}
]
squality_info_unit_re = re.compile(r"(\#\d+\s=\s)(.*)(Contextualized =\s?)(.*)(Span =\s?)(.*)(Support\s?=\s?)(.*)")
ms2_info_unit_re = re.compile(r"(\#\s?\d+\s?=\s?)(.*)(Support\s?=\s?)(.*)")

def get_linking_fn(linking_algorithm):
    per_doc_setup_fn = None
    if linking_algorithm == "sim_wieting_2019":
        matrix_fn = find_similarity_matrix
    elif linking_algorithm == "bm25":
        per_doc_setup_fn = bm25_setup
        matrix_fn = bm25_matrix_fn
    elif linking_algorithm == "squad_f1":
        matrix_fn = squad_f1_matrix_fn
    elif linking_algorithm.startswith("rouge"):
        scorer = rouge_setup(linking_algorithm)
        matrix_fn = functools.partial(rouge_matrix_fn, scorer=scorer, metric=linking_algorithm)
    elif linking_algorithm == "dpr":
        matrix_fn = dpr_setup()
    elif linking_algorithm == "bert_score":
        matrix_fn = bert_score_setup()
    elif linking_algorithm.startswith("multivers"):
        matrix_fn = multivers_setup(linking_algorithm)
    elif linking_algorithm == "summac_zero":
        matrix_fn = summac_zero_setup()
    elif linking_algorithm == "superpal":
        matrix_fn = superpal_setup()
    else:
        raise ValueError("Invalid --linking_algorithm value")
    return per_doc_setup_fn, matrix_fn


def load_label_studio(dataset, contextualization="spans", keep_newlines=False):
    files = glob.glob(dataset["source_file"])
    data = []
    for filename in files:
        with open(filename, "r") as f:
            data.extend(json.loads(f.read()))
    source_data = []
    linked_data = []
    doc_id_done = {}
    for dd in data:
        if dd['data']['dataset'] != dataset['source_dataset']:
            continue
        if keep_newlines:
            doc_text = dd['data']['document'].replace("<br />", "\n")
        else:
            doc_text = " ".join(dd['data']['document'].replace("<br />", " ").split())
        doc_id = dd['data']['review_id']
        document = dd['data']['document']

        if doc_id not in doc_id_done:
            source_data.append({
                "document": doc_text,
                "doc_id": doc_id
            })
            doc_id_done[doc_id] = 1
        for ann in dd['annotations']:
            output = ann['result'][0]
            parsed_output = parse_labelstudio_output(output['value']['text'][0], contextualization)
            for op_tuple in parsed_output:
                linked_data.append(
                    [doc_id, document, dd['data']['background'], dd['data']['reference']] + list(op_tuple)
                )
    if contextualization == "both":
        linked_data = pd.DataFrame(linked_data, columns=["Document ID", "Document", "Context", "Source", "Decontextualized SCU", "Contextualized SCU", "Best Source Match", "Annotation Type"])
    else:
        linked_data = pd.DataFrame(linked_data, columns=["Document ID", "Document", "Context", "Source", "Summary SCU", "Best Source Match", "Annotation Type"])
    return source_data, linked_data


def parse_labelstudio_output(output, contextualization="decontextualized", debug=False):
    output = " ".join(output.split())
    output = output.split("Info Unit")
    parsed_output = []
    for op in output:
        op = op.strip()
        if not op or len(op) < 5 or op == "Skip - can't decontextualize":
            continue

        info_unit_matches = squality_info_unit_re.findall(op)
        if len(info_unit_matches) == 0:
            if debug:
                import pdb; pdb.set_trace()
                pass
            try:
                info_unit_matches = ms2_info_unit_re.findall(op)[0]
            except:
                import pdb; pdb.set_trace()
                pass
            decontext_idx = 1
            context_idx = 1
            span_idx = 1
            support_idx = 3
            annotation_type = "partial"
        else:
            info_unit_matches = info_unit_matches[0]
            decontext_idx = 1
            context_idx = 3
            span_idx = 5
            support_idx = 7
            annotation_type = "full"

        if contextualization == "both":
            parsed_output.append((info_unit_matches[decontext_idx].strip(), info_unit_matches[context_idx].strip(), info_unit_matches[support_idx].strip(), annotation_type))
        elif contextualization == "contextualized":
            parsed_output.append((info_unit_matches[context_idx].strip(), info_unit_matches[support_idx].strip(), annotation_type))
        elif contextualization == "decontextualized":
            parsed_output.append((info_unit_matches[decontext_idx].strip(), info_unit_matches[support_idx].strip(), annotation_type))
        elif contextualization == "spans":
            span_unit = info_unit_matches[span_idx].strip()
            if not span_unit.endswith(string.punctuation):
                span_unit = span_unit + "."
            parsed_output.append((span_unit, info_unit_matches[support_idx].strip(), annotation_type))
        elif contextualization == "diff_contextualized":
            if not exact_match(info_unit_matches[context_idx].strip(), info_unit_matches[decontext_idx].strip()):
                parsed_output.append((info_unit_matches[context_idx].strip(), info_unit_matches[support_idx].strip(), annotation_type))
        elif contextualization == "diff_decontextualized":
            if not exact_match(info_unit_matches[context_idx].strip(), info_unit_matches[decontext_idx].strip()):
                parsed_output.append((info_unit_matches[decontext_idx].strip(), info_unit_matches[support_idx].strip(), annotation_type))
        else:
            raise ValueError("Incorrect contextualization type")

    return parsed_output

def print_recall(ranks, prefix, k_list=[3, 5, 10]):
    print(f"{prefix}:")
    for k in k_list:
        num_valid = sum([x <= k for x in ranks])
        print(f"Recall@{k} = {num_valid / len(ranks):.4f} ({num_valid} / {len(ranks)})")

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def dpr_setup():
    tokenizer = AutoTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
    model = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
    model.cuda()
    model.eval()
    return functools.partial(dpr_matrix_fn, tokenizer=tokenizer, model=model)

def bert_score_setup():
    scorer = BERTScorer(model_type="microsoft/deberta-xlarge-mnli") # , lang="en", rescale_with_baseline=True)
    return functools.partial(bert_score_fn, scorer=scorer)

def bert_score_fn(first_list, second_list, scorer):
    all_scores = []
    for fl in first_list:
        scores = scorer.score([fl for _ in range(len(second_list))], second_list)[2].unsqueeze(dim=0)
        all_scores.append(scores)
    return torch.cat(all_scores, dim=0)

def dpr_matrix_fn(first_list, second_list, tokenizer, model):

    def encode_str(str_list):
        with torch.inference_mode():
            dpr_tensors = tokenizer(str_list, return_tensors="pt", padding=True, truncation=True, max_length=512)
            dpr_tensors.to("cuda")
            return model(**dpr_tensors).pooler_output

    BATCH_SIZE = 32
    # Encode the first list
    first_vectors = []
    for inst_num in range(0, len(first_list), BATCH_SIZE):
        first_vectors.append(encode_str(first_list[inst_num:inst_num + BATCH_SIZE]))
    first_vectors = torch.cat(first_vectors, dim=0)

    # Encode the second list
    second_vectors = []
    for inst_num in range(0, len(second_list), BATCH_SIZE):
        second_vectors.append(encode_str(second_list[inst_num:inst_num + BATCH_SIZE]))
    second_vectors = torch.cat(second_vectors, dim=0)

    # compute the matrix similarity between them
    return torch.matmul(first_vectors, second_vectors.t())


def bm25_setup(doc_sents, matrix_fn):
    bm25 = BM25Okapi([normalize_answer(x).split() for x in doc_sents])
    _matrix_fn = partial(matrix_fn, bm25=bm25)
    return _matrix_fn

def bm25_matrix_fn(first_list, second_list, bm25):
    all_scores = []
    for fl in first_list:
        scores = bm25.get_scores(normalize_answer(fl).split())
        all_scores.append(scores)
    return torch.Tensor(all_scores)

def rouge_setup(metric):
    # metric can be "rouge1", "rouge2" or "rougeL"
    scorer = rouge_scorer.RougeScorer([metric], use_stemmer=True)
    return scorer

def rouge_matrix_fn(first_list, second_list, scorer, metric):
    all_scores = []
    for fl in first_list:
        curr_scores = []
        for sl in second_list:
            curr_scores.append(scorer.score(fl, sl)[metric].fmeasure)
        all_scores.append(curr_scores)
    return torch.Tensor(all_scores)

def squad_f1_matrix_fn(first_list, second_list, stopwords=None):
    all_scores = []
    for fl in first_list:
        curr_scores = []
        for sl in second_list:
            curr_scores.append(f1_score(fl, sl, stopwords=stopwords)[2])
        all_scores.append(curr_scores)
    return torch.Tensor(all_scores)

def squad_em_matrix_fn(first_list, second_list, stopwords=None):
    all_scores = []
    for fl in first_list:
        curr_scores = []
        for sl in second_list:
            curr_scores.append(exact_match(fl, sl, stopwords=stopwords))
        all_scores.append(curr_scores)
    return torch.Tensor(all_scores)

def f1_score(prediction, ground_truth, gram=1, stopwords=None):
    """Calculate word level F1 score."""
    prediction = normalize_answer(prediction)
    ground_truth = normalize_answer(ground_truth)
    prediction_tokens = prediction.split()
    ground_truth_tokens = ground_truth.split()
    prediction_tokens = [
        " ".join(prediction_tokens[i:i + gram])
        for i in range(0, len(prediction_tokens) - gram + 1)
    ]
    ground_truth_tokens = [
        " ".join(ground_truth_tokens[i:i + gram])
        for i in range(0, len(ground_truth_tokens) - gram + 1)
    ]

    if stopwords:
        prediction_tokens = [x for x in prediction_tokens if x not in stopwords]
        ground_truth_tokens = [x for x in ground_truth_tokens if x not in stopwords]

    if not prediction_tokens and not ground_truth_tokens:
        return 1.0, 1.0, 1.0
    common = cll.Counter(prediction_tokens) & cll.Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0, 0, 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return precision, recall, f1

def exact_match(prediction, ground_truth, stopwords=None):
    """Calculate word level F1 score."""
    prediction = normalize_answer(prediction)
    ground_truth = normalize_answer(ground_truth)
    prediction_tokens = prediction.split()
    ground_truth_tokens = ground_truth.split()

    if stopwords:
        prediction_tokens = [x for x in prediction_tokens if x not in stopwords]
        ground_truth_tokens = [x for x in ground_truth_tokens if x not in stopwords]

    return " ".join(prediction_tokens) == " ".join(ground_truth_tokens)
