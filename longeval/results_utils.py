from collections import defaultdict, Counter
import glob
import hashlib
import json
import os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from dateutil import parser as time_parser

from statsmodels.stats.inter_rater import fleiss_kappa

from longeval.metrics.time_utils import extract_worker_stats
from longeval.preprocessing_utils import csv_dict_read, jsonl_read


worker_map = {
    "A8E488R3B1IJX": "S",
    "A3W4CVF2TOE02J": "N",
    "A3BHTV9OQFZ0I5": "I",
    "A1RPGDHSHD8Z7S": "E",
    "ALNG6YHXX5QCB": "M",
    "A1KPNR32IOU0TP": "S2",
    "A2ID05QYC2M08O": "C",
    "ARMPOMI2QW88Q": "A",
    "AH7KZI27KMN26": "T",
    "A3I3IWO6NHV4YD": "P",
    "A1J8KV750X2VT0": "K",
    "A33ITCQQAQU0ZN": "Y"
}

def compute_fine_scores(fine_annotation_data, story_list, story_key_fn, skip_model, num_keep, verbose=False):
    all_fine_scores = []

    for story in story_list:
        if skip_model and skip_model in story:
            continue
        relevant_rows = [x for x in fine_annotation_data if story_key_fn(x) == story]
        worker_scores = get_worker_scores(relevant_rows, fine_grained=True)
        all_y_pts = []
        for worker_data in worker_scores:
            sorted_rows = sort_by_timestamps(worker_data["rows"])
            num_yes, num_no = extract_counts(sorted_rows[:num_keep], print_counts=False)
            all_y_pts.append(100 * num_yes / (num_yes + num_no))
        all_fine_scores.append(all_y_pts)

    if verbose:
        print(f"Average fine max - min rating = {np.mean([np.max(x) - np.min(x) for x in all_fine_scores])}")
        print(f"Average fine std-dev rating = {np.mean([np.std(x) for x in all_fine_scores])}")
    return all_fine_scores


def read_squality_coarse_data(coarse_src_files, story_list, skip_model, verbose=False):
    squality_human_data = jsonl_read(coarse_src_files)
    all_squality_scores = []
    for story in story_list:
        if skip_model and skip_model in story:
            continue
        orig_human_ratings = get_squality_data_stats(squality_human_data, story, rating_type='correctness-rating')
        all_squality_scores.append(orig_human_ratings["ratings"])

    if verbose:
        print(f"Average coarse max - min rating = {np.mean([np.max(x) - np.min(x) for x in all_squality_scores])}")
        print(f"Average coarse std-dev rating = {np.mean([np.std(x) for x in all_squality_scores])}")
    return all_squality_scores, squality_human_data

def get_squality_data_stats(squality_human_data, story, rating_type='correctness-rating', question_ids=['0']):
    passage_id = story.split("_")[0]
    model_name = "-".join(story.split("_")[1:])
    all_data = []
    for ques_id in question_ids:
        human_data = [x for x in squality_human_data if x['passage-id'] == passage_id][0]['questions'][ques_id][model_name]
        data = {
            "ratings": [x[rating_type] for x in human_data['reviews']],
            "passage_id": passage_id,
            "model_name": model_name,
            "response": human_data["response"]
        }
        all_data.append(data)
    if len(all_data) == 1:
        return all_data[0]
    else:
        return all_data

def read_coarse_data(coarse_src_files, story_list, skip_model, verbose=False):
    # Read in the annotation data
    coarse_src_files = glob.glob(coarse_src_files)
    all_coarse_data = defaultdict(list)
    summary_src_doc_data_coded = {}

    for file in coarse_src_files:
        with open(file) as f:
            data = json.loads(f.read())
        for dd in data:
            model_article_code = dd['data']['model_article_code']
            try:
                annotation = dd['annotations'][0]['result'][0]['value']['choices'][0]
            except:
                annotation = dd['annotations'][0]['result'][1]['value']['choices'][0]
            # multiply by 20 to project 0-5 Likert scale to 0-100
            all_coarse_data[model_article_code].append(20 * int(annotation))
            if model_article_code not in summary_src_doc_data_coded:
                summary_src_doc_data_coded[model_article_code] = {
                    'source_doc': dd['data']['original_source_doc'],
                    'summary': dd['data']['original_summary']
                }
    # compute the COARSE scores
    all_coarse_scores = []
    summary_src_doc_data = {}

    for story in story_list:
        model_article_code = hashlib.md5(
            story.replace("bigbird_pegasus", "bigbird-pegasus").encode('utf-8')).hexdigest()
        summary_src_doc_data[story] = summary_src_doc_data_coded[model_article_code]
        if skip_model and skip_model in story:
            continue
        all_coarse_scores.append(all_coarse_data[model_article_code])

    if verbose:
        print(f"Average coarse max - min rating = {np.mean([np.max(x) - np.min(x) for x in all_coarse_scores])}")
        print(f"Average coarse std-dev rating = {np.mean([np.std(x) for x in all_coarse_scores])}")
    return all_coarse_scores, summary_src_doc_data


def sort_by_timestamps(rows, extract_fn=lambda x: x):
    submit_times = [x['SubmitTime'] for x in rows]
    assert all(["PDT" in st for st in submit_times])
    submit_times = [time_parser.parse(st.replace("PDT", "")).timestamp() for st in submit_times]
    work_submit_times = [(extract_fn(x), y) for x, y in zip(rows, submit_times)]
    work_submit_times.sort(key=lambda x: x[1])
    work_submit_times = [x[0] for x in work_submit_times]
    return work_submit_times

def get_story_list(rows):
    story_list = []
    story_key_fn = lambda x: f'{x["Input.doc_id"]}_{x["Input.model"]}'.replace("-", "_")
    story_list = list(set([story_key_fn(x) for x in rows]))
    story_list.sort()
    return story_list, story_key_fn

def read_mturk_data(files):
    pattern_list = files.split(",")
    all_data = []
    for pattern in pattern_list:
        curr_files = glob.glob(pattern)
        for x in curr_files:
            dataset = csv_dict_read(x)
            filename = os.path.basename(x)[:-4]
            for dd in dataset:
                dd["filename"] = filename
            all_data.extend(dataset)
    return [x for x in all_data if x["WorkerId"] not in ["A1KPNR32IOU0TP", "A8OTFKR8I2PL0", "A3DC192EMQSNUQ"] and "claim-sent" in x["Input.annotated_summary"]]

def plot_time_vary(uniq_settings, all_settings, relevant_rows, extract_fn, filename, y_label="Time taken (seconds)", trim=None, smoothing=1):
    matplotlib.rcParams.update({'font.size': 18})
    for uset in uniq_settings:
        filtered_rows = [y for x, y in zip(all_settings, relevant_rows) if x == uset]
        # cluster by filenames
        file_to_rows = defaultdict(list)
        for row in filtered_rows:
            file_to_rows[row["WorkerId"] + "filename"].append(row)

        for k, v in file_to_rows.items():
            file_to_rows[k] = sort_by_timestamps(v, extract_fn=extract_fn)
            if trim:
                file_to_rows[k] = file_to_rows[k][:trim]

        x_pts = [x / trim for x in range(trim)]
        work_submit_times = []
        for i in range(trim):
            try:
                work_submit_times.append(np.mean([x[i] for x in file_to_rows.values()]))
            except:
                import pdb; pdb.set_trace()
                pass
        work_submit_times = [np.mean(work_submit_times[i:i + smoothing]) for i in range(len(work_submit_times))]
        plt.plot(x_pts, work_submit_times, label=uset)

    plt.xlabel("Fraction of summary units annotated")
    plt.ylabel(y_label)
    plt.legend(bbox_to_anchor=(1.0, 1.0), loc="upper right", ncol=1)
    plt.savefig(filename, bbox_inches="tight")
    plt.clf()


def get_scores(rows, str1, num_anns_agreement, accuracy):
    if len(rows) == 0:
        return

    extract_agreement(rows, num_anns_agreement)

    if accuracy:
        positives, negatives, all1 = extract_accuracy(rows)

    time_stats = extract_worker_stats(rows)
    median_times = []
    for worker_id, worker_stats in time_stats[1].items():
        median_times.append(worker_stats['median_first_five'])
    print(f"Time taken using Akoury et al. 2019 = {np.mean(median_times)}")
    time_taken = [int(x["WorkTimeInSeconds"]) for x in rows]
    scroll_after_vals = [extract_source_used(x) for x in rows]

    time_taken_support_unused = [x for x, y in zip(time_taken, scroll_after_vals) if y]

    print(f"Setting = {str1}")
    print(f"Time taken = {np.median(time_taken):.2f} median, {np.mean(time_taken):.2f} mean")
    print(f"Support unused = {np.mean(scroll_after_vals):.2f} ({sum(scroll_after_vals)} / {len(scroll_after_vals)})",)

    extract_agreement(rows, num_anns_agreement)

    if accuracy:
        # print(f"Positive scores ({str1}) = {np.mean(positives):.2f} ({sum(positives)} / {len(positives)})")
        # print(f"Negative scores ({str1}) = {np.mean(negatives):.2f} ({sum(negatives)} / {len(negatives)})")
        print(f"Accuracy = {np.mean(all1):.2f} ({sum(all1)} / {len(all1)}), Time Taken = {np.median(time_taken):.2f}")

    if sum(scroll_after_vals) > 0:
        if accuracy:
            acc_support_unused = [x for x, y in zip(all1, scroll_after_vals) if y]
            print(f"Accuracy on support unused = {np.mean(acc_support_unused):.2f} ({sum(acc_support_unused)} / {len(acc_support_unused)})")
        # extract_agreement([x for x, y in zip(rows, scroll_after_vals) if y], prefix="Agreement on support unused")
        print(f"Time on support unused = {np.median(time_taken_support_unused):.2f}")
    print("")


def extract_counts(relevant_rows, prefix="", print_counts=True):
    num_yes = [x["Answer.semantic-similarity.label"] == "Yes" for x in relevant_rows]
    num_no = [x["Answer.semantic-similarity.label"] == "No" for x in relevant_rows]
    if print_counts:
        print(f"{prefix} counts = {sum(num_yes)} yes ({100 * sum(num_yes) / len(relevant_rows):.1f}%), {sum(num_no)} no ({100 * sum(num_no) / len(relevant_rows):.1f}%)")
    else:
        return sum(num_yes), sum(num_no)

def get_worker_scores(relevant_rows, fine_grained=False):
    if isinstance(relevant_rows[0], tuple) and len(relevant_rows[0]) == 2:
        relevant_rows = [y for x in relevant_rows for y in x[1]]

    assert isinstance(relevant_rows, list) and isinstance(relevant_rows[0], dict)

    worker_ids = list(set([x["WorkerId"] for x in relevant_rows]))
    scores = []
    for wid in worker_ids:
        rows = [x for x in relevant_rows if x["WorkerId"] == wid]
        num_yes, num_no = extract_counts(rows, print_counts=False)
        if fine_grained:
            scores.append({
                "num_yes": num_yes,
                "num_no": num_no,
                "rows": rows,
                "wid": wid
            })
        else:
            scores.append(100 * num_yes / (num_yes + num_no))
    return scores

def extract_source_used(row):
    logs = json.loads(row['Answer.logging_info'])
    try:
        return int(max([y["scroll_after"] for y in logs]) == 0)
    except:
        return 0

def extract_accuracy(rows):
    positives = [x["Answer.semantic-similarity.label"] == "Yes" for x in rows if x["Input.transformation"] == "Identity"]
    negatives = [x["Answer.semantic-similarity.label"] == "No" for x in rows if x["Input.transformation"] != "Identity"]
    all1 = positives + negatives

    clustered_rows = defaultdict(list)
    for row in rows:
        clustered_rows[row["Input.annotated_summary"]].append(row)
    positives2 = []
    negatives2 = []
    for k, v in clustered_rows.items():
        if len(v) != 3:
            continue
        labels = [x["Answer.semantic-similarity.label"] for x in v]
        num_yes = sum([x == "Yes" for x in labels])
        num_no = sum([x == "No" for x in labels])

        if v[0]["Input.transformation"] == "Identity":
            positives2.append(num_yes > num_no)
        else:
            negatives2.append(num_yes < num_no)
    all2 = positives2 + negatives2

    # return positives, negatives, all1
    return positives2, negatives2, all2

def extract_agreement(rows, num_anns_agreement=None, prefix="Agreement"):
    annotation_dict = defaultdict(list)
    for row in rows:
        annotation_dict[row['Input.annotated_summary']].append(row["Answer.semantic-similarity.label"])

    if num_anns_agreement is None:
        num_anns_agreement = max([len(v) for k, v in annotation_dict.items()])

    annotation_table = []
    num_unique = []
    for key, anns in annotation_dict.items():
        assert len(anns) <= num_anns_agreement
        if len(anns) == num_anns_agreement:
            annotation_table.append(
                [anns.count("Yes"), anns.count("No")]
            )
            num_unique.append(len(set(anns)))

    if len(annotation_table) == 0:
        return
    fleiss_k = fleiss_kappa(annotation_table)
    randolph_k = fleiss_kappa(annotation_table, method='randolph')
    print(f"{prefix} ({len(annotation_table)} pairs, {num_anns_agreement} anns) = {fleiss_k:.4f} Fleiss, {randolph_k:.4f} Randolph, {Counter(num_unique)} unique counter")
