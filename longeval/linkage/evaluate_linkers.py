"""
python eval_summary/linkage/evaluate_sim.py
"""

import argparse
import csv
import functools
import json
import os
import numpy as np
import pandas as pd
import torch
import tqdm

from utils import get_linking_fn, print_recall, dataset_list, squad_f1_matrix_fn, load_label_studio

from longeval.linkage.similarity.test_sim import find_similarity_matrix
from longeval.preprocessing_utils import export_server, process_ms2, get_sents


parser = argparse.ArgumentParser()
parser.add_argument('--linking_algorithm', default="superpal")
parser.add_argument('--doc_to_doc_matching', default="superpal")
parser.add_argument('--disable', default="")
parser.add_argument('--verbose', action='store_true')
parser.add_argument('--export_outputs', action='store_true')
parser.add_argument('--contextualization', default="decontextualized")
args = parser.parse_args()

per_doc_setup_fn, matrix_fn = get_linking_fn(args.linking_algorithm)

if args.doc_to_doc_matching == "sim_wieting_2019":
    doc_to_doc_matching_fn = find_similarity_matrix
else:
    doc_to_doc_matching_fn = squad_f1_matrix_fn

disabled_datasets = args.disable.split(",")

total_miss = 0

for dataset in dataset_list:
    if dataset['name'] in disabled_datasets:
        continue
    sim_cache = None

    if dataset["label_studio"]:
        source_data, linked_data = load_label_studio(dataset, args.contextualization)
    else:
        with open(dataset['source_file'], "r") as f:
            source_data = [json.loads(x) for x in f.read().strip().split("\n")]
        linked_data = pd.read_csv(dataset['data_file'])
        linked_data = linked_data[linked_data["Best Source Match"].notnull()]

    if args.export_outputs:
        output_dir = f"outputs/linking_{dataset['name']}_{args.linking_algorithm}_{args.contextualization}"
        os.makedirs(output_dir, exist_ok=True)

    all_ranks = []
    num_candidates = []
    all_ranks_best = []

    all_dists = []
    all_dists_best = []

    doc_ids_done = {}
    file_num = 0

    for dd in tqdm.tqdm(source_data, disable=len(source_data) == 1):
        outputs = ""
        doc_id = dd['doc_id']
        df_small = linked_data.loc[linked_data["Document ID"] == int(doc_id)]
        if len(df_small) == 0 or doc_id in doc_ids_done:
            continue

        summary_scus = df_small['Summary SCU'].values.tolist()
        best_source = [process_ms2(x) for x in df_small['Best Source Match'].values.tolist()]
        assert len(summary_scus) == len(best_source)

        for scu, best_src in tqdm.tqdm(zip(summary_scus, best_source), total=len(summary_scus), disable=len(source_data) > 1):
            if len(best_src) < 2:
                # empty best source sentences, appended by .
                total_miss += 1
                continue
            best_src = get_sents(best_src)
            doc_sents = get_sents(dd['document'])
            doc_sents = [x.strip() for x in doc_sents if x.strip()]

            if dataset["cache"]:
                # only use this for datasets with a single document
                best_src_matches, sim_cache, _ = find_similarity_matrix(doc_sents, best_src, vecs1=sim_cache, return_cache=True)
            else:
                best_src_matches = doc_to_doc_matching_fn(doc_sents, best_src)

            if per_doc_setup_fn is not None:
                _matrix_fn = per_doc_setup_fn(doc_sents, matrix_fn)
            else:
                _matrix_fn = matrix_fn

            best_src_idx = best_src_matches.argmax(dim=0).tolist()

            if _matrix_fn == find_similarity_matrix and dataset["cache"]:
                scu_matches = _matrix_fn([scu], doc_sents, vecs2=sim_cache)[0]
            else:
                scu_matches = _matrix_fn([scu], doc_sents)[0]
            scu_scores = torch.sort(scu_matches, descending=True)

            top_five_matches = scu_scores.indices[:5]
            curr_dists = []
            for x in best_src_idx:
                dists = [abs(x - y.item()) for y in top_five_matches]
                curr_dists.append(min(dists))

            all_dists.extend(curr_dists)
            all_dists_best.append(min(curr_dists))

            best_src_ranks = [scu_scores.indices.tolist().index(x) + 1 for x in best_src_idx]

            all_ranks.extend(best_src_ranks)
            all_ranks_best.append(min(best_src_ranks))
            num_candidates.append(len(doc_sents))

            outputs += f"<b>SCU</> = {scu}\n<b>Gold Evidence</> = {best_src[np.argmin(best_src_ranks)]}\n<b>Gold Evidence Rank</> = {min(best_src_ranks)}\n"
            outputs += f"<b>Predicted evidence #1</> = {doc_sents[scu_scores.indices[0].item()]}\n"
            outputs += f"<b>Predicted evidence #2</> = {doc_sents[scu_scores.indices[1].item()]}\n"
            outputs += f"<b>Predicted evidence #2</> = {doc_sents[scu_scores.indices[2].item()]}\n\n"
            outputs += "-------------------------\n\n"

            if args.verbose:
                print_recall(all_ranks_best, prefix="Best match")
                print_recall(all_dists_best, prefix="Best distance to top 5 matches", k_list=[1, 3, 5])

        if args.export_outputs:
            export_server(outputs, f"{output_dir}/doc_{doc_id}")

        doc_ids_done[doc_id] = 1
    print(f"\nDataset = {dataset['name']}")
    print_recall(all_ranks_best, prefix="Best match")
    print_recall(all_ranks, prefix="All matches")
    print(f"Avg candidates = {np.mean(num_candidates):.2f}")
