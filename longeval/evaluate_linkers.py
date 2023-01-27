import argparse
import os
import numpy as np
import torch
import tqdm

from longeval.linkage.utils import get_linking_fn, print_recall, dataset_list, squad_f1_matrix_fn, load_label_studio
from longeval.linkage.similarity.test_sim import find_similarity_matrix
from longeval.preprocessing_utils import process_ms2, get_sents


parser = argparse.ArgumentParser()
parser.add_argument('--linking_algorithm', default="superpal")
parser.add_argument('--doc_to_doc_matching', default="sim_wieting_2019")
parser.add_argument('--verbose', action='store_true')
parser.add_argument('--export_outputs', action='store_true')
parser.add_argument('--contextualization', default="spans")
args = parser.parse_args()

def main(args):
    per_doc_setup_fn, matrix_fn = get_linking_fn(args.linking_algorithm)

    if args.doc_to_doc_matching == "sim_wieting_2019":
        doc_to_doc_matching_fn = find_similarity_matrix
    else:
        doc_to_doc_matching_fn = squad_f1_matrix_fn


    total_miss = 0

    for dataset in dataset_list:
        sim_cache = None

        source_data, linked_data = load_label_studio(dataset, args.contextualization)

        if args.export_outputs:
            output_dir = f"outputs/linking_{dataset['name']}_{args.linking_algorithm}_{args.contextualization}"
            os.makedirs(output_dir, exist_ok=True)

        all_ranks = []
        num_candidates = []
        all_ranks_best = []

        all_dists = []
        all_dists_best = []

        doc_ids_done = {}

        for dd in tqdm.tqdm(source_data, disable=len(source_data) == 1, desc="Processing summaries..."):
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

                if args.verbose:
                    print_recall(all_ranks_best, prefix="Best match")
                    print_recall(all_dists_best, prefix="Best distance to top 5 matches", k_list=[1, 3, 5])

            doc_ids_done[doc_id] = 1
        print(f"\nDataset = {dataset['name']}")
        print_recall(all_ranks_best, prefix="Best match")
        print_recall(all_ranks, prefix="All matches")
        print(f"Avg candidates = {np.mean(num_candidates):.2f}")

if __name__ == '__main__':
    main(args)
