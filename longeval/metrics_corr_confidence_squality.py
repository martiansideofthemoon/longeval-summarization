import argparse
import tqdm

from longeval.metrics.plot_violin_utils import plot_data
from longeval.results_utils import read_squality_coarse_data, compute_fine_scores, get_story_list, read_mturk_data, get_squality_data_stats
from longeval.metrics.metric_utils import get_metric, get_metrics_list, get_correlation_intervals
from longeval.preprocessing_utils import jsonl_read

parser = argparse.ArgumentParser()
parser.add_argument('--fine_src_files', default="data/squality_annotations/squality_fine/*", type=str)
parser.add_argument('--coarse_src_files', default="data/squality_annotations/squality_coarse/all-human-responses.jsonl", type=str)
parser.add_argument('--test_set_input_file', default="data/squality_annotations/squality_coarse/test.jsonl", type=str)
parser.add_argument('--num_samples', default=10000, type=int)
parser.add_argument('--num_keep', default=1000, type=int,
                    help=("Number of FINE units to keep for each summary."
                          "A large number like 1000 will include all units"))
parser.add_argument('--corr_type', default="pearson", type=str)
parser.add_argument('--skip_model', default=None, type=str)
parser.add_argument('--verbose', action='store_true')
parser.add_argument('--output_dir', default="outputs/metric_corrs", type=str)
parser.add_argument('--compare_to', default="references", type=str)
args = parser.parse_args()

def main(args):
    test_set_inputs = jsonl_read(args.test_set_input_file)
    test_set_inputs = {x['metadata']['passage_id']: x for x in test_set_inputs}

    # Read in the FINE annotations, which has the AMT format
    fine_annotation_data = read_mturk_data(args.fine_src_files)
    story_list, story_key_fn = get_story_list(fine_annotation_data)

    # Read in the COARSE annotations and compute COARSE scores
    all_coarse_scores, raw_data = read_squality_coarse_data(
        args.coarse_src_files, story_list, args.skip_model, verbose=args.verbose
    )
    # compute the FINE scores
    all_fine_scores = compute_fine_scores(
        fine_annotation_data, story_list, story_key_fn, args.skip_model, args.num_keep, verbose=args.verbose
    )

    # Get all metric scores
    metrics_list = get_metrics_list(args.compare_to)
    all_data = []

    for metric_name in metrics_list:
        metric_fn = get_metric(metric_name)
        all_metric_scores = []
        for story in tqdm.tqdm(story_list):
            if args.skip_model and args.skip_model in story:
                continue
            orig_human_ratings = get_squality_data_stats(raw_data, story)
            test_set_instance = test_set_inputs[orig_human_ratings["passage_id"]]
            if args.compare_to == "references":
                orig_refs = [x['response_text'] for x in test_set_instance['questions'][0]['responses'] if x['response_text']]
                # avoid comparing references to themselves - this happens in the human summary case
                orig_refs = [x for x in orig_refs if x != orig_human_ratings["response"]]
                if "human" in story:
                    # compare human summary to all *other* references
                    assert len(orig_refs) == 3
                else:
                    assert len(orig_refs) == 4
                metric_score = metric_fn(orig_human_ratings["response"], orig_refs)

            elif args.compare_to == "source":
                metric_score = metric_fn(orig_human_ratings["response"], [' '.join(test_set_instance['document'].split('\n'))])

            all_metric_scores.append(metric_score)

        # Compute correlation bounds using bootstrap resampling
        samples_coarse = get_correlation_intervals(
            all_coarse_scores, all_metric_scores, args.num_samples, corr_type=args.corr_type, verbose=args.verbose)
        samples_fine = get_correlation_intervals(
            all_fine_scores, all_metric_scores, args.num_samples, corr_type=args.corr_type, verbose=args.verbose)

        all_data.append((metric_name, samples_fine, samples_coarse))

        if args.skip_model == "human":
            filename = f"{args.output_dir}/squality_{args.corr_type}_{args.num_keep}_{args.compare_to}_model_only.pdf"
        else:
            filename = f"{args.output_dir}/squality_{args.corr_type}_{args.num_keep}_{args.compare_to}.pdf"

        plot_data(metrics=all_data,
                output_file=filename,
                xlabel="Pearson correlation" if "pearson" in args.corr_type else "KT correlation",
                first_patch_label=f"FINE",
                second_patch_label="COARSE",
                xlim_low=-0.2,
                font_size=18,
                plot_title="SQuALITY")

if __name__ == "__main__":
    main(args)
