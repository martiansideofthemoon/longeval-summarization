import argparse
import random
import string
import tqdm

from longeval.linkage.utils import get_linking_fn
from longeval.preprocessing_utils import csv_dict_write, jsonl_read, split_sents, get_sents

from longeval.utils import get_predicted_evidence, fix_quote_issues


parser = argparse.ArgumentParser()
parser.add_argument('--src_file', default="data/pubmed_summaries/beam_3.jsonl", type=str,
                    help=("Input JSONL file with each line containing a dictionary with keys"
                          "'article' and the set of models to be evaluated (each model is a"
                          " separate key)."))
parser.add_argument('--scu_fraction', default=None, type=float,
                    help="Fraction of content units / SCUs to keep in the summary (default: 1.0).")
parser.add_argument('--scu_num', default=None, type=int,
                    help="Number of content units / SCUs to keep in the summary (default: all SCUs preserved).")
parser.add_argument('--num_truncate_splits', default=3, type=int,
                    help=("If --scu_num or --scu_fraction is set, make multiple splits of the data."
                          "Each split will have a different randomly selected set of SCUs."
                          "Each annotator should be provided with a different split."))
parser.add_argument('--num_articles', default=None, type=int, help="Number of articles to process.")
parser.add_argument('--output_dir', default="outputs/pubmed_beam_3", type=str)
parser.add_argument('--linking_algorithm', default="superpal", type=str)
parser.add_argument('--included_models', default="bigbird_pegasus;longt5;human", type=str)
args = parser.parse_args()

def main(args):
    # prepare linking function / SuperPAL
    per_doc_setup_fn, matrix_fn = get_linking_fn(args.linking_algorithm)
    summary_data = jsonl_read(args.src_file)

    if args.num_articles is not None:
        summary_data = summary_data[:args.num_articles]

    total_scus = 0
    included_models = args.included_models.split(";")

    for i, instance in enumerate(summary_data):
        instance["passage_id"] = f"article_{i}"
        # Each article will be output in a different file
        # The main reason for this is to let the annotator complete
        # multiple summaries from a single article and avoid context switching
        filename = f"{instance['passage_id']}_{'_'.join(included_models)}"
        print(filename)

        # break up article into invididual sentences
        original_doc_sents = get_sents(instance['article'])
        original_doc_sents = [x.strip() for x in original_doc_sents if x.strip() and "DOCTYPE html" not in x and not x.startswith("<!")]
        original_doc_sents = fix_quote_issues("\n".join(original_doc_sents)).split("\n")

        if per_doc_setup_fn is not None:
            _matrix_fn = per_doc_setup_fn(original_doc_sents, matrix_fn)
        else:
            _matrix_fn = matrix_fn

        output_list = []
        for model_name in included_models:
            # basic preprocessing on the summary
            response = instance[model_name]
            response = response.replace("<n>", "\n").replace("<s>", "").replace("</s>", "")
            # replace letter-punc-letter
            new_response = ""
            for i, dd in enumerate(response):
                new_response += dd
                if i == len(response) - 1:
                    continue
                if dd in string.punctuation and response[i + 1] in string.ascii_letters:
                    new_response += " "
            response = " ".join(new_response.split())

            # split the summary into SCUs
            sent_splits = [y for x in split_sents(response)[0] for y in x]
            # remove duplicate SCUs. They will later be marked together during annotation to save work
            sent_splits = list(set(sent_splits))
            assert all([x in response for x in sent_splits])

            total_scus += len(sent_splits)

            for x in tqdm.tqdm(sent_splits):
                annotated_summary = response.replace(x.strip(), f"<span id='claim-sent' style='background-color: #ffd500'>{x.strip()}</span>")
                if not x.endswith(string.punctuation):
                    x = x + "."
                # Align the SCU with the source document
                evidence_doc, num_predicted_evidence, _ = get_predicted_evidence(claim_scu=x,
                                                                                 doc_sents=original_doc_sents.copy(),
                                                                                 linker_matrix_fn=_matrix_fn,
                                                                                 style="highlight")
                output_instance = {
                    "doc_id": instance["passage_id"],
                    "source_doc": instance["article"].replace("\n", "<br />"),
                    "original_summary": response,
                    "annotated_summary": annotated_summary,
                    "original_source_doc": "<br />".join(original_doc_sents),
                    "num_predicted_evidence": num_predicted_evidence,
                    "prediction_annotated_source_doc": evidence_doc,
                    "model": model_name
                }
                output_list.append(output_instance)
        print(f"Total SCUs = {total_scus}")

        if output_list:
            if args.scu_fraction or args.scu_num:
                if args.scu_num:
                    num_keep = args.scu_num
                else:
                    num_keep = int(args.scu_fraction * len(output_list)) // len(included_models)
                for split_num in range(args.num_truncate_splits):
                    new_dset = []
                    for model_name in included_models:
                        model_dset = [x for x in output_list if x["model"] == model_name]
                        random.shuffle(model_dset)
                        new_dset.extend(model_dset[:num_keep])
                    random.shuffle(new_dset)
                    csv_dict_write(new_dset, f'{args.output_dir}/{filename}_trunc_num_{num_keep}_split_{split_num}.csv')
            else:
                random.shuffle(output_list)
                csv_dict_write(output_list, f'{args.output_dir}/{filename}.csv')

# main() function
if __name__ == '__main__':
    main(args)
