## LongEval: Guidelines for Human Evaluation of Faithfulness in Long-form Summarization

This is the official repository for our EACL 2023 paper, [LongEval: Guidelines for Human Evaluation of Faithfulness in Long-form Summarization](https://martiansideofthemoon.github.io/assets/longeval.pdf). LongEval is a set of three guidelines to help manually evaluate factuality of long summaries. This repository provides the annotation data we collected, along with a command-line tool to prepare data in a format compatible with our annotation guidelines.

### Setup

```
# from PyPI

python3.7 -m virtualenv longeval-venv
source longeval-venv/bin/activate
pip install longeval
python -m spacy download en_core_web_lg

# from source

python3.7 -m virtualenv longeval-venv
source longeval-venv/bin/activate
git clone https://github.com/martiansideofthemoon/longeval-summarization
cd longeval-summarization
pip install --editable .
python -m spacy download en_core_web_lg
```

**Other setup**

Download the SIM model from [here](https://drive.google.com/drive/folders/1lBN2nbzxtpqbPUyeURtzt0k1kBY6u6Mj?usp=share_link) if you are interested in using the non-default linker from [Wieting et al. 2019](https://aclanthology.org/P19-1427/). Place both files in `longeval/linkage/similarity/sim`.

### Crowdsourcing Templates

Our FINE-grained crowdsourcing interface can be found in [`templates/fine_sandbox_interface.html`](templates/fine_sandbox_interface.html). To use this interface, login to [AMT Sandbox](https://requestersandbox.mturk.com) and create a new project. Add this HTML code to the "Design Layout" tab. We also used this short instruction [video](https://youtu.be/LbZPo0AmXYI) to familiarize our FINE-grained annotators with the interface. Instructions to Upworkers for COARSE-grained evaluations on PubMed are provided in [`templates/coarse_instructions.md`](templates/coarse_instructions.md).

Note that while we used AMT Sandbox to host our annotation interface, all our annotators were hired on Upwork only - no MTurk crowdworkers were used in our experiments. We provided Upwork annotations with the AMT Sandbox URL, and requested them to make an account on the interface. All payments were processed through Upwork only.

### Preprocessing data

To get your summarization data in a format compatible with our templates,

```
python -m longeval.prepare_summaries \
    --src_file data/pubmed/beam_3.jsonl \
    --scu_fraction 0.5 \
    --num_articles 3 \
    --num_truncate_splits 3 \
    --linking_algorithm superpal \
    --output_dir outputs/pubmed_beam_3 \
    --included_models "bigbird_pegasus;longt5;human"
```

Each source article produces a different file containing all the summaries for that particular article. Make sure the input file is a JSONL file, with the `"article"` key representing the source document and one key for each model's summary. See [`data/pubmed/beam_3.jsonl`](data/pubmed/beam_3.jsonl) for an example.

### Citation

If you found our paper or repository useful, please cite us using:

```
@inproceedings{longeval23,
author={Kalpesh Krishna and Erin Bransom and Bailey Kuehl and Mohit Iyyer and Pradeep Dasigi and Arman Cohan and Kyle Lo},
booktitle = {European Chapter of the Association for Computational Linguistics},
Year = "2023",
Title={LongEval: Guidelines for Human Evaluation of Faithfulness in Long-form Summarization},
}
```