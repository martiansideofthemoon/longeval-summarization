## LongEval: Guidelines for Human Evaluation of Faithfulness in Long-form Summarization

[![made-with-python](https://img.shields.io/badge/Made%20with-Python-red.svg)](#python)
[![arxiv](https://img.shields.io/badge/arXiv-2301.13298-b31b1b.svg)](https://arxiv.org/abs/2301.13298)
[![PyPI version longeval](https://badge.fury.io/py/longeval.svg)](https://pypi.python.org/pypi/longeval/) [![License: Apache 2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Downloads](https://pepy.tech/badge/longeval)](https://pepy.tech/project/longeval)

This is the official repository for our EACL 2023 paper, [LongEval: Guidelines for Human Evaluation of Faithfulness in Long-form Summarization](https://arxiv.org/abs/2301.13298). LongEval is a set of three guidelines to help manually evaluate factuality of long summaries. This repository provides the annotation data we collected, along with a command-line tool to prepare data in a format compatible with our annotation guidelines.

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

Additionally, download the SIM model from [here](https://drive.google.com/drive/folders/1lBN2nbzxtpqbPUyeURtzt0k1kBY6u6Mj?usp=share_link) if you are interested in using the non-default linker from [Wieting et al. 2019](https://aclanthology.org/P19-1427/). Place both files in `longeval/linkage/similarity/sim`.

To test the implementation works correctly, run the experiment to evaluate SuperPAL's linking abilities (Table 4 in Section 3.3):

```
python -m longeval.evaluate_linkers --linking_algorithm superpal

# Expected output (takes 5-6 min to run)
Best match:
Recall@3 = 0.6080 (76 / 125)
Recall@5 = 0.6800 (85 / 125)
Recall@10 = 0.7680 (96 / 125)
```

### LongEval crowdsourcing templates

Our FINE-grained crowdsourcing interface can be found in [`templates/fine_sandbox_interface.html`](templates/fine_sandbox_interface.html). To use this interface, login to [AMT Sandbox](https://requestersandbox.mturk.com) and create a new project. Add this HTML code to the "Design Layout" tab. We also used this short instruction [video](https://youtu.be/LbZPo0AmXYI) to familiarize our FINE-grained annotators with the interface. Instructions to Upworkers for COARSE-grained evaluations on PubMed are provided in [`templates/coarse_instructions.md`](templates/coarse_instructions.md).

Note that while we used AMT Sandbox to host our annotation interface, all our annotators were hired on Upwork only - no MTurk crowdworkers were used in our experiments. We provided Upwork annotations with the AMT Sandbox URL, and requested them to make an account on the interface. All payments were processed through Upwork only.

#### Preprocessing summarization data to LongEval format

To get your summarization data in a format compatible with our templates,

```
python -m longeval.prepare_summaries \
    --src_file data/pubmed_summaries/beam_3.jsonl \
    --scu_fraction 0.5 \
    --num_articles 3 \
    --num_truncate_splits 3 \
    --linking_algorithm superpal \
    --output_dir outputs/pubmed_beam_3 \
    --included_models "bigbird_pegasus;longt5;human"
```

Each source article produces a different file containing all the summaries for that particular article. Make sure the input file is a JSONL file, with the `"article"` key representing the source document and one key for each model's summary. See [`data/pubmed_summaries/beam_3.jsonl`](data/pubmed_summaries/beam_3.jsonl) for an example.

### Human judgments we collected using LongEval

**FINE/COARSE annotations**

All the annotations can be found in this [Google Drive link](https://drive.google.com/drive/folders/1KxLGYp0t7iZ5J5uTVByhN37pL4jqqXVK?usp=sharing). After downloading the data, place it in `data`. The annotations follow the AMT / LabelStudio formats, which may appear a bit complex. Functions to read-in the data are provided in [`longeval/metrics_corr_confidence_pubmed.py`](metrics_corr_confidence_pubmed.py).

Running metric correlation scripts on this data (Figure 2) needs a few additional setup steps which we haven't included in the PyPI package due to dependency issues.

1. Setup BLEURT using the instructions here: https://github.com/google-research/bleurt

2. Setup SacreROUGE: https://github.com/danieldeutsch/sacrerouge, or simply run `pip install sacrerouge`

3. Upgrade HuggingFace Hub since SacreROUGE downgrades it to an incompatible version.

```
pip install --upgrade huggingface-hub
```

After this setup simply run the following to reproduce Figure 2:

```
python -m longeval.metrics_corr_confidence_squality
python -m longeval.metrics_corr_confidence_pubmed
```

**SQuALITY source-summary alignments**

Finally, our hand-annotated source-summary alignment data in SQuALITY can be found in [`data/squality_alignment/data.json`](data/squality_alignment/data.json). To test linking algorithms on this run:

```
python -m longeval.evaluate_linkers --linking_algorithm superpal
```

You can set `--linking_algorithm` to any of the algorithms in the `get_linking_fn` function written in [`longeval/linkage/utils.py`](longeval/linkage/utils.py).

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
