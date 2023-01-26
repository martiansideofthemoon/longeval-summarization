## LongEval: Guidelines for Human Evaluation of Faithfulness in Long-form Summarization

This is the official repository for our EACL 2023 paper, LongEval: Guidelines for Human Evaluation of Faithfulness in Long-form Summarization. LongEval is a set of three guidelines to help manually evaluate factuality of long summaries. This repository provides the annotation data we collected, along with a command-line tool to prepare data in a format compatible with our annotation guidelines.

### Setup

### Crowdsourcing Templates

Our FINE-grained crowdsourcing interface can be found in [`templates/mturk_sandbox.html`](templates/mturk_sandbox.html). To use this interface, login to [AMT Sandbox](https://requestersandbox.mturk.com) and create a new project. Add this HTML code to the "Design Layout" tab. We also used this short instruction [video](https://youtu.be/LbZPo0AmXYI) to familiarize our FINE-grained annotators with the interface.

Note that while we used AMT Sandbox to host our annotation interface, all our annotators were hired on Upwork only - no MTurk crowdworkers were used in our experiments. We provided Upwork annotations with the AMT Sandbox URL, and requested them to make an account on the interface. All payments were processed through Upwork only.

### Citation

If you found our paper or repository useful, please cite us using:

```

```