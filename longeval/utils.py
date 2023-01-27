"""
Data augmentation (transformations) operations used to generate
synthetic training data for the `FactCC` and `FactCCX` models.

Adapted from https://github.com/salesforce/factCC/blob/master/data_generation/augmentation_ops.py
"""

from collections import Counter
import numpy as np
from longeval.linkage.utils import f1_score
from longeval.preprocessing_utils import get_sents
from longeval.linkage.similarity.test_sim import find_similarity_matrix
import torch

LABEL_MAP = {True: "CORRECT", False: "INCORRECT"}

def token_counter(dataset, evidence_key="gold_evidence"):
    total_tokens = 0
    for dd in dataset:
        total_tokens += len(dd[evidence_key].split()) + len(dd["transformed_scu"].split())
    return total_tokens


def get_most_common(lst):
    counts = Counter(lst)
    sorted_counts = sorted([(k, v) for k, v in counts.items()], key=lambda x: x[1], reverse=True)
    return {
        "element": sorted_counts[0][0],
        "frequency": sorted_counts[0][1],
        "unique": len(sorted_counts) == 1 or (sorted_counts[0][1] > sorted_counts[1][1]),
    }

def get_best_span_match(sentence, scu):
    sent_tokens = sentence.split()
    all_f1 = []
    all_strs = []
    for i in range(0, len(sent_tokens) - 1):
        for j in range(i + 1, len(sent_tokens)):
            curr_str = " ".join(sent_tokens[i:j])
            all_f1.append(f1_score(curr_str, scu)[2])
            all_strs.append(curr_str)
    return all_strs[np.argmax(all_f1)]

def align_ws(old_token, new_token):
    # Align trailing whitespaces between tokens
    if old_token[-1] == new_token[-1] == " ":
        return new_token
    elif old_token[-1] == " ":
        return new_token + " "
    elif new_token[-1] == " ":
        return new_token[:-1]
    else:
        return new_token


def get_gold_evidence(best_src, doc_sents, style="highlight", num_prefix_sents=None, num_evidence=None):
    best_src_sents = get_sents(best_src)
    best_src_matches = find_similarity_matrix(doc_sents, best_src_sents).argmax(dim=0).tolist()
    if style == "snippet":
        return build_evidence_string(best_src_matches, doc_sents, prefix_sents=num_prefix_sents, num_evidence=num_evidence)
    elif style == "highlight":
        return build_highlight_doc(best_src_matches, doc_sents)


def build_highlight_doc(best_src_matches, doc_sents):
    """Highlight the alignments in the source document."""
    src_matches_done = {x: False for x in best_src_matches}
    evidence_num = 0
    for x in best_src_matches:
        if src_matches_done[x]:
            continue
        # find largest continguous span of highlight
        left_pos = x
        while left_pos in src_matches_done:
            src_matches_done[left_pos] = True
            left_pos = left_pos - 1
        doc_sents[left_pos + 1] = f"<span id='evidence-{evidence_num}' style='background-color: #ffd500'>{doc_sents[left_pos + 1]}"

        right_pos = x
        while right_pos in src_matches_done:
            src_matches_done[right_pos] = True
            right_pos = right_pos + 1
        doc_sents[right_pos - 1] = f"{doc_sents[right_pos - 1]}</span>"

        evidence_num += 1
    return "<br />".join(doc_sents), evidence_num, [doc_sents[x] for x in best_src_matches]


def get_predicted_evidence(claim_scu, doc_sents, linker_matrix_fn, num_prefix_sents=None, num_evidence=5, style="highlight", score_threshold=0.3):
    """Align a claim-scu pair with a document, and return a string with the evidence highlighted."""
    scu_matches = linker_matrix_fn([claim_scu], doc_sents)[0]
    if style == "snippet":
        scu_matches = torch.sort(scu_matches, descending=True).indices[:2 * num_evidence].tolist()
        return build_evidence_string(scu_matches, doc_sents, prefix_sents=num_prefix_sents, num_evidence=num_evidence)
    elif style == "highlight":
        scu_matches  = torch.sort(scu_matches, descending=True)
        top_indices, top_values = scu_matches.indices[:num_evidence].tolist(), scu_matches.values[:num_evidence].tolist()
        top_indices = [x for x, y in zip(top_indices, top_values) if y > score_threshold]
        return build_highlight_doc(top_indices, doc_sents)


def build_evidence_string(best_src_matches, src_sents, prefix_sents=2, num_evidence=3):
    best_src_matches_plus_prefix = []
    for x in best_src_matches:
        best_src_matches_plus_prefix.extend([xx for xx in range(x - prefix_sents, x + 1) if xx >= 0])
    best_src_matches_plus_prefix = sorted(list(set(best_src_matches_plus_prefix)))

    prev_sent = None
    evidence_str = ""
    total_evidence = 0
    for x in best_src_matches_plus_prefix:
        if prev_sent is None:
            evidence_str += src_sents[x]
        elif x == prev_sent + 1:
            evidence_str += f" {src_sents[x]}"
        else:
            total_evidence += 1
            if total_evidence >= num_evidence:
                break
            evidence_str += f"<br /> ... <br />{src_sents[x]}"
        prev_sent = x

    return evidence_str

def fix_quote_issues(text):
    """Try to get the quotes in the same <br> separated line as the sentence they are in."""
    if "<br />" in text:
        split_token = "<br />"
    else:
        split_token = "\n"
    text_units = text.split(split_token)

    new_units = []
    for i in range(len(text_units)):
        if text_units[i].strip() == "\"":
            new_units[-1] += "\""
        else:
            new_units.append(text_units[i])


    for i, unit in enumerate(text_units):
        non_space_chars = [x for x in unit if x != ' ']
        # check if first/last two non-space characters are "
        if len(non_space_chars) <= 2:
            continue
        if non_space_chars[-2] == "\"" and non_space_chars[-1] == "\"":
            text_units[i] = text_units[i].strip()[:-1]
            text_units[i + 1] = "\"" + text_units[i + 1]

    return split_token.join(text_units)
