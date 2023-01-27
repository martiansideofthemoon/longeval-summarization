import torch
import os

from longeval.linkage.similarity.sim_models import WordAveraging
from longeval.linkage.similarity.sim_utils import Example
from nltk.tokenize import TreebankWordTokenizer
import sentencepiece as spm

tok = TreebankWordTokenizer()

if os.path.exists('longeval/linkage/similarity/sim/sim.pt'):
    model = torch.load('longeval/linkage/similarity/sim/sim.pt')
    state_dict = model['state_dict']
    vocab_words = model['vocab_words']
    args = model['args']
    # turn off gpu
    model = WordAveraging(args, vocab_words)
    model.load_state_dict(state_dict, strict=True)
    sp = spm.SentencePieceProcessor()
    sp.Load('longeval/linkage/similarity/sim/sim.sp.30k.model')
    model.eval()

def make_example(sentence, model):
    sentence = sentence.lower()
    sentence = " ".join(tok.tokenize(sentence))
    sentence = sp.EncodeAsPieces(sentence)
    wp1 = Example(" ".join(sentence))
    wp1.populate_embeddings(model.vocab)
    return wp1

def find_similarity(s1, s2):
    with torch.no_grad():
        s1 = [make_example(x, model) for x in s1]
        s2 = [make_example(x, model) for x in s2]
        wx1, wl1, wm1 = model.torchify_batch(s1)
        wx2, wl2, wm2 = model.torchify_batch(s2)
        BATCH_SIZE = 512
        all_scores = []
        for i in range(0, len(wx1), BATCH_SIZE):
            scores = model.scoring_function(wx1[i:i + BATCH_SIZE], wm1[i:i + BATCH_SIZE], wl1[i:i + BATCH_SIZE],
                                            wx2[i:i + BATCH_SIZE], wm2[i:i + BATCH_SIZE], wl2[i:i + BATCH_SIZE])
            all_scores.extend([x.item() for x in scores])
        return all_scores

def find_similarity_matrix(s1, s2, vecs1=None, vecs2=None, return_cache=False):
    with torch.no_grad():
        if vecs1 is None:
            s1 = [make_example(x, model) for x in s1]
            wx1, wl1, wm1 = model.torchify_batch(s1)
        else:
            assert len(vecs1) == len(s1)

        if vecs2 is None:
            s2 = [make_example(x, model) for x in s2]
            wx2, wl2, wm2 = model.torchify_batch(s2)
        else:
            assert len(vecs2) == len(s2)

        BATCH_SIZE = 2000

        if vecs1 is None:
            vecs1 = []
            for i in range(0, len(wx1), BATCH_SIZE):
                curr_vecs1 = model.encode(idxs=wx1[i:i + BATCH_SIZE],
                                        mask=wm1[i:i + BATCH_SIZE],
                                        lengths=wl1[i:i + BATCH_SIZE])
                vecs1.append(curr_vecs1)
            vecs1 = torch.cat(vecs1)

        if vecs2 is None:
            vecs2 = []
            for i in range(0, len(wx2), BATCH_SIZE):
                curr_vecs2 = model.encode(idxs=wx2[i:i + BATCH_SIZE],
                                        mask=wm2[i:i + BATCH_SIZE],
                                        lengths=wl2[i:i + BATCH_SIZE])
                vecs2.append(curr_vecs2)
            vecs2 = torch.cat(vecs2)

        dot_product = torch.matmul(vecs1, vecs2.t())

        vecs1_norm = torch.norm(vecs1, dim=1, keepdim=True)
        vecs2_norm = torch.norm(vecs2, dim=1, keepdim=True)
        norm_product = torch.matmul(vecs1_norm, vecs2_norm.t())

    if return_cache:
        return torch.div(dot_product, norm_product), vecs1, vecs2
    else:
        return torch.div(dot_product, norm_product)

def encode_text(s1):
    with torch.no_grad():
        s1 = [make_example(x, model) for x in s1]
        wx1, wl1, wm1 = model.torchify_batch(s1)
        vecs1 = model.encode(idxs=wx1, mask=wm1, lengths=wl1)
        return vecs1
