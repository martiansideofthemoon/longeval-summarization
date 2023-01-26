import functools
import time
import torch

from torch.utils.data import DataLoader

from longeval.linkage.longchecker.model import LongCheckerModel
from longeval.linkage.longchecker.data import get_tokenizer, LongCheckerDataset, Collator


def multivers_setup(model_type="multivers_fever_sci---50"):
    model_type, doc_window = model_type.split("---")
    print(f"loading multivers model of type {model_type} with a {doc_window} sentence window")
    model = LongCheckerModel.load_from_checkpoint(checkpoint_path=f"models/multivers_models/{model_type}.ckpt")
    # If not predicting NEI, set the model label threshold to 0.
    model.label_threshold = 0.0
    model.to("cuda")
    model.eval()
    model.freeze()
    tokenizer = get_tokenizer()
    return functools.partial(multivers_matrix_fn, model=model, tokenizer=tokenizer, doc_window=int(doc_window))

def multivers_matrix_fn(first_list, second_list, tokenizer, model, doc_window):
    rationale_probs = []
    preprocess = 0.0
    infer = 0.0
    all_data = []
    for i in range(0, len(second_list), doc_window):
        sec_list = second_list[i:i + doc_window]
        res = convert_multivers_format(claims=first_list,
                                       document_sents=sec_list)
        all_data.extend(res)

    dataset_obj = LongCheckerDataset(all_data, tokenizer)
    collator = Collator(tokenizer)
    dataloader = DataLoader(dataset_obj,
                            num_workers=1,
                            batch_size=1,
                            collate_fn=collator,
                            shuffle=False,
                            pin_memory=True)

    for batch in dataloader:
        preds_batch = model.predict(batch, force_rationale=True)
        for x in preds_batch:
            rationale_probs.extend(x['rationale_probs'].tolist())

    return torch.Tensor([rationale_probs])


def convert_multivers_format(claims, document_sents):
    res = []
    for i, claim in enumerate(claims):
        to_tensorize = {"claim": claim,
                        "sentences": document_sents,
                        "title": None}
        res.append({"claim_id": i, "abstract_id": 0, "to_tensorize": to_tensorize})
    return res
