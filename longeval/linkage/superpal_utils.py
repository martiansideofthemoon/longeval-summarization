import functools
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def superpal_setup(granularity="sentence"):
    tokenizer = AutoTokenizer.from_pretrained("biu-nlp/superpal")
    model = AutoModelForSequenceClassification.from_pretrained("biu-nlp/superpal")
    model.cuda()
    model.eval()
    return functools.partial(superpal_matrix_fn, model=model, tokenizer=tokenizer)

def superpal_matrix_fn(first_list, second_list, model, tokenizer):
    BATCH_SIZE = 128
    all_align_scores = []
    for f1 in first_list:
        curr_align_scores = []
        for i in range(0, len(second_list), BATCH_SIZE):
            all_input_str = [f"{f1.strip()} </s><s> {f2.strip()}" for f2 in second_list[i:i + BATCH_SIZE]]
            with torch.inference_mode():
                tensors = tokenizer(all_input_str, return_tensors="pt", padding=True, truncation=True, max_length=512)
                tensors.to("cuda")
                align_scores = torch.nn.functional.softmax(model(**tensors).logits, dim=1)[:, 1]
                curr_align_scores.extend(align_scores)
        all_align_scores.append(curr_align_scores)
    return torch.Tensor(all_align_scores)
