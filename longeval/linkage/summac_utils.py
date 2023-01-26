import functools
import torch
from longeval.linkage.summac.model_summac import SummaCZS

def summac_zero_setup(granularity="sentence", model_name="vitc"):
    model = SummaCZS(granularity=granularity, model_name=model_name)
    return functools.partial(summac_matrix_fn, model=model)

def summac_matrix_fn(first_list, second_list, model):
    return torch.Tensor([model.score([first_list[0] for _ in range(len(second_list))], second_list)['scores']])
