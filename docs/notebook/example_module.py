import torch

def example_function() -> torch.Tensor:
    t1 = torch.tensor(list(range(10000000))).cuda()
    t2 = torch.tensor(list(range(10000000))).cuda()
    return t1 * t2
