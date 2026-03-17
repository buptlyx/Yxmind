import torch

def _norm(x):
    return x*torch.rsqrt(x.pow(2).mean(-1,keepdim=True)+1e-5)
x=torch.tensor([[1,2,3],[4,5,6]]).float()
print(x)
print(_norm(x))