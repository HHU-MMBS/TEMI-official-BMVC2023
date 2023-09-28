import torch

@torch.jit.script
def sim_weight(p1, p2, gamma: float = 1.):
    return (p1 * p2).pow(gamma).sum(dim=-1)

@torch.jit.script
def beta_mi(p1, p2, pk, beta: float = 1., clip_min: float = -torch.inf):
    beta_emi = (((p1 * p2)**beta) / pk).sum(dim=-1)
    beta_pmi = beta_emi.log().clamp(min=clip_min)
    return -beta_pmi
