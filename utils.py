"""
© Copyright 2021
RUIMIN MA
"""
import torch
import numpy as np


def Variable(tensor):
    if isinstance(tensor, np.ndarray):
        tensor = torch.from_numpy(tensor)
    if torch.cuda.is_available():
        return torch.autograd.Variable(tensor).cuda()
    return torch.autograd.Variable(tensor)


def decrease_learning_rate(optimizer, decrease_by=0.01):
    for param_group in optimizer.param_groups:
        param_group['lr'] *= (1 - decrease_by)