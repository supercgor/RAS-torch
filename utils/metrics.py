import torch
import numpy as np
from typing import Iterable

def l2_rel_error(preds, targs):
    return torch.norm(targs - preds) / torch.norm(targs)

@torch.jit.script
def mean_l2_relative_error(preds, targs):
    targs_norm = (targs ** 2).sum(dim=1).sqrt()
    
    return torch.where(targs_norm > 0, ((targs - preds) ** 2).sum(dim=1).sqrt() / targs_norm, torch.zeros_like(targs_norm))

class metStat():
    def __init__(self, value = None, reduction:str = "mean"):
        """To help you automatically find the mean or sum, which allows us to calculate something easily and consuming less space. 
        Args:
            reduction (str): should be 'mean', 'sum', 'max', 'min', 'none'
        """
        self._value = []
        if reduction == "mean":
            self._reduction = np.mean
        elif reduction == "sum":
            self._reduction = np.sum
        elif reduction == "max":
            self._reduction = np.max
        elif reduction == "min":
            self._reduction = np.min
        else:
            raise ValueError(f"reduction should be 'mean' or 'sum', but got {reduction}")

        if value is not None:
            self.add(value)
    
    def add(self, other):
        if isinstance(other, Iterable) or isinstance(other ,metStat):
            self.extend(other)
        else:
            self.append(other)
        
    def append(self, x: np.ndarray | torch.Tensor | float | int):
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        if isinstance(x, np.ndarray) and x.size != 1:
            raise ValueError(f"Only support scalar input, but got {x}")
        self._value.append(x)
    
    def extend(self, xs: Iterable):
        if isinstance(xs, (metStat, list, tuple)):
            self._value.extend(xs)
        elif isinstance(xs, np.ndarray):
            xs = xs.view(-1)
            self._value.extend(xs)
        elif isinstance(xs, torch.Tensor):
            xs = xs.detach().cpu().view(-1).numpy()
            self._value.extend(xs)
        elif isinstance(xs, Iterable):
            self._value.extend(xs)
        else:
            raise TypeError(f"{type(xs)} is not an iterable")
    
    def reset(self):
        self._value = []
    
    def calc(self) -> float:
        return self._reduction(self._value)
    
    def __call__(self, other):
        self.add(other)
    
    def __repr__(self):
        return f"{self._reduction(self._value)}"
        
    def __str__(self):
        return str(self._reduction(self._value))
    
    def __len__(self):
        return len(self._value)
    
    def __format__(self, code):
        return self._reduction(self._value).__format__(code)

    @property
    def value(self):
        return self._value
    