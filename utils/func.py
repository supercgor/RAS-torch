import torch
from torch import nn

from deepxde.data.function_spaces import FunctionSpace
from typing import overload
from functools import partial

import model
from omegaconf import OmegaConf
from dataset import solver
import logging

from . import funcspace as fs

def get_logger(name):
    logger = logging.getLogger(name)
    return logger

def get_model(name, params, input_transform: dict | None = None, output_transform: dict | None = None):
    net = getattr(model, name)(**params)
    if input_transform is not None:
        net.apply_feature_transform(get_transform(**input_transform))
    if output_transform is not None:
        net.apply_output_transform(get_transform(**output_transform))
    return net
        
def get_transform(name, params):
    return getattr(model, name)(**params)

def get_solver(name, params):
    return partial(getattr(solver, name), **params)

@overload
def get_space(name: str, params: dict) -> FunctionSpace: ...

@overload
def get_space(name: str, params: list[dict]) -> FunctionSpace: ...

def get_space(name, params = None):
    if OmegaConf.is_dict(params):
        space = getattr(fs, name)(**params)
    elif OmegaConf.is_list(params):
        space = [getattr(fs, space_dict['name'])(**space_dict['params']) for space_dict in params]
        space = getattr(fs, name)(*space)
    return space

def get_space_name(funcspace) -> str:
    if isinstance(funcspace, fs.UnionSpace):
        return "+".join([get_space_name(space)[:3] for space in funcspace.space])
    else:
        return funcspace.__class__.__name__
    
def model_save(module: nn.Module, path: str):
        try:
            state_dict = module.module.state_dict()
        except AttributeError:
            state_dict = module.state_dict()
        torch.save(state_dict, path)

def model_load(module: nn.Module, path: str, strict = False) -> list[str]:
    state_dict = module.state_dict()
    param_names = list(state_dict.keys())
    pretrained_state_dict = torch.load(path, map_location = 'cpu')
    pretrained_param_names = list(pretrained_state_dict.keys())
    match_list = []
    for i, param in enumerate(pretrained_param_names):
        if i == len(param_names):
            break
        if param == param_names[i]:
            match_list.append(param)
            state_dict[param] = pretrained_state_dict[param]
        else:
            break
    if strict:
        module.load_state_dict(state_dict)
    else:
        try:
            module.load_state_dict(state_dict)
        except RuntimeError:
            pass
    return match_list