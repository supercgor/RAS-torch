import numpy as np
import time
import torch
from torch import Tensor
import os
import hydra
from utils.funcspace import getspace

@hydra.main(config_path = "conf", config_name = "config", version_base = None)
def main(cfg):
    space = getspace(cfg.funcspace.name, cfg.funcspace.params)
    

if __name__ == "__main__":
    pass