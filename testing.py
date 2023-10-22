import time
import torch
import numpy as np
from dataset import Cartesian, BidxDataLoader

f = np.random.normal(size=(10, 101))
grid = np.asarray(np.meshgrid(np.linspace(0, 1, 101), np.linspace(0, 1, 201))).T
grid = grid.reshape(-1, 2)
print(grid)
print(f.shape, grid.shape)

# b = torch.utils.data.SubsetRandomSampler(np.arange(1000))

dataset = Cartesian(f, grid)
loader = BidxDataLoader(dataset, batch_size=10000, iters=50000, shuffle=True, num_workers=0)
dataset.add_funcs(f)
t = time.time()
for funcs, grids in loader:
    # print(funcs.shape, grids.shape)
    continue
print(time.time() - t)