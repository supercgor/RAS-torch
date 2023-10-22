import torch
import numpy as np

def non_batched_fn(x):
    return x[0]

class it_loader(object):
    def __init__(self, loader, iters):
        self.loader = loader
        self.iters = iters
        self.dataiter = iter(self.loader)
    
    def __iter__(self):
        for i in range(self.iters):
            try:
                yield next(self.dataiter)
            except StopIteration:
                self.dataiter = iter(self.loader)
                yield next(self.dataiter)
    
    def __len__(self):
        return self.iters

class BidxSampler(torch.utils.data.Sampler):
    def __init__(self, dts: torch.utils.data.Dataset, batch_size: int, drop_last: bool = False, shuffle: bool = False):
        self.dts = dts
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
    
    def __iter__(self):
        if self.shuffle:
            idxs = torch.randperm(len(self.dts))
        else:
            idxs = torch.arange(len(self.dts))
        idxs = torch.split(idxs, self.batch_size)
        if self.drop_last and len(idxs[-1]) < self.batch_size:
            idxs = idxs[:-1]
        return iter(idxs)

def BidxDataLoader(dataset, batch_size = 1, iters = None, shuffle = False, drop_last = False, num_workers = 0, pin_memory = True, *args, **kwargs):
    """
    This is an extension of torch.utils.data.DataLoader that allows for batched-indexing of the data and repeats iteration if iters is specified.
    This will significantly speed up data loading when batch_size is big and iters is much bigger than `len(dataset)//batch_size`.
    

    Args:
        dataset (_type_): _description_
        batch_size (int, optional): _description_. Defaults to 1.
        iters (_type_, optional): _description_. Defaults to None.
        shuffle (bool, optional): _description_. Defaults to False.
        drop_last (bool, optional): _description_. Defaults to False.
        num_workers (int, optional): _description_. Defaults to 0.

    Returns:
        _type_: _description_
    """
    # if shuffle:
    #     seq_sampler = torch.utils.data.RandomSampler(dataset)
    # else:
    #     seq_sampler = torch.utils.data.SequentialSampler(dataset)
        
    # sampler = torch.utils.data.BatchSampler(seq_sampler, batch_size=batch_size, drop_last= drop_last)
    
    sampler = BidxSampler(dataset, batch_size=batch_size, drop_last= drop_last, shuffle=shuffle)
    
    if iters is None:
        return torch.utils.data.DataLoader(dataset, num_workers=num_workers, sampler=sampler, collate_fn=non_batched_fn, pin_memory=pin_memory, *args, **kwargs)
    else:
        return torch.utils.data.DataLoader(dataset, num_workers=num_workers, sampler=it_loader(sampler, iters), collate_fn=non_batched_fn, pin_memory=pin_memory, *args, **kwargs)
            
class Cartesian(torch.utils.data.Dataset):
    def __init__(self, funcs: np.ndarray, grids: np.ndarray, truths: np.ndarray | None = None):
        self.funcs = torch.as_tensor(funcs, dtype = torch.float) # N1, M
        self.grids = torch.as_tensor(grids, dtype = torch.float)  # N2, D1
        self.truths = torch.as_tensor(truths, dtype = torch.float) if truths is not None else None # N1, N2, D2

    def __len__(self):
        return len(self.grids) * len(self.funcs)

    def __getitem__(self, idx: int | tuple[int]):
        func_idx = idx // len(self.grids)
        grid_idx = idx % len(self.grids)
        truths = self.truths[func_idx, grid_idx] if self.truths is not None else None
        if truths is None:
            return self.funcs[func_idx], self.grids[grid_idx]
        else:
            return self.funcs[func_idx], self.grids[grid_idx], truths
    
    def len_funcs(self):
        return len(self.funcs)
    
    def len_grids(self):
        return len(self.grids)
        
    def eval_funcs(self, idx: int):
        assert idx < len(self.funcs), "idx out of range"
        return self.funcs[idx].repeat(len(self.grids), 1), self.grids

    def add_funcs(self, func: np.ndarray, truth: np.ndarray | None = None):
        if self.truths is None:
            assert truth is None, "truth should be None"
        else:
            assert truth is not None, "truth should not be None"
        self.funcs = torch.cat([self.funcs, torch.as_tensor(func, dtype = torch.float)], dim=0)
        if truth is not None:
            self.truths = torch.cat([self.truths, torch.as_tensor(truth, dtype = torch.float)], dim=0)
            
# class Cartesian(torch.utils.data.Dataset):
#     def __init__(self, funcs: np.ndarray, grids: np.ndarray, truths: np.ndarray | None = None):
#         self.funcs = torch.as_tensor(funcs, dtype = torch.float).repeat(grids.shape[0],1) # N1, M
#         self.grids = torch.as_tensor(grids, dtype = torch.float).repeat_interleave(funcs.shape[0], dim=0)  # N2, D1
#         self.truths = torch.as_tensor(truths, dtype = torch.float).flatten(end_dim=-2) if truths is not None else None # N1, N2, D2


#     def __len__(self):
#         return len(self.grids)

#     def __getitem__(self, idx: int | tuple[int]):
#         return self.funcs[idx], self.grids[idx], self.truths[idx]
    
#     def len_funcs(self):
#         return len(self.funcs)
    
#     def len_grids(self):
#         return len(self.grids)
        
#     def eval_funcs(self, idx: int):
#         assert idx < len(self.funcs), "idx out of range"
#         return self.funcs[idx].repeat(len(self.grids), 1), self.grids

#     def add_funcs(self, func: np.ndarray, truth: np.ndarray | None = None):
#         if self.truths is None:
#             assert truth is None, "truth should be None"
#         else:
#             assert truth is not None, "truth should not be None"
#         self.funcs = torch.cat([self.funcs, torch.as_tensor(func, dtype = torch.float)], dim=0)
#         if truth is not None:
#             self.truths = torch.cat([self.truths, torch.as_tensor(truth, dtype = torch.float)], dim=0)