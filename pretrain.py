import numpy as np
import torch
import os

import hydra
import utils
from dataset import Cartesian, BidxDataLoader

class Trainer():
    def __init__(self, cfg, net, train_dataset, test_dataset, optimizer, device, log):
        self.cfg = cfg
        self.work_dir = hydra.core.hydra_config.HydraConfig.get()['runtime']['output_dir']
        self.device = device
        self.net = net.to(device)
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.optimizer = optimizer
        self.log = log
        
        self.train_loader = BidxDataLoader(self.train_dataset, batch_size=cfg.pde.train.batch_size, iters=cfg.pde.train.iters, shuffle=True, num_workers = 3)
        self.test_loader = BidxDataLoader(self.test_dataset, batch_size=cfg.pde.train.batch_size, shuffle=False, num_workers = 3)

        self.criterion = torch.nn.MSELoss()
        
        self.train_loss = utils.metrics.metStat(reduction="min")
        self.test_loss = utils.metrics.metStat(reduction="min")
        self.test_met = utils.metrics.metStat(reduction="min")
        
        self.total_iter = 0
        self.best = np.inf
        self.save_paths = []
            
    def train(self, log_every = 1000, iters = None):
        if iters is not None:
            train_loader = BidxDataLoader(self.train_dataset, batch_size=self.cfg.pde.train.batch_size, iters=iters, shuffle=True, num_workers = 3)
        else:
            train_loader = self.train_loader
            
        lossStat = utils.metrics.metStat()
        metStat = utils.metrics.metStat()
        
        test_loss, test_met = self.test()
        
        for i, (funcs, grids, targs) in enumerate(train_loader, 1):
            funcs, grids, targs = funcs.to(self.device, non_blocking=True), grids.to(self.device, non_blocking=True), targs.to(self.device, non_blocking=True)
            preds = self.net((funcs, grids))
            loss = self.criterion(preds, targs)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            l2rel = utils.metrics.mean_l2_relative_error(preds, targs)
            lossStat.add(loss)
            metStat.add(l2rel)
            self.total_iter += 1
            
            if i % log_every == 0 or i == 1 or i == len(train_loader):
                if i == 1:
                    idu = 0
                else:
                    test_loss, test_met = self.test()
                    idu = i
                self.log.info(f"Iter {idu:6d}/{self.cfg.pde.train.iters:6d} | train loss {lossStat:.2e} | test loss {test_loss:.2e} | test met {test_met:.2e}")
                self.train_loss.add(lossStat.calc())
                self.test_loss.add(test_loss.calc())
                self.test_met.add(test_met.calc())
                lossStat.reset()
                metStat.reset()
                self.save(test_met.calc())
        
        self.save(test_met.calc())
        return self.train_loss.calc(), self.test_loss.calc(), self.test_met.calc()
            
    @torch.no_grad()
    def test(self):
        self.net.eval()
        test_loss = utils.metrics.metStat()
        test_met = utils.metrics.metStat()
        for i, (funcs, grids, targs) in enumerate(self.test_loader):
            funcs, grids, targs = funcs.to(self.device, non_blocking=True), grids.to(self.device, non_blocking=True), targs.to(self.device, non_blocking=True)
            preds = self.net((funcs, grids))
            loss = self.criterion(preds, targs)
            l2rel = utils.metrics.mean_l2_relative_error(preds, targs)
            test_loss.add(loss)
            test_met.add(l2rel)
        self.net.train()
        return test_loss, test_met
    
    def save(self, met):
        if met < self.best:
            self.best = met
        path = f"{self.work_dir}/{self.total_iter}_{met}.pkl"
        if len(self.save_paths) >= self.cfg.pde.train.max_save:
            os.remove(self.save_paths.pop(0))
        utils.model_save(self.net, path)
        self.save_paths.append(path)
    
def prepare(cfg):
    net = utils.get_model(cfg.pde.model.name, cfg.pde.model.params, cfg.pde.model.input_transform, cfg.pde.model.output_transform)
    if cfg.pde.datasets.pretrain_path is not None:
        match_list = utils.model_load(net, cfg.pde.datasets.pretrain_path)
        
    train_data = np.load(cfg.pde.datasets.train_path)
    test_data = np.load(cfg.pde.datasets.test_path)
    train_data = Cartesian(train_data['funcs'], train_data['grids'].reshape(-1, 2), train_data['out'].reshape(train_data['out'].shape[0], -1, 1))
    test_data = Cartesian(test_data['funcs'], test_data['grids'].reshape(-1,2), test_data['out'].reshape(test_data['out'].shape[0], -1, 1))
    
    optimizer = torch.optim.Adam(net.parameters(), lr=cfg.optimizer.lr, weight_decay=cfg.optimizer.weight_decay)
    
    log = utils.get_logger("Rank 0")
    
    return net, train_data, test_data, optimizer, log

def spawn_data(path, size, func_space, solver):
    funcs = []
    out = []
    for i in range(size):
        print(f"Generating {i}/{size}", end = "\r")
        vx = func_space.eval_batch(func_space.random(1), np.linspace(0, 1, 101)[:, None])[0]
        funcs.append(vx)
        uxt, grids = solver(vx)
        out.append(uxt)
    funcs = np.stack(funcs, axis = 0, dtype = np.float32)
    out = np.stack(out, axis = 0, dtype = np.float32)
    np.savez(path, funcs = funcs, out = out, grids = grids)

@hydra.main(config_path="config", config_name="config", version_base=None)
def main(cfg):
    solver = utils.get_solver(cfg.pde.solver.name, cfg.pde.solver.params)
    space = utils.get_space(cfg.funcspace.name, cfg.funcspace.params)
    datadir = cfg.pde.datasets.workdir
    os.makedirs(datadir, exist_ok=True)
    train_path = cfg.pde.datasets.train_path or f"{cfg.pde.datasets.workdir}/{utils.get_space_name(space)}_{cfg.pde.train.init_train_size}_t.npz"
    test_path = cfg.pde.datasets.test_path or f"{cfg.pde.datasets.workdir}/{utils.get_space_name(space)}_{cfg.pde.train.test_size}_v.npz"
    
    if not os.path.exists(train_path):
        print("Making train data")
        spawn_data(train_path, cfg.pde.train.init_train_size, space, solver)
    if not os.path.exists(test_path):
        print("Making test data")
        spawn_data(test_path, cfg.pde.train.test_size, space, solver)
    
    cfg.pde.datasets.train_path = train_path
    cfg.pde.datasets.test_path = test_path
    
    if torch.cuda.is_available():
        device = torch.device("cuda") 
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
        
    net, train_dataset, test_dataset, optimizer, log = prepare(cfg)
    trainer = Trainer(cfg, net, train_dataset, test_dataset, optimizer, device, log)
    trainer.train()

if __name__ == "__main__":
    main()