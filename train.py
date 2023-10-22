import numpy as np
import torch
import os
import deepxde as dde

import hydra
from omegaconf import OmegaConf
import utils
from dataset import Cartesian, BidxDataLoader

class Trainer():
    def __init__(self, cfg, net, train_dataset, test_dataset, optimizer, log):
        self.cfg = cfg
        self.work_dir = hydra.core.hydra_config.HydraConfig.get()['runtime']['output_dir']
        self.net = net
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.optimizer = optimizer
        self.log = log
        
        self.train_loader = BidxDataLoader(self.train_dataset, batch_size=cfg.pde.train.batch_size, iters=cfg.pde.train.iters, shuffle=True, num_workers = 3)
        self.test_loader = BidxDataLoader(self.test_dataset, batch_size=cfg.pde.test.batch_size, shuffle=False, num_workers = 3)

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
        
        for i, (funcs, grids, targs) in enumerate(train_loader):
            preds = self.net((funcs, grids))
            loss = self.criterion(preds, targs)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            l2rel = utils.metrics.mean_l2_relative_error(preds, targs)
            lossStat.add(loss)
            metStat.add(l2rel)
            self.total_iter += 1
            
            if i % log_every == 0:
                test_loss, test_met = self.test()
                self.log.info(f"Iter {i:6d}/{self.cfg.pde.train.iters:6d} | train loss {lossStat:.2e} | test loss {test_loss:.2e} | test met {test_met:.2e}")
                self.train_loss.add(lossStat.calc())
                self.test_loss.add(test_loss.calc())
                self.test_met.add(test_met.calc())
                lossStat.reset()
                metStat.reset()
                self.save(test_met)
        
        self.save(test_met)
        return self.train_loss.calc(), self.test_loss.calc(), self.test_met.calc()
            
    @torch.no_grad()
    def test(self):
        test_loss = utils.metrics.metStat()
        test_met = utils.metrics.metStat()
        for i, (funcs, grids, targs) in enumerate(self.test_loader):
            preds = self.net((funcs, grids))
            loss = self.criterion(preds, targs)
            l2rel = utils.metrics.mean_l2_relative_error(preds, targs)
            test_loss.add(loss)
            test_met.add(l2rel)
        return test_loss, test_met
    
    def save(self, met):
        if met < self.best:
            self.best = met
        path = f"{self.work_dir}/{self.total_iter}.pkl"
        if len(self.save_paths) >= self.cfg.pde.max_save:
            os.remove(self.save_paths.pop(0))
        utils.model_save(self.net, path)
        self.save_paths.append(path)
    
def prepare(cfg):
    net = utils.get_model(cfg.pde.model.name, cfg.pde.model.params, cfg.pde.model.input_transform, cfg.pde.model.output_transform)
    if cfg.datasets.pretrain_path is not None:
        match_list = utils.model_load(net, cfg.datasets.pretrain_path)
        
    train_data = np.load(cfg.datasets.train_path)
    test_data = np.load(cfg.datasets.test_path)
    
    train_data = Cartesian(train_data['funcs'], train_data['grids'], train_data['out'])
    test_data = Cartesian(test_data['funcs'], test_data['grids'], test_data['out'])
    
    optimizer = torch.optim.Adam(net.parameters(), lr=cfg.optimizer.lr, weight_decay=cfg.optimizer.weight_decay)
    
    log = utils.logger("Rank 0")
    
    return net, train_data, test_data, optimizer, log

@hydra.main(config_path="config", config_name="config", version_base=None)
def main(cfg):
    # net, train_dataset, test_dataset, optimizer, log = prepare(cfg)
    space = utils.get_space(cfg.funcspace.name, cfg.funcspace.params)
    print(utils.get_space_name(space))

if __name__ == "__main__":
    main()