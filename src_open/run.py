import json
import os
import glob
import hydra
from omegaconf import DictConfig

def prerender(cfg):
    from .tools.prerender import prerender_RBOT, preprender_BCOT, preprender_BOP, preprender_OPT
    if cfg.prerender_method == 'BOP':
        preprender_BOP(cfg)
    elif cfg.prerender_method == 'BCOT':
        preprender_BCOT(cfg)
    elif cfg.prerender_method == 'OPT':
        preprender_OPT(cfg)
    elif cfg.prerender_method == 'RBOT':
        prerender_RBOT(cfg)
    elif cfg.prerender_method == 'MyModel':
        preprender_BCOT(cfg)
    else: 
        raise NotImplementedError

def train(cfg):
    from .tools.train import train
    train(cfg)

def test_deepac(cfg):
    from .tools.test_deepac import main
    main(cfg)

def test_json(cfg):
    from .tools.test_json import main
    main(cfg)

def deploy_deepac(cfg):
    from .tools.deploy_deepac import main
    main(cfg)

def demo(cfg):
    from .tools.demo import main
    main(cfg)

@hydra.main(config_path='configs/', config_name='config.yaml')
def main(cfg: DictConfig):
    globals()[cfg.type](cfg)

if __name__ == "__main__":
    main()