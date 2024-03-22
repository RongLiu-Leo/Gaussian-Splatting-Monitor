from dataloader import ColmapData
from model import (
    GaussianRepr, 
    L1WithSSIMLoss, 
    AdamWithcustomlrParamOptim, 
    DiffRasterizerRenderer,
    SplitWithCloneWithPrune,
    BaseTrainer)
import time
import yaml
import argparse
from pathlib import Path
from utils import *



def main(config_path):    
    # init config
    with open(config_path, 'r') as file:
        cfg = yaml.safe_load(file)
    name = cfg.get('name')
    exp_folder = name + '@' + time.strftime("%Y%m%d%H%M%S")
    exp_path = Path(cfg.get('exp_path')) / exp_folder
    exp_path.mkdir(parents=True, exist_ok=True)

    # init logger
    info_path = exp_path / "info"
    info_path.mkdir(parents=True, exist_ok=True)
    setup_logging(cfg.get('logger'), info_path)
    logger =  init_logger(name)
     
    # init data
    data = ColmapData(cfg = cfg.get('data'), logger = logger)

    # init representation
    repr = GaussianRepr(cfg = cfg.get('repr'), logger = logger, 
                        spatial_lr_scale = data.spatial_scale)
    repr.init_from_pcd(data.point_cloud)
    
    # init Optimizers
    paramOptim = AdamWithcustomlrParamOptim(cfg = cfg.get('paramOptim'), logger = logger, 
                                            spatial_lr_scale = data.spatial_scale, 
                                            repr = repr)
                                            
    structOptim = SplitWithCloneWithPrune(cfg = cfg.get('structOptim'), logger = logger, 
                                            spatial_lr_scale = data.spatial_scale, 
                                            num_init_points = repr.xyz.shape[0])
    
    # init rederer and loss
    renderer = DiffRasterizerRenderer(cfg = cfg.get('renderer'), logger = logger)
    loss = L1WithSSIMLoss(cfg = cfg.get('loss'), logger = logger)
     
    # init trainer
    result_path = exp_path / "results"
    ckpt_path = exp_path / "ckpts"
    result_path.mkdir(parents=True, exist_ok=True)
    ckpt_path.mkdir(parents=True, exist_ok=True)
    trainer = BaseTrainer(cfg = cfg.get('trainer'), logger = logger, 
                          data = data, repr = repr, loss = loss, 
                          paramOptim = paramOptim, 
                          structOptim = structOptim,
                          renderer = renderer,
                          result_path = result_path,
                          ckpt_path = ckpt_path)
    trainer.init_save_results()
    # train!
    trainer.train()
    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", default="config.yaml", help="Path to config file")
    args = parser.parse_args()
    main(args.config)
