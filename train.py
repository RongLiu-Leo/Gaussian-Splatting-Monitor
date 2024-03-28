from dataloader import ColmapData
from model import (
    GaussianRepr, 
    L1WithSSIMLoss, 
    AdamWithcustomlrParamOptim, 
    DiffRasterizerRenderer,
    SplitWithCloneWithPrune,
    BaseTrainer)
import time

from pathlib import Path
from utils import *
import random

def init_settings(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.set_device(torch.device("cuda:0"))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main(cfg):    

    # init folder paths
    name = cfg.get('name')
    exp_folder = name + '@' + time.strftime("%Y%m%d-%H%M%S")
    exp_path = Path(cfg.get('exp_path')) / exp_folder
    exp_path.mkdir(parents=True, exist_ok=True)
    info_path = exp_path / "info"
    info_path.mkdir(parents=True, exist_ok=True)
    result_path = exp_path / "results"
    result_path.mkdir(parents=True, exist_ok=True)
    ckpt_path = exp_path / "ckpts"
    ckpt_path.mkdir(parents=True, exist_ok=True)

    # init settings
    seed = cfg.get('seed')
    init_settings(seed)
    cp_config = info_path / Path(cfg.get("config_path")).name
    cp_config.write_text(yaml.dump(cfg))

    # init logger
    setup_logging(cfg.get('logger'), info_path)
    logger = init_logger(name)

    # check if continue training from a checkpoint
    use_checkpoint = cfg.get('checkpoint').get('use')
    if use_checkpoint:
        exist_ckpt_path = Path(cfg.get('checkpoint').get('path'))
        logger.info(f"Using checkpoint {exist_ckpt_path}")
        exist_ckpt_state = torch.load(str(exist_ckpt_path))
        
    # init network gui
    networkGui = NetworkGUI(cfg = cfg.get('networkGui'), logger = logger)

    # init data
    data = ColmapData(cfg = cfg.get('data'), logger = logger)

    # init representation
    repr = GaussianRepr(cfg = cfg.get('repr'), logger = logger)
    if use_checkpoint:
        repr.load(exist_ckpt_state['repr'])
    else:
        repr.init_from_pcd(data.point_cloud)
    
    # init Optimizers
    paramOptim = AdamWithcustomlrParamOptim(cfg = cfg.get('paramOptim'), logger = logger, 
                                            spatial_lr_scale = data.spatial_scale, 
                                            repr = repr)
    
                          
    structOptim = SplitWithCloneWithPrune(cfg = cfg.get('structOptim'), logger = logger, 
                                            spatial_lr_scale = data.spatial_scale, 
                                            num_init_points = repr.xyz.shape[0])
    if use_checkpoint:
        paramOptim.load(exist_ckpt_state['paramOptim'])
        structOptim.load(exist_ckpt_state['structOptim'])

    # init rederer and loss
    renderer = DiffRasterizerRenderer(cfg = cfg.get('renderer'), logger = logger)
    loss = L1WithSSIMLoss(cfg = cfg.get('loss'), logger = logger)
     
    
    # init recorder
    if use_checkpoint:
        first_iteration = exist_ckpt_state['iteration'] + 1
    else:
        first_iteration = 1
    recorder = Recorder(cfg = cfg.get('recorder'), logger = logger, 
                        info_path = info_path, first_iter = first_iteration, 
                        max_iter = cfg.get('trainer').get('iterations'))
    

    # init trainer
    trainer = BaseTrainer(cfg = cfg.get('trainer'), logger = logger, 
                          data = data, repr = repr, loss = loss, 
                          paramOptim = paramOptim, 
                          structOptim = structOptim,
                          renderer = renderer,
                          networkGui = networkGui,
                          result_path = result_path,
                          ckpt_path = ckpt_path,
                          recorder = recorder,
                          first_iteration = first_iteration)
    trainer.init_save_results()
    
    # train!
    trainer.train()
    
if __name__ == "__main__":
    config_loader = ConfigLoader()
    config_loader.parse_args()
    main(config_loader.cfg)
