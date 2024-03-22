from dataloader import ColmapData
from model import (
    GaussianRepr, 
    L1WithSSIMLoss, 
    AdamWithcustomlrParamOptim, 
    DiffRasterizerRenderer,
    SplitWithCloneWithPrune,
    BaseTrainer)
import yaml


def main():    
    # init config and logger
    with open('config.yaml', 'r') as file:
        cfg = yaml.safe_load(file)
    logger = 0

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
    trainer = BaseTrainer(cfg = cfg.get('trainer'), logger = logger, 
                          data = data, repr = repr, loss = loss, 
                          paramOptim = paramOptim, 
                          structOptim = structOptim,
                          renderer = renderer)
    return
    # train!
    trainer.train()
    
    

if __name__ == "__main__":
    main()
