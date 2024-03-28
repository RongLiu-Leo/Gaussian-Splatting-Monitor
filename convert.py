from utils import *
from dataloader.colmap_converter import ColmapProcessor
if __name__ == "__main__":
    config_loader = ConfigLoader('dataloader/dataprep.yaml')
    config_loader.parse_args()
    cfg = config_loader.cfg.get('ColmapConverter')
    print(config_loader.cfg)
    logger = init_logger('ColmapConverter', cfg.get('source_path'))
    p = ColmapProcessor(cfg, logger)
    p.run()