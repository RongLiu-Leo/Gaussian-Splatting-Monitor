
class BaseModule():
    def __init__(self, cfg, logger):
        self.logger = logger
        for key, value in cfg.items():
            setattr(self, key, value)
        
        
