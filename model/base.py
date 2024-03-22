
class BaseModule():
    def __init__(self, cfg, logger):
        self.logger = logger
        for key, value in cfg.items():
            if not hasattr(self.__class__, key) or not isinstance(getattr(self.__class__, key), property):
                setattr(self, key, value)
        
        
