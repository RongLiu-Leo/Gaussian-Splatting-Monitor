import logging
import logging.config
from pathlib import Path
VERBOSE = 15
def verbose(self, message, *args, **kws):
    if self.isEnabledFor(VERBOSE):
        # Yes, logger takes its '*args' as 'args'.
        self._log(VERBOSE, message, args, **kws) 
logging.addLevelName(VERBOSE, 'VERBOSE')
logging.Logger.verbose = verbose

def setup_logging(logger_cfg, logger_path):
    """
    Setup logging configuration
    """
    log_file_path = Path(logger_path) / logger_cfg['handlers']['info_file_handler']['filename']
    log_file_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
    logger_cfg['handlers']['info_file_handler']['filename'] = str(log_file_path)
    logging.config.dictConfig(logger_cfg)

def init_logger(name):
    logger = logging.getLogger(name)
    return logger