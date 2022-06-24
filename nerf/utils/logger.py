import logging
import numpy as np
from collections import OrderedDict

def get_root_logger(log_level=logging.INFO):
    logger = logging.getLogger()
    if not logger.hasHandlers():
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(message)s", level=log_level
        )
    return logger