from ppad.utils import get_logger
from ppad.modeling import build_model

def train_model(cfg):
    logger = get_logger()

    model = build_model(cfg.MODEL)
    pass