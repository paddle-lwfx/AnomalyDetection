import os
import sys
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '..')))
import random

import numpy as np
import paddle

from ppad.utils import ArgsParser, get_config
from ppad import train_model


def main():
    args = ArgsParser().parse_args()
    cfg = get_config(args.config, overrides=args.opt)

    # set seed if specified
    seed = args.seed
    if seed is not None:
        assert isinstance(
            seed,
            int), f"seed must be a integer when specified, but got {seed}"
        random.seed(seed)
        np.random.seed(seed)
        paddle.seed(seed)

    train_model(cfg, args.validate)


if __name__ == '__main__':
    main()
