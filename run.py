import os  # noqa
import sys  # noqa
sys.path.insert(0, os.path.abspath('src')) # noqa
sys.path.insert(1, os.path.abspath('dino'))
import random
from parser import get_parser
import numpy as np
import torch
from roadslam import RoadSLAM


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


if __name__ == '__main__':
    args = get_parser().parse_args()
    if hasattr(args, 'seeding'):
        setup_seed(args.seeding)

    slam = RoadSLAM(args)
    slam.start()
    slam.wait_child_processes()
