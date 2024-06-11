from ultralytics.models import YOLO
from train_cfg import all_train_cfgs
import argparse
import random
import numpy as np
import os
import torch

def set_seed(seed):
    # seed init.
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    # torch seed init.
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'

    torch.use_deterministic_algorithms(True)


parser = argparse.ArgumentParser()
parser.add_argument('--cfg', type=str, default='v8n')
args = parser.parse_args()

hyp = all_train_cfgs[args.cfg]
set_seed(hyp['seed'])

# Load a model
model = YOLO(hyp['model_name'])
del hyp["model_name"]
model.train(**hyp)

