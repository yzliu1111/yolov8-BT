from ultralytics.models import YOLO

# import numpy as np
# import random
# import os
# import torch
# def seed_torch(seed=0):
#     random.seed(seed)
#     os.environ['PYTHONHASHSEED'] = str(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
#     torch.backends.cudnn.benchmark = False
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.enabled = False

# Load a model
model = YOLO("yolov8n.pt")  # build a new model from scratch
hyp = {
    'name': 'endovis-mup-A100',
    'data': '/homes/yzliu/Code/EnhancedYOLOv8/datasets/endovis.yaml',
    'epochs': 100,
    'device': '0',
    'workers': 8,
    # 'batch': 16,
    # 'rmt': 0.5,
    'mixup':0.5,
    # 'warmup_epochs': 10.0
    # 'cutoff': 0.5,
    # 'copy_paste': 0.5
}

model.train(**hyp)  # train the model


# nohup python -u main.py > endovis-spd.log 2>&1 &
# nohup python -u main.py > endovis.log 2>&1 &
# nohup python -u main.py > endovis-mu.log 2>&1 &
# nohup python -u main.py > endovis-mup.log 2>&1 &
# nohup python -u main.py > endovis-all.log 2>&1 &

# nohup python -u main.py > endovis-all-3p1.log 2>&1 &
# nohup python -u main.py > endovis-all-3p2.log 2>&1 &

# nohup python -u main.py > endovis-rmt2-p1.log 2>&1 &
# nohup python -u main.py > endovis-rmt2-p2.log 2>&1 &
# nohup python -u main.py > endovis-rmt2-p5.log 2>&1 &

# nohup python -u main.py > endovis-rmt1-p1.log 2>&1 &
# nohup python -u main.py > endovis-rmt1-p2.log 2>&1 &
# nohup python -u main.py > endovis-rmt1-p5.log 2>&1 &

# nohup python -u main.py > endovis-rmt3-p1.log 2>&1 &
# nohup python -u main.py > endovis-rmt3-p2.log 2>&1 &
# nohup python -u main.py > endovis-rmt3-p5.log 2>&1 &

# nohup python -u main.py > endovis-cutoffp1.log 2>&1 &
# nohup python -u main.py > endovis-cutoffp2.log 2>&1 &
# nohup python -u main.py > endovis-cutoffp5.log 2>&1 &

# nohup python -u main.py > endovis-copyp1.log 2>&1 &
# nohup python -u main.py > endovis-copyp2.log 2>&1 &
# nohup python -u main.py > endovis-copyp5.log 2>&1 &