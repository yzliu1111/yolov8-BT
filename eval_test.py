from ultralytics.models import YOLO

import numpy as np
import random
import os
import torch
def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False

seed_torch(0)

# Load a model
ckpt_path = "runs/detect/endovis/weights/best.pt"

# ckpt_path = "runs/detect/endovis-spd/weights/best.pt"
# ckpt_path = "runs/detect/endovis-spd-cbam/weights/best.pt"

# ckpt_path = "runs/detect/endovis-rmt1-p1/weights/best.pt"
# ckpt_path = "runs/detect/endovis-rmt1-p2/weights/best.pt"
# ckpt_path = "runs/detect/endovis-rmt1-p5/weights/best.pt"
# ckpt_path = "runs/detect/endovis-rmt2-p1/weights/best.pt"
# ckpt_path = "runs/detect/endovis-rmt2-p2/weights/best.pt"
# ckpt_path = "runs/detect/endovis-rmt2-p5/weights/best.pt"
# ckpt_path = "runs/detect/endovis-rmt3-p1/weights/best.pt"
# ckpt_path = "runs/detect/endovis-rmt3-p2/weights/best.pt"
# ckpt_path = "runs/detect/endovis-rmt3-p5/weights/best.pt"

# ckpt_path = "runs/detect/endovis-mup/weights/best.pt"

# ckpt_path = "runs/detect/endovis-all/weights/best.pt"

# Load a model
model = YOLO(ckpt_path)  # build a new model from scratch
name = ckpt_path.split("/")[-3] + "-test"
hyp = {
    'seed': 0,
    'name': name,
    'data': 'datasets/endovis_test.yaml',
    'epochs': 100,
    'device': '3',
    'workers': 8,
    # 'mixup':0.5,
}
model.eval(**hyp)  # train the model

# nohup yolo task=detect mode=val name=test-endovis model=runs/detect/endovis/weights/best.pt data=datasets/endovis_test.yaml device=3 > test-endovis.log 2>&1 &
# nohup yolo task=detect mode=val name=test-endovis-spd model=runs/detect/endovis-spd/weights/best.pt data=datasets/endovis_test.yaml device=3 > test-endovis-spd.log 2>&1 &
# nohup yolo task=detect mode=val name=test-endovis-spd-cbma model=runs/detect/endovis-spd-cbma/weights/best.pt data=datasets/endovis_test.yaml device=3 > test-endovis-spd-cbma.log 2>&1 &
# nohup yolo task=detect mode=val name=test-endovis-rmt1-p1 model=runs/detect/endovis-rmt1-p1/weights/best.pt data=datasets/endovis_test.yaml device=3 > test-endovis-rmt1-p1.log 2>&1 &
# nohup yolo task=detect mode=val name=test-endovis-rmt1-p2 model=runs/detect/endovis-rmt1-p2/weights/best.pt data=datasets/endovis_test.yaml device=3 > test-endovis-rmt1-p2.log 2>&1 &
# nohup yolo task=detect mode=val name=test-endovis-rmt1-p5 model=runs/detect/endovis-rmt1-p5/weights/best.pt data=datasets/endovis_test.yaml device=3 > test-endovis-rmt1-p5.log 2>&1 &
# nohup yolo task=detect mode=val name=test-endovis-rmt2-p1 model=runs/detect/endovis-rmt2-p1/weights/best.pt data=datasets/endovis_test.yaml device=3 > test-endovis-rmt2-p1.log 2>&1 &
# nohup yolo task=detect mode=val name=test-endovis-rmt2-p2 model=runs/detect/endovis-rmt2-p2/weights/best.pt data=datasets/endovis_test.yaml device=3 > test-endovis-rmt2-p2.log 2>&1 &
# nohup yolo task=detect mode=val name=test-endovis-rmt2-p5 model=runs/detect/endovis-rmt2-p5/weights/best.pt data=datasets/endovis_test.yaml device=3 > test-endovis-rmt2-p5.log 2>&1 &
# nohup yolo task=detect mode=val name=test-endovis-rmt3-p1 model=runs/detect/endovis-rmt3-p1/weights/best.pt data=datasets/endovis_test.yaml device=3 > test-endovis-rmt3-p1.log 2>&1 &
# nohup yolo task=detect mode=val name=test-endovis-rmt3-p2 model=runs/detect/endovis-rmt3-p2/weights/best.pt data=datasets/endovis_test.yaml device=3 > test-endovis-rmt3-p2.log 2>&1 &
# nohup yolo task=detect mode=val name=test-endovis-rmt3-p5 model=runs/detect/endovis-rmt3-p5/weights/best.pt data=datasets/endovis_test.yaml device=3 > test-endovis-rmt3-p5.log 2>&1 &

# nohup yolo task=detect mode=val name=test-endovis-mup model=runs/detect/endovis-mup5/weights/best.pt data=datasets/endovis_test.yaml device=3 > test-endovis-mup.log 2>&1 &
# nohup yolo task=detect mode=val name=test-endovis-all model=runs/detect/endovis-all/weights/best.pt data=datasets/endovis_test.yaml device=3 > test-endovis-all.log 2>&1 &

# nohup yolo task=detect mode=val name=test-endovis-v5 model=runs/detect/endovis-v5/weights/best.pt data=datasets/endovis_test.yaml device=3 > test-endovis-v5.log 2>&1 &
# nohup yolo task=detect mode=val name=test-endovis-v6 model=runs/detect/endovis-v6/weights/best.pt data=datasets/endovis_test.yaml device=3 > test-endovis-v6.log 2>&1 &


