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
# ckpt_path = "runs/detect/endovis-spd-cbma/weights/best.pt"

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
name = ckpt_path.split("/")[-3] + "-bt"
hyp = {
    'seed': 0,
    'name': name,
    'data': 'datasets/endovis_bt.yaml',
    'epochs': 100,
    'device': '1',
    'workers': 8,
    # 'mixup':0.5,
}
model.eval(**hyp)  # train the model

# nohup yolo task=detect mode=val name=bt-endovis model=runs/detect/endovis/weights/best.pt data=datasets/endovis_bt.yaml device=3 > bt-endovis.log 2>&1 &
# nohup yolo task=detect mode=val name=bt-endovis-spd-cf model=runs/detect/endovis-spd-cf/weights/best.pt data=datasets/endovis_bt.yaml device=3 > bt-endovis-spd-cf.log 2>&1 &
# nohup yolo task=detect mode=val name=bt-endovis-spd-ct model=runs/detect/endovis-spd-ct/weights/best.pt data=datasets/endovis_bt.yaml device=3 > bt-endovis-spd-ct.log 2>&1 &

# nohup yolo task=detect mode=val name=bt-endovis-spd-cfnew model=runs/detect/endovis-spd-cfnew/weights/best.pt data=datasets/endovis_bt.yaml device=3 > bt-endovis-spd-cfnew.log 2>&1 &
# nohup yolo task=detect mode=val name=bt-endovis-spd-ctnew model=runs/detect/endovis-spd-ctnew/weights/best.pt data=datasets/endovis_bt.yaml device=3 > bt-endovis-spd-ctnew.log 2>&1 &
# nohup yolo task=detect mode=val name=bt-endovis-spd-cfnew-bias model=runs/detect/endovis-spd-cfnew-bias/weights/best.pt data=datasets/endovis_bt.yaml device=3 > bt-endovis-spd-cfnew-bias.log 2>&1 &
# nohup yolo task=detect mode=val name=bt-endovis-spd-ctnew-bias model=runs/detect/endovis-spd-ctnew-bias/weights/best.pt data=datasets/endovis_bt.yaml device=3 > bt-endovis-spd-ctnew-bias.log 2>&1 &

# nohup yolo task=detect mode=val name=bt-endovis-rmt1-p1 model=runs/detect/endovis-rmt1-p1/weights/best.pt data=datasets/endovis_bt.yaml device=3 > bt-endovis-rmt1-p1.log 2>&1 &
# nohup yolo task=detect mode=val name=bt-endovis-rmt1-p2 model=runs/detect/endovis-rmt1-p2/weights/best.pt data=datasets/endovis_bt.yaml device=3 > bt-endovis-rmt1-p2.log 2>&1 &
# nohup yolo task=detect mode=val name=bt-endovis-rmt1-p5 model=runs/detect/endovis-rmt1-p5/weights/best.pt data=datasets/endovis_bt.yaml device=3 > bt-endovis-rmt1-p5.log 2>&1 &
# nohup yolo task=detect mode=val name=bt-endovis-rmt2-p1 model=runs/detect/endovis-rmt2-p1/weights/best.pt data=datasets/endovis_bt.yaml device=3 > bt-endovis-rmt2-p1.log 2>&1 &
# nohup yolo task=detect mode=val name=bt-endovis-rmt2-p2 model=runs/detect/endovis-rmt2-p2/weights/best.pt data=datasets/endovis_bt.yaml device=3 > bt-endovis-rmt2-p2.log 2>&1 &
# nohup yolo task=detect mode=val name=bt-endovis-rmt2-p5 model=runs/detect/endovis-rmt2-p5/weights/best.pt data=datasets/endovis_bt.yaml device=3 > bt-endovis-rmt2-p5.log 2>&1 &
# nohup yolo task=detect mode=val name=bt-endovis-rmt3-p1 model=runs/detect/endovis-rmt3-p1/weights/best.pt data=datasets/endovis_bt.yaml device=3 > bt-endovis-rmt3-p1.log 2>&1 &
# nohup yolo task=detect mode=val name=bt-endovis-rmt3-p2 model=runs/detect/endovis-rmt3-p2/weights/best.pt data=datasets/endovis_bt.yaml device=3 > bt-endovis-rmt3-p2.log 2>&1 &
# nohup yolo task=detect mode=val name=bt-endovis-rmt3-p5 model=runs/detect/endovis-rmt3-p5/weights/best.pt data=datasets/endovis_bt.yaml device=3 > bt-endovis-rmt3-p5.log 2>&1 &
# nohup yolo task=detect mode=val name=bt-endovis-mup model=runs/detect/endovis-mup/weights/best.pt data=datasets/endovis_bt.yaml device=3 > bt-endovis-mup.log 2>&1 &
# nohup yolo task=detect mode=val name=bt-endovis-all model=runs/detect/endovis-all/weights/best.pt data=datasets/endovis_bt.yaml device=3 > bt-endovis-all.log 2>&1 &

# nohup yolo task=detect mode=val name=bt-endovis-all-spd-cfalse-sgd model=runs/detect/endovis-all-spd-cfalse-sgd/weights/best.pt data=datasets/endovis_bt.yaml device=1 > bt-endovis-all-spd-cfalse-sgd.log 2>&1 &

# nohup yolo task=detect mode=val name=bt-endovis-v5 model=runs/detect/endovis-v5/weights/best.pt data=datasets/endovis_bt.yaml device=3 > bt-endovis-v5.log 2>&1 &
# nohup yolo task=detect mode=val name=bt-endovis-v6 model=runs/detect/endovis-v6/weights/best.pt data=datasets/endovis_bt.yaml device=3 > bt-endovis-v6.log 2>&1 &
# nohup yolo task=detect mode=val name=bt-endovis-v8n-mup model=runs/detect/endovis-v8n-mup/weights/best.pt data=datasets/endovis_bt.yaml device=0 > bt-endovis-v8n-mup.log 2>&1 &