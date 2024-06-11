nohup yolo task=detect mode=val name=bt-endovis-v8n model=runs/detect/endovis-v8n/weights/best.pt data=datasets/endovis_bt.yaml device=0 > bt-endovis-v8n.log 2>&1 &
nohup yolo task=detect mode=val name=bt-endovis-v8n-spd-42 model=runs/detect/endovis-v8n-spd-42/weights/best.pt data=datasets/endovis_bt.yaml device=0 > bt-endovis-v8n-spd-42.log 2>&1 &
nohup yolo task=detect mode=val name=bt-endovis-v8n-rmt1-p1-42 model=runs/detect/endovis-v8n-rmt1-p1-42/weights/best.pt data=datasets/endovis_bt.yaml device=0 > bt-endovis-v8n-rmt1-p1-42.log 2>&1 &
nohup yolo task=detect mode=val name=bt-endovis-v8n-rmt1-p2-42 model=runs/detect/endovis-v8n-rmt1-p2-42/weights/best.pt data=datasets/endovis_bt.yaml device=0 > bt-endovis-v8n-rmt1-p2-42.log 2>&1 &
nohup yolo task=detect mode=val name=bt-endovis-v8n-rmt1-p5-42 model=runs/detect/endovis-v8n-rmt1-p5-42/weights/best.pt data=datasets/endovis_bt.yaml device=0 > bt-endovis-v8n-rmt1-p5-42.log 2>&1 &
nohup yolo task=detect mode=val name=bt-endovis-v8n-rmt2-p1-42 model=runs/detect/endovis-v8n-rmt2-p1-42/weights/best.pt data=datasets/endovis_bt.yaml device=1 > bt-endovis-v8n-rmt2-p1-42.log 2>&1 &
nohup yolo task=detect mode=val name=bt-endovis-v8n-rmt2-p2-42 model=runs/detect/endovis-v8n-rmt2-p2-42/weights/best.pt data=datasets/endovis_bt.yaml device=1 > bt-endovis-v8n-rmt2-p2-42.log 2>&1 &
nohup yolo task=detect mode=val name=bt-endovis-v8n-rmt2-p5-42 model=runs/detect/endovis-v8n-rmt2-p5-42/weights/best.pt data=datasets/endovis_bt.yaml device=1 > bt-endovis-v8n-rmt2-p5-42.log 2>&1 &
nohup yolo task=detect mode=val name=bt-endovis-v8n-rmt3-p1-42 model=runs/detect/endovis-v8n-rmt3-p1-42/weights/best.pt data=datasets/endovis_bt.yaml device=1 > bt-endovis-v8n-rmt3-p1-42.log 2>&1 &
nohup yolo task=detect mode=val name=bt-endovis-v8n-rmt3-p2-42 model=runs/detect/endovis-v8n-rmt3-p2-42/weights/best.pt data=datasets/endovis_bt.yaml device=1 > bt-endovis-v8n-rmt3-p2-42.log 2>&1 &
nohup yolo task=detect mode=val name=bt-endovis-v8n-rmt3-p5-42 model=runs/detect/endovis-v8n-rmt3-p5-42/weights/best.pt data=datasets/endovis_bt.yaml device=1 > bt-endovis-v8n-rmt3-p5-42.log 2>&1 &
nohup yolo task=detect mode=val name=bt-endovis-v8n-mup-42 model=runs/detect/endovis-v8n-mup-42/weights/best.pt data=datasets/endovis_bt.yaml device=3 > bt-endovis-v8n-mup-42.log 2>&1 &
nohup yolo task=detect mode=val name=bt-endovis-v8n-mu model=runs/detect/endovis-v8n-mu/weights/best.pt data=datasets/endovis_bt.yaml device=3 > bt-endovis-v8n-mu.log 2>&1 &
nohup yolo task=detect mode=val name=bt-endovis-v8n-cf-p1 model=runs/detect/endovis-v8n-cf-p1/weights/best.pt data=datasets/endovis_bt.yaml device=3 > bt-endovis-v8n-cf-p1.log 2>&1 &
nohup yolo task=detect mode=val name=bt-endovis-v8n-cf-p2 model=runs/detect/endovis-v8n-cf-p2/weights/best.pt data=datasets/endovis_bt.yaml device=3 > bt-endovis-v8n-cf-p2.log 2>&1 &
nohup yolo task=detect mode=val name=bt-endovis-v8n-cf-p5 model=runs/detect/endovis-v8n-cf-p5/weights/best.pt data=datasets/endovis_bt.yaml device=3 > bt-endovis-v8n-cf-p5.log 2>&1 &

# nohup yolo task=detect mode=val name=bt-endovis-v8n-all-1p2-42 model=runs/detect/endovis-v8n-all-1p2-42/weights/best.pt data=datasets/endovis_bt.yaml device=1 batch=1 plots=True > bt-endovis-v8n-all-1p2-42.log 2>&1 &
# nohup yolo task=detect mode=val name=bt-endovis-v8n-all-3p2-42 model=runs/detect/endovis-v8n-all-3p2-42/weights/best.pt data=datasets/endovis_bt.yaml device=3 > bt-endovis-v8n-all-3p2-42.log 2>&1 &

nohup yolo task=detect mode=val name=bt-endovis-v8n-mup-423 model=runs/detect/endovis-v8n-mup-post2/weights/epoch11.pt data=datasets/endovis_bt.yaml device=1 > bt-endovis-v8n-mup-423.log 2>&1 &