nohup yolo task=detect mode=val name=speed-v8n model=yolov8n.yaml data=datasets/endovis_speed.yaml device=0 batch=1 > speed-v8n.log 2>&1 &
nohup yolo task=detect mode=val name=speed-v8s model=yolov8s.yaml data=datasets/endovis_speed.yaml device=0 batch=1 > speed-v8s.log 2>&1 &
nohup yolo task=detect mode=val name=speed-v8m model=yolov8m.yaml data=datasets/endovis_speed.yaml device=0 batch=1 > speed-v8m.log 2>&1 &
nohup yolo task=detect mode=val name=speed-v8l model=yolov8l.yaml data=datasets/endovis_speed.yaml device=0 batch=1 > speed-v8l.log 2>&1 &
nohup yolo task=detect mode=val name=speed-v8x model=yolov8x.yaml data=datasets/endovis_speed.yaml device=0 batch=1 > speed-v8x.log 2>&1 &