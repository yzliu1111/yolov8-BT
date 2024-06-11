nohup yolo task=detect mode=val name=test-endovis-v8spt model=runs/detect/endovis-v8spt/weights/last.pt data=datasets/endovis_test.yaml device=3 > test-endovis-v8spt1.log 2>&1 &
nohup yolo task=detect mode=val name=test-endovis-v8mpt model=runs/detect/endovis-v8mpt/weights/last.pt data=datasets/endovis_test.yaml device=3 > test-endovis-v8mpt1.log 2>&1 &
nohup yolo task=detect mode=val name=test-endovis-v8lpt model=runs/detect/endovis-v8lpt/weights/last.pt data=datasets/endovis_test.yaml device=3 > test-endovis-v8lpt1.log 2>&1 &
nohup yolo task=detect mode=val name=test-endovis-v8xpt model=runs/detect/endovis-v8xpt/weights/last.pt data=datasets/endovis_test.yaml device=3 > test-endovis-v8xpt1.log 2>&1 &


nohup yolo task=detect mode=val name=bt-endovis-v8spt model=runs/detect/endovis-v8spt/weights/last.pt data=datasets/endovis_bt.yaml device=3 > bt-endovis-v8spt1.log 2>&1 &
nohup yolo task=detect mode=val name=bt-endovis-v8mpt model=runs/detect/endovis-v8mpt/weights/last.pt data=datasets/endovis_bt.yaml device=3 > bt-endovis-v8mpt1.log 2>&1 &
nohup yolo task=detect mode=val name=bt-endovis-v8lpt model=runs/detect/endovis-v8lpt/weights/last.pt data=datasets/endovis_bt.yaml device=3 > bt-endovis-v8lpt1.log 2>&1 &
nohup yolo task=detect mode=val name=bt-endovis-v8xpt model=runs/detect/endovis-v8xpt/weights/last.pt data=datasets/endovis_bt.yaml device=3 > bt-endovis-v8xpt1.log 2>&1 &