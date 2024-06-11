import os
from ultralytics import YOLO

exp = 'v8n-all-1p2-42'
model_path = 'runs/detect/endovis-{}/weights/predict.pt'.format(exp)
name = 'predict-{}'.format(exp)
device = '1'

data_folder = 'datasets/endovis_wrapup/images/bt'
img_list =[os.path.join(data_folder, img_name) for img_name in os.listdir(data_folder)] 

model = YOLO(model_path)

model.predict(img_list, name=name, save=True, device=device)