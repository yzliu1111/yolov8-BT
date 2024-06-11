all_train_cfgs = {

    'v8n-mup': {
        'model_name': 'yolov8n.pt',
        'name': 'endovis-v8n-mup-aug',
        'data': 'datasets/endovis_aug.yaml',
        'epochs': 300,
        'device': '3',
        'workers': 8,
        'mixup': 0.5,
        'use_mixupplus': True,
        'seed': 42,
    },

    'v8n-all-1p1': {
        'model_name': 'yolov8n-spd.yaml',
        'name': 'endovis-v8n-all-1p1-42-aug',
        'data': 'datasets/endovis_aug.yaml',
        'epochs': 300,
        'device': '1',
        'workers': 8,
        'rmt': 0.1,
        'rmt_type': 1,
        'mixup': 0.5,
        'use_mixupplus': True,
        'seed': 42,
    },

    'v8n-all-1p2': {
        'model_name': 'yolov8n-spd.yaml',
        'name': 'endovis-v8n-all-1p2-42-aug',
        'data': 'datasets/endovis_aug.yaml',
        'epochs': 300,
        'device': '2',
        'workers': 8,
        'rmt': 0.2,
        'rmt_type': 1,
        'mixup': 0.5,
        'use_mixupplus': True,
        'seed': 42,
    },

    'v8n-all-3p2': {
        'model_name': 'yolov8n-spd.yaml',
        'name': 'endovis-v8n-all-3p2-42-aug',
        'data': 'datasets/endovis_aug.yaml',
        'epochs': 300,
        'device': '3',
        'workers': 8,
        'rmt': 0.2,
        'rmt_type': 3,
        'mixup': 0.5,
        'use_mixupplus': True,
        'seed': 42,
    },

    'v8n-all-post': {
        'model_name': 'runs/detect/endovis-v8n-all-1p2-42/weights/best.pt',
        'name': 'endovis-v8n-all-post',
        'data': 'datasets/endovis_fine.yaml',
        'epochs': 15,
        'device': '1',
        'workers': 8,
        # 'mixup': 0.5,
        # 'use_mixupplus': True,
        'seed': 42,
        # 'optimizer': 'SGD',
        # 'lr0': 1e-3
        'save_period': 1
    },

    'v8n-mup-post': {
        'model_name': 'runs/detect/endovis-v8n-mup-42/weights/best.pt',
        'name': 'endovis-v8n-mup-post',
        'data': 'datasets/endovis_fine.yaml',
        'epochs': 15,
        'device': '1',
        'workers': 8,
        # 'mixup': 0.5,
        # 'use_mixupplus': True,
        'seed': 42,
        # 'optimizer': 'SGD',
        # 'lr0': 1e-3
        'save_period': 1
    },
}