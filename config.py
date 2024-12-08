import os

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# 数据集路径
DATASET_DIR = os.path.join(PROJECT_ROOT, 'dataset')
PICS_DIR = os.path.join(DATASET_DIR, 'pics')
LABEL_DIR = os.path.join(DATASET_DIR, 'label')

# 输出目录
IMAGES_DIR = os.path.join(PROJECT_ROOT, 'data', 'images')
LABELS_DIR = os.path.join(PROJECT_ROOT, 'data', 'labels')

# 模型路径
MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'best.pt')

# 训练参数
TRAIN_PARAMS = {
    'data': os.path.join(PROJECT_ROOT, 'data.yaml'),
    'epochs': 150,
    'imgsz': 640,
    'batch': 36,
    'device': None,
    'verbose': True,
    'workers': 8,
    'cache': False
}

# 早停机制参数
EARLY_STOPPING = {
    'patience': 5  # 可以根据需要调整
} 