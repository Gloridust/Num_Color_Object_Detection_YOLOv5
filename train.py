import yaml
import torch
import subprocess
from pathlib import Path
import config

def create_data_yaml():
    """创建数据配置文件"""
    data = {
        'path': config.DATASET_DIR,  # 数据集根目录
        'train': str(Path(config.IMAGES_DIR) / 'train'),  # 训练集图片路径
        'val': str(Path(config.IMAGES_DIR) / 'val'),      # 验证集图片路径
        
        # 类别名称：0-9表示数字，每个数字有红色和蓝色两种，总共20个类别
        'names': [
            '0_red', '1_red', '2_red', '3_red', '4_red', 
            '5_red', '6_red', '7_red', '8_red', '9_red',
            '0_blue', '1_blue', '2_blue', '3_blue', '4_blue', 
            '5_blue', '6_blue', '7_blue', '8_blue', '9_blue'
        ],
        'nc': 20  # 类别数量
    }
    
    # 保存配置文件
    yaml_path = config.TRAIN_PARAMS['data']
    with open(yaml_path, 'w') as f:
        yaml.dump(data, f)
    return yaml_path

def train():
    try:
        # 创建数据配置文件
        yaml_path = create_data_yaml()
        
        # 构建训练命令
        cmd = [
            'python3', '-m', 'yolov5.train',
            '--img', str(config.TRAIN_PARAMS['imgsz']),
            '--batch', str(config.TRAIN_PARAMS['batch']),
            '--epochs', str(config.TRAIN_PARAMS['epochs']),
            '--data', yaml_path,
            '--weights', 'yolov5s.pt',  # 使用预训练模型
            '--device', str(config.TRAIN_PARAMS['device']),
            '--project', 'runs/train',
            '--name', 'exp',
            '--cache'  # 缓存图像以加快训练
        ]
        
        # 执行训练命令
        subprocess.run(cmd, check=True)
        
    except Exception as e:
        print(f"训练过程中出现错误: {e}")

if __name__ == "__main__":
    train()
