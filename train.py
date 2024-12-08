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

def get_training_device():
    """
    获取训练设备，并设置相应的显存限制
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        # 设置显存限制，预留部分显存给系统
        torch.cuda.set_per_process_memory_fraction(0.9, 0)  # 使用90%显存用于训练
        return device
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

def train():
    try:
        # 创建数据配置文件
        yaml_path = create_data_yaml()
        
        # 获取最优训练设备
        device = get_training_device()
        
        # 更新训练参数
        train_params = config.TRAIN_PARAMS.copy()
        train_params['device'] = device
        
        # 如果使用CUDA，启用混合精度训练
        if device.type == 'cuda':
            train_params['amp'] = True  # 自动混合精度
        
        # 构建训练命令
        cmd = [
            'python3', '-m', 'yolov5.train',
            '--img', str(train_params['imgsz']),
            '--batch', str(train_params['batch']),
            '--epochs', str(train_params['epochs']),
            '--data', yaml_path,
            '--weights', 'yolov5s.pt',  # 使用预训练模型
            '--device', str(train_params['device']),
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
