import cv2
import torch
import numpy as np
from pathlib import Path
import config

def get_optimal_device():
    """
    按照优先级选择最优运行设备：CUDA > MPS > CPU
    """
    if torch.cuda.is_available():
        # 如果有CUDA，设置显存限制
        torch.cuda.set_per_process_memory_fraction(0.9, 0)  # 限制显存使用为最大显存的90%
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

class RobotMasterDetector:
    def __init__(self):
        # 选择最优设备
        self.device = get_optimal_device()
        
        # 加载模型时添加显存优化参数
        self.model = torch.hub.load('ultralytics/yolov5', 'custom',
                                  path=config.MODEL_PATH,
                                  force_reload=True,
                                  device=self.device)
        
        # 优化推理参数
        self.model.conf = 0.25
        self.model.iou = 0.45
        self.model.classes = None
        self.model.max_det = 100  # 限制最大检测数量
        
        # 设置推理时的batch size
        self.model.batch_size = 1
        
        # 启用半精度推理以节省显存
        if self.device.type == 'cuda':
            self.model.half()  # 使用FP16
    
    def detect(self, image):
        """
        检测图像中的数字及其颜色
        Args:
            image: BGR格式的图像
        Returns:
            detections: 列表，每个元素为 (数字, 颜色, 中心坐标x, 中心坐标y, 置信度)
        """
        # 根据设备类型处理输入
        if self.device.type == 'cuda':
            image = torch.from_numpy(image).to(self.device).half()  # 转换为FP16
        
        # 推理
        with torch.no_grad():  # 禁用梯度计算
            results = self.model(image)
        
        # 解析结果
        detections = []
        for det in results.xyxy[0]:  # 遍历每个检测结果
            x1, y1, x2, y2, conf, cls = det.cpu().numpy()
            
            # 计算中心点坐标
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            # 获取类别名称
            class_name = results.names[int(cls)]
            
            # 解析数字和颜色
            number = int(class_name.split('_')[0])
            color = class_name.split('_')[1]
            
            detections.append((number, color, center_x, center_y, conf))
        
        return detections

    def process_frame(self, frame):
        """
        处理单帧图像
        Args:
            frame: BGR格式的图像
        Returns:
            processed_frame: 绘制了检测结果的图像
            detections: 检测结果列表
        """
        # 复制原始图像
        processed_frame = frame.copy()
        
        # 执行检测
        detections = self.detect(frame)
        
        # 在图像上绘制检测结果
        for number, color, cx, cy, conf in detections:
            # 根据颜色选择绘制颜色
            bbox_color = (0, 0, 255) if color == 'red' else (255, 0, 0)
            
            # 绘制中心点
            cv2.circle(processed_frame, (int(cx), int(cy)), 5, bbox_color, -1)
            
            # 添加文本标签
            label = f"{number} ({conf:.2f})"
            cv2.putText(processed_frame, label, (int(cx), int(cy) - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, bbox_color, 2)
        
        return processed_frame, detections

def main():
    # 初始化检测器
    detector = RobotMasterDetector()
    
    # 打开摄像头（根据实际情况修改摄像头索引）
    cap = cv2.VideoCapture(0)
    
    # 初始化FPS计算相关变量
    fps = 0
    frame_count = 0
    start_time = cv2.getTickCount()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 处理帧
        processed_frame, detections = detector.process_frame(frame)
        
        # 计算FPS
        frame_count += 1
        if frame_count >= 30:  # 每30帧更新一次FPS
            current_time = cv2.getTickCount()
            elapsed_time = (current_time - start_time) / cv2.getTickFrequency()
            fps = frame_count / elapsed_time
            frame_count = 0
            start_time = current_time
        
        # 在画面上显示FPS
        cv2.putText(processed_frame, f"FPS: {fps:.1f}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   1, (0, 255, 0), 2)
        
        # 显示结果
        cv2.imshow('RobotMaster Detection', processed_frame)
        
        # 打印检测结果
        for number, color, cx, cy, conf in detections:
            print(f"检测到{color}色数字{number}，位置({cx:.1f}, {cy:.1f})，置信度{conf:.2f}")
        
        # 按'q'退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 