import cv2
import torch
import numpy as np
from pathlib import Path
import config

class RobotMasterDetector:
    def __init__(self):
        # 加载模型
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', 
                                  path=config.MODEL_PATH, force_reload=True)
        
        # 设置推理参数
        self.model.conf = 0.25  # 置信度阈值
        self.model.iou = 0.45   # NMS IOU阈值
        self.model.classes = None  # 检测所有类别
        
        # 如果有GPU则使用GPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
    
    def detect(self, image):
        """
        检测图像中的数字及其颜色
        Args:
            image: BGR格式的图像
        Returns:
            detections: 列表，每个元素为 (数字, 颜色, 中心坐标x, 中心坐标y, 置信度)
        """
        # 推理
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
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 处理帧
        processed_frame, detections = detector.process_frame(frame)
        
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