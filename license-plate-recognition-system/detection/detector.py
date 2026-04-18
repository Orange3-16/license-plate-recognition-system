import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO


class PlateDetector:
    """
    车牌检测器，基于YOLOv8
    """
    def __init__(self, model_path=None, device='cpu', conf_threshold=0.5):
        """
        初始化检测器
        
        Args:
            model_path: 模型权重文件路径
            device: 设备 ('cpu' 或 'cuda')
            conf_threshold: 置信度阈值
        """
        self.device = device
        self.conf_threshold = conf_threshold
        
        if model_path is None:
            model_path = 'runs/detect/detection/runs/train/weights/best.pt'
        
        self.model = YOLO(model_path)
        self.model.to(device)
        
        print(f"加载YOLOv8模型: {model_path}")
        print(f"设备: {device}")
    
    def detect_plate(self, image):
        """
        检测图像中的车牌
        
        Args:
            image: 输入BGR图像
            
        Returns:
            list: 检测到的车牌列表，每个元素为 (x1, y1, x2, y2, confidence)
        """
        results = self.model(image, conf=self.conf_threshold, verbose=False)
        
        plates = []
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = box.conf[0].cpu().numpy()
                
                plates.append((int(x1), int(y1), int(x2), int(y2), float(confidence)))
        
        return plates
    
    def detect_plate_single(self, image):
        """
        检测图像中的车牌，返回置信度最高的一个
        
        Args:
            image: 输入BGR图像
            
        Returns:
            tuple or None: (x1, y1, x2, y2) 或 None
        """
        plates = self.detect_plate(image)
        
        if len(plates) == 0:
            return None
        
        best_plate = max(plates, key=lambda x: x[4])
        
        return (best_plate[0], best_plate[1], best_plate[2], best_plate[3])
    
    def crop_plate(self, image, bbox):
        """
        根据边界框裁剪车牌区域
        
        Args:
            image: 输入BGR图像
            bbox: 边界框 (x1, y1, x2, y2)
            
        Returns:
            裁剪后的车牌图像
        """
        x1, y1, x2, y2 = bbox
        
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(image.shape[1], x2)
        y2 = min(image.shape[0], y2)
        
        if x2 <= x1 or y2 <= y1:
            return None
        
        plate = image[y1:y2, x1:x2]
        
        return plate
    
    def draw_detection(self, image, plates, labels=None):
        """
        在图像上绘制检测结果
        
        Args:
            image: 输入BGR图像
            plates: 车牌列表 [(x1, y1, x2, y2, confidence), ...]
            labels: 车牌标签列表（可选）
            
        Returns:
            绘制了检测结果的图像
        """
        result_img = image.copy()
        
        for i, plate in enumerate(plates):
            x1, y1, x2, y2, conf = plate
            
            color = (0, 255, 0)
            
            cv2.rectangle(result_img, (x1, y1), (x2, y2), color, 2)
            
            label_text = f"{conf:.2f}"
            if labels and i < len(labels):
                label_text = f"{labels[i]} ({conf:.2f})"
            
            label_size, _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            
            cv2.rectangle(result_img, (x1, y1 - label_size[1] - 10),
                         (x1 + label_size[0], y1), color, -1)
            
            cv2.putText(result_img, label_text, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        return result_img


def load_detector(model_path=None, device='cpu', conf_threshold=0.5):
    """
    加载车牌检测器
    
    Args:
        model_path: 模型权重文件路径
        device: 设备
        conf_threshold: 置信度阈值
        
    Returns:
        PlateDetector实例
    """
    return PlateDetector(model_path, device, conf_threshold)


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("使用方法: python detector.py <input_image> [output_image] [model_path]")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else 'detection_result.jpg'
    model_path = sys.argv[3] if len(sys.argv) > 3 else None
    
    image = cv2.imread(input_path)
    if image is None:
        print(f"无法读取图像: {input_path}")
        sys.exit(1)
    
    detector = PlateDetector(model_path)
    
    print("正在检测车牌...")
    plates = detector.detect_plate(image)
    
    print(f"检测到 {len(plates)} 个车牌:")
    for i, plate in enumerate(plates):
        x1, y1, x2, y2, conf = plate
        print(f"  车牌 {i+1}: ({x1}, {y1}, {x2}, {y2}), 置信度: {conf:.2f}")
    
    result_img = detector.draw_detection(image, plates)
    
    cv2.imwrite(output_path, result_img)
    print(f"检测结果已保存到: {output_path}")
