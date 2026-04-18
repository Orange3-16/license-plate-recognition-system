import argparse
import sys
from pathlib import Path
import cv2
import torch
import numpy as np

from detection.detector import PlateDetector
from dehaze.dark_channel_prior import dehaze
from recognition.model import CRNN, decode_prediction, CHARS
from utils import read_image, save_image, draw_bbox, draw_text


class LicensePlateSystem:
    """
    车牌识别系统
    """
    def __init__(self, detection_model_path=None, recognition_model_path=None, 
                 device='cpu', use_dehaze=True):
        """
        初始化车牌识别系统
        
        Args:
            detection_model_path: 检测模型路径
            recognition_model_path: 识别模型路径
            device: 设备 ('cpu' 或 'cuda')
            use_dehaze: 是否使用去雾
        """
        self.device = device
        self.use_dehaze = use_dehaze
        
        print("正在加载检测模型...")
        self.detector = PlateDetector(detection_model_path, device=device)
        
        print("正在加载识别模型...")
        self.recognition_model = CRNN()
        if recognition_model_path:
            self.recognition_model.load_state_dict(
                torch.load(recognition_model_path, map_location=device)
            )
        self.recognition_model.to(device)
        self.recognition_model.eval()
        
        print("系统初始化完成！")
    
    def preprocess_plate(self, plate_image):
        """
        预处理车牌图像
        
        Args:
            plate_image: 车牌图像 (BGR)
            
        Returns:
            预处理后的图像张量
        """
        gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
        
        h, w = gray.shape
        target_h = 32
        scale = target_h / h
        new_w = int(w * scale)
        
        resized = cv2.resize(gray, (new_w, target_h))
        
        target_w = 100
        if new_w < target_w:
            padding = target_w - new_w
            resized = cv2.copyMakeBorder(resized, 0, 0, 0, padding, 
                                         cv2.BORDER_CONSTANT, value=0)
        elif new_w > target_w:
            resized = resized[:, :target_w]
        
        resized = resized.astype(np.float32) / 255.0
        tensor = torch.from_numpy(resized).unsqueeze(0).unsqueeze(0)
        
        return tensor
    
    def recognize_plate(self, plate_image):
        """
        识别车牌文字
        
        Args:
            plate_image: 车牌图像 (BGR)
            
        Returns:
            识别的车牌字符串
        """
        input_tensor = self.preprocess_plate(plate_image)
        input_tensor = input_tensor.to(self.device)
        
        with torch.no_grad():
            output = self.recognition_model(input_tensor)
        
        decoded = decode_prediction(output)
        
        if decoded:
            return decoded[0]
        else:
            return ""
    
    def process_image(self, image, draw_results=True):
        """
        处理图像，检测并识别车牌
        
        Args:
            image: 输入BGR图像
            draw_results: 是否绘制结果
            
        Returns:
            dict: 包含检测结果和识别结果的字典
        """
        result = {
            'plates': [],
            'original_image': image.copy()
        }
        
        plates = self.detector.detect_plate(image)
        
        for plate_info in plates:
            x1, y1, x2, y2, conf = plate_info
            
            plate_crop = self.detector.crop_plate(image, (x1, y1, x2, y2))
            
            if plate_crop is None:
                continue
            
            plate_text = ""
            
            if self.use_dehaze:
                dehazed_plate = dehaze(plate_crop)
                plate_text = self.recognize_plate(dehazed_plate)
            else:
                plate_text = self.recognize_plate(plate_crop)
            
            result['plates'].append({
                'bbox': (x1, y1, x2, y2),
                'confidence': conf,
                'text': plate_text,
                'crop': plate_crop
            })
        
        if draw_results:
            result['result_image'] = self.draw_results(image, result['plates'])
        
        return result
    
    def draw_results(self, image, plates):
        """
        在图像上绘制结果
        
        Args:
            image: 输入图像
            plates: 车牌列表
            
        Returns:
            绘制了结果的图像
        """
        result_image = image.copy()
        
        for plate in plates:
            bbox = plate['bbox']
            text = plate['text']
            conf = plate['confidence']
            
            color = (0, 255, 0)
            
            result_image = draw_bbox(result_image, bbox, color=color, 
                                    thickness=2, label=f"{text} ({conf:.2f})")
        
        return result_image


def main():
    parser = argparse.ArgumentParser(description='面向雨雾天气的车牌识别系统')
    parser.add_argument('--image', type=str, required=True,
                        help='输入图片路径')
    parser.add_argument('--output', type=str, default='result.jpg',
                        help='输出图片路径')
    parser.add_argument('--detection_model', type=str, 
                        default='runs/detect/detection/runs/train/weights/best.pt',
                        help='检测模型路径')
    parser.add_argument('--recognition_model', type=str,
                        default='recognition/best_crnn.pth',
                        help='识别模型路径')
    parser.add_argument('--device', type=str, default='cpu',
                        choices=['cpu', 'cuda'],
                        help='设备')
    parser.add_argument('--no_dehaze', action='store_true',
                        help='不使用去雾')
    parser.add_argument('--show', action='store_true',
                        help='显示结果')
    
    args = parser.parse_args()
    
    input_path = Path(args.image)
    if not input_path.exists():
        print(f"错误: 输入图片不存在: {input_path}")
        sys.exit(1)
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("=" * 50)
    print("面向雨雾天气的车牌识别系统")
    print("=" * 50)
    print(f"输入图片: {input_path}")
    print(f"输出图片: {output_path}")
    print(f"设备: {args.device}")
    print(f"使用去雾: {not args.no_dehaze}")
    print("=" * 50)
    
    image = read_image(input_path)
    if image is None:
        print("错误: 无法读取图片")
        sys.exit(1)
    
    print(f"\n图片尺寸: {image.shape[1]} x {image.shape[0]}")
    
    use_dehaze = not args.no_dehaze
    
    system = LicensePlateSystem(
        detection_model_path=args.detection_model,
        recognition_model_path=args.recognition_model,
        device=args.device,
        use_dehaze=use_dehaze
    )
    
    print("\n正在处理图片...")
    result = system.process_image(image)
    
    print(f"\n检测到 {len(result['plates'])} 个车牌:")
    for i, plate in enumerate(result['plates']):
        bbox = plate['bbox']
        text = plate['text']
        conf = plate['confidence']
        print(f"  车牌 {i+1}:")
        print(f"    位置: ({bbox[0]}, {bbox[1]}) - ({bbox[2]}, {bbox[3]})")
        print(f"    车牌号: {text}")
        print(f"    置信度: {conf:.2f}")
    
    if 'result_image' in result:
        save_image(result['result_image'], output_path)
        print(f"\n结果已保存到: {output_path}")
        
        if args.show:
            cv2.imshow('License Plate Recognition', result['result_image'])
            print("按任意键关闭窗口...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    
    print("\n处理完成！")


if __name__ == '__main__':
    main()
