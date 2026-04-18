import os
import argparse
from pathlib import Path
from ultralytics import YOLO


def train_yolo(args):
    """
    训练YOLOv8车牌检测模型
    """
    data_yaml = args.data_yaml
    
    if not os.path.exists(data_yaml):
        print(f"数据配置文件不存在: {data_yaml}")
        return
    
    print(f"使用数据配置文件: {data_yaml}")
    
    model_size = args.model_size
    
    model = YOLO(f'{model_size}.pt')
    
    print(f"加载预训练模型: {model_size}.pt")
    
    results = model.train(
        data=data_yaml,
        epochs=args.epochs,
        batch=args.batch_size,
        imgsz=args.img_size,
        device=args.device,
        project=args.project,
        name=args.name,
        exist_ok=True,
        pretrained=True,
        optimizer=args.optimizer,
        lr0=args.lr0,
        lrf=args.lrf,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        patience=args.patience,
        save=args.save,
        plots=args.plots,
        verbose=args.verbose
    )
    
    print(f"训练完成！模型保存在: {args.project}/{args.name}")
    
    best_model_path = Path(args.project) / args.name / 'weights' / 'best.pt'
    if best_model_path.exists():
        print(f"最佳模型: {best_model_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='训练YOLOv8车牌检测模型')
    parser.add_argument('--data_yaml', type=str, default='data/processed/yolo_data/data.yaml',
                        help='YOLO数据配置文件')
    parser.add_argument('--model_size', type=str, default='yolov8n',
                        choices=['yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x'],
                        help='YOLOv8模型大小')
    parser.add_argument('--epochs', type=int, default=100,
                        help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='批次大小')
    parser.add_argument('--img_size', type=int, default=640,
                        help='输入图像大小')
    parser.add_argument('--device', type=str, default='0',
                        help='设备 (0表示GPU 0, cpu表示CPU)')
    parser.add_argument('--project', type=str, default='detection/runs',
                        help='项目目录')
    parser.add_argument('--name', type=str, default='train',
                        help='实验名称')
    parser.add_argument('--optimizer', type=str, default='auto',
                        choices=['SGD', 'Adam', 'AdamW', 'auto'],
                        help='优化器')
    parser.add_argument('--lr0', type=float, default=0.01,
                        help='初始学习率')
    parser.add_argument('--lrf', type=float, default=0.01,
                        help='最终学习率因子')
    parser.add_argument('--momentum', type=float, default=0.937,
                        help='SGD动量/Adam beta1')
    parser.add_argument('--weight_decay', type=float, default=0.0005,
                        help='权重衰减')
    parser.add_argument('--warmup_epochs', type=int, default=3,
                        help='预热轮数')
    parser.add_argument('--patience', type=int, default=50,
                        help='早停耐心值')
    parser.add_argument('--save', action='store_true', default=True,
                        help='保存训练检查点')
    parser.add_argument('--plots', action='store_true', default=True,
                        help='保存训练曲线')
    parser.add_argument('--verbose', action='store_true', default=True,
                        help='详细输出')
    
    args = parser.parse_args()
    
    Path(args.project).mkdir(parents=True, exist_ok=True)
    
    train_yolo(args)


if __name__ == '__main__':
    main()
