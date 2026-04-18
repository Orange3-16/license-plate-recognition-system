import os
import argparse
import random
from pathlib import Path


PROVINCES = [
    '皖', '沪', '津', '渝', '冀', '晋', '蒙', '辽', '吉', '黑',
    '苏', '浙', '皖', '闽', '赣', '鲁', '豫', '鄂', '湘', '粤',
    '桂', '琼', '川', '贵', '云', '藏', '陕', '甘', '青', '宁', '新'
]

ALPHABETS = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
    'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
    'W', 'X', 'Y', 'Z'
]

ADS = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'
]


def decode_plate(label_str):
    """
    解码CCPD车牌编码为真实车牌号
    
    Args:
        label_str: 车牌编码字符串，如 "0_0_22_27_27_33_16"
        
    Returns:
        车牌号码字符串，如 "皖ASXX9G"
    """
    labels = label_str.split('_')
    
    plate = ""
    
    if len(labels) >= 1:
        province_idx = int(labels[0])
        if 0 <= province_idx < len(PROVINCES):
            plate += PROVINCES[province_idx]
    
    if len(labels) >= 2:
        alphabet_idx = int(labels[1])
        if 0 <= alphabet_idx < len(ALPHABETS):
            plate += ALPHABETS[alphabet_idx]
    
    if len(labels) >= 7:
        for i in range(2, 7):
            ads_idx = int(labels[i])
            if 0 <= ads_idx < len(ADS):
                plate += ADS[ads_idx]
    
    return plate


def parse_filename(filename):
    """
    解析CCPD文件名，提取车牌信息
    
    Args:
        filename: 文件名，如 "025-95_113-154&383_386&473-386&473_177&454_154&383_363&402-0_0_22_27_27_33_16-37-15.jpg"
        
    Returns:
        dict: 包含边界框、车牌号码等信息的字典
    """
    name = Path(filename).stem
    parts = name.split('-')
    
    result = {
        'bbox': None,
        'plate': None,
        'label_str': None
    }
    
    # 解析边界框信息（第3部分）
    if len(parts) >= 3:
        bbox_str = parts[2]
        bbox_parts = bbox_str.split('_')
        
        if len(bbox_parts) == 2:
            left_top = bbox_parts[0].split('&')
            right_bottom = bbox_parts[1].split('&')
            
            if len(left_top) == 2 and len(right_bottom) == 2:
                try:
                    x1 = int(left_top[0])
                    y1 = int(left_top[1])
                    x2 = int(right_bottom[0])
                    y2 = int(right_bottom[1])
                    result['bbox'] = (x1, y1, x2, y2)
                except ValueError:
                    pass
    
    # 查找包含车牌编码的部分（通常是第5部分，也可能是第4部分）
    for i in range(3, min(len(parts), 6)):
        label_str = parts[i]
        labels = label_str.split('_')
        
        # 车牌编码通常有7-8个部分
        if 7 <= len(labels) <= 8:
            try:
                # 检查是否都是数字
                for label in labels:
                    int(label)
                result['label_str'] = label_str
                result['plate'] = decode_plate(label_str)
                break
            except ValueError:
                continue
    
    return result


def create_yolo_label(image_path, bbox, image_width, image_height):
    """
    创建YOLO格式的标签
    
    Args:
        image_path: 图片路径
        bbox: 边界框 (x1, y1, x2, y2)
        image_width: 图片宽度
        image_height: 图片高度
        
    Returns:
        str: YOLO格式标签字符串
    """
    x1, y1, x2, y2 = bbox
    
    x_center = (x1 + x2) / 2.0 / image_width
    y_center = (y1 + y2) / 2.0 / image_height
    width = (x2 - x1) / image_width
    height = (y2 - y1) / image_height
    
    return f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"


def process_ccpd_dataset(ccpd_root, output_dir, train_ratio=0.9):
    """
    处理CCPD数据集，生成训练和验证集文件
    
    Args:
        ccpd_root: CCPD数据集根目录
        output_dir: 输出目录
        train_ratio: 训练集比例
    """
    ccpd_root = Path(ccpd_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 获取项目根目录
    root_dir = Path(__file__).parent.parent
    
    image_info_list = []
    
    print(f"正在扫描CCPD数据集: {ccpd_root}")
    print(f"当前工作目录: {Path.cwd()}")
    
    for img_file in ccpd_root.rglob('*.jpg'):
        try:
            print(f"Found file: {img_file}")
            info = parse_filename(img_file.name)
            
            if info['bbox'] is not None and info['plate'] is not None:
                # 构建相对路径：使用os.path.relpath
                import os
                relative_path = os.path.relpath(img_file, Path.cwd())
                image_info = {
                    'path': relative_path,
                    'bbox': info['bbox'],
                    'plate': info['plate']
                }
                image_info_list.append(image_info)
        except Exception as e:
            print(f"处理文件 {img_file} 时出错: {e}")
            continue
    
    print(f"共找到 {len(image_info_list)} 张有效图片")
    
    random.shuffle(image_info_list)
    
    split_idx = int(len(image_info_list) * train_ratio)
    train_list = image_info_list[:split_idx]
    val_list = image_info_list[split_idx:]
    
    print(f"训练集: {len(train_list)} 张")
    print(f"验证集: {len(val_list)} 张")
    
    train_txt_path = output_dir / 'train.txt'
    val_txt_path = output_dir / 'val.txt'
    
    # 为CRNN创建标注文件（包含路径和标签）
    crnn_train_path = output_dir / 'crnn_train.txt'
    crnn_val_path = output_dir / 'crnn_val.txt'
    
    with open(crnn_train_path, 'w', encoding='utf-8') as f:
        for info in train_list:
            f.write(f"{info['path']} {info['plate']}\n")
    
    with open(crnn_val_path, 'w', encoding='utf-8') as f:
        for info in val_list:
            f.write(f"{info['path']} {info['plate']}\n")
    
    # 为YOLO创建标注文件（只包含路径）
    yolo_train_path = output_dir / 'train.txt'
    yolo_val_path = output_dir / 'val.txt'
    
    with open(yolo_train_path, 'w', encoding='utf-8') as f:
        for info in train_list:
            f.write(f"{info['path']}\n")
    
    with open(yolo_val_path, 'w', encoding='utf-8') as f:
        for info in val_list:
            f.write(f"{info['path']}\n")
    
    print(f"识别任务标注文件已生成:")
    print(f"  - {crnn_train_path} (CRNN训练)")
    print(f"  - {crnn_val_path} (CRNN验证)")
    print(f"  - {yolo_train_path} (YOLO训练)")
    print(f"  - {yolo_val_path} (YOLO验证)")
    
    print("\n正在生成YOLO格式标签文件...")
    
    import cv2
    
    for info in image_info_list:
        # 直接使用info['path']作为完整路径
        img_path = Path(info['path'])
        
        try:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            
            h, w = img.shape[:2]
            
            label_dir = img_path.parent
            label_path = label_dir / f"{img_path.stem}.txt"
            
            yolo_label = create_yolo_label(info['path'], info['bbox'], w, h)
            
            with open(label_path, 'w', encoding='utf-8') as f:
                f.write(yolo_label)
                
        except Exception as e:
            print(f"生成标签文件 {img_path} 时出错: {e}")
            continue
    
    print("YOLO格式标签文件已生成")
    
    yolo_data_dir = output_dir / 'yolo_data'
    yolo_data_dir.mkdir(parents=True, exist_ok=True)
    
    yolo_yaml_content = f"""path: {ccpd_root.absolute()}
train: train.txt
val: val.txt

names:
  0: license_plate
"""
    
    yolo_yaml_path = yolo_data_dir / 'data.yaml'
    with open(yolo_yaml_path, 'w', encoding='utf-8') as f:
        f.write(yolo_yaml_content)
    
    print(f"YOLO数据配置文件已生成: {yolo_yaml_path}")


def main():
    parser = argparse.ArgumentParser(description='处理CCPD数据集')
    parser.add_argument('--ccpd_root', type=str, default='CCPD2020',
                        help='CCPD数据集根目录')
    parser.add_argument('--output_dir', type=str, default='data/processed',
                        help='输出目录')
    parser.add_argument('--train_ratio', type=float, default=0.9,
                        help='训练集比例 (默认: 0.9)')
    
    args = parser.parse_args()
    
    process_ccpd_dataset(args.ccpd_root, args.output_dir, args.train_ratio)


if __name__ == '__main__':
    main()
