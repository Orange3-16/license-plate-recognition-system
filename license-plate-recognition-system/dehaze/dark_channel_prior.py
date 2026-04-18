import cv2
import numpy as np


def get_dark_channel(image, radius=7):
    """
    计算暗通道
    
    Args:
        image: 输入BGR图像
        radius: 暗通道计算半径
        
    Returns:
        暗通道图像
    """
    min_channel = np.min(image, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * radius + 1, 2 * radius + 1))
    dark_channel = cv2.erode(min_channel, kernel)
    return dark_channel


def estimate_atmospheric_light(image, dark_channel, percentile=0.001):
    """
    估计大气光照值
    
    Args:
        image: 输入BGR图像
        dark_channel: 暗通道图像
        percentile: 亮度分位数
        
    Returns:
        大气光照值
    """
    h, w = dark_channel.shape
    num_pixels = h * w
    num_brightest = int(max(num_pixels * percentile, 1))
    
    dark_channel_vec = dark_channel.reshape(num_pixels)
    indices = np.argpartition(dark_channel_vec, -num_brightest)[-num_brightest:]
    
    indices = np.unravel_index(indices, dark_channel.shape)
    
    brightest_pixels = image[indices]
    
    atmospheric_light = np.max(brightest_pixels, axis=0)
    
    return atmospheric_light


def guided_filter(I, p, radius=60, epsilon=1e-3):
    """
    导向滤波
    
    Args:
        I: 引导图像 (单通道)
        p: 输入图像 (单通道)
        radius: 滤波半径
        epsilon: 正则化参数
        
    Returns:
        滤波后的图像
    """
    mean_I = cv2.boxFilter(I, cv2.CV_64F, (radius, radius))
    mean_p = cv2.boxFilter(p, cv2.CV_64F, (radius, radius))
    
    mean_Ip = cv2.boxFilter(I * p, cv2.CV_64F, (radius, radius))
    cov_Ip = mean_Ip - mean_I * mean_p
    
    mean_II = cv2.boxFilter(I * I, cv2.CV_64F, (radius, radius))
    var_I = mean_II - mean_I * mean_I
    
    a = cov_Ip / (var_I + epsilon)
    b = mean_p - a * mean_I
    
    mean_a = cv2.boxFilter(a, cv2.CV_64F, (radius, radius))
    mean_b = cv2.boxFilter(b, cv2.CV_64F, (radius, radius))
    
    q = mean_a * I + mean_b
    
    return q


def recover_radiance(image, transmission, atmospheric_light, t0=0.1):
    """
    恢复场景辐射
    
    Args:
        image: 输入BGR图像
        transmission: 透射率图
        atmospheric_light: 大气光照值
        t0: 最小透射率阈值
        
    Returns:
        去雾后的图像
    """
    transmission = np.maximum(transmission, t0)
    
    transmission = transmission[:, :, np.newaxis]
    atmospheric_light = atmospheric_light[np.newaxis, np.newaxis, :]
    
    radiance = (image - atmospheric_light) / transmission + atmospheric_light
    
    radiance = np.clip(radiance, 0, 255)
    
    return radiance.astype(np.uint8)


def dehaze(image, omega=0.95, t0=0.1, radius=7, use_guided_filter=True, guided_radius=60, guided_epsilon=1e-3):
    """
    暗通道先验去雾算法
    
    Args:
        image: 输入BGR图像
        omega: 去雾强度 (0-1)
        t0: 最小透射率阈值
        radius: 暗通道计算半径
        use_guided_filter: 是否使用导向滤波
        guided_radius: 导向滤波半径
        guided_epsilon: 导向滤波正则化参数
        
    Returns:
        去雾后的BGR图像
    """
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    image_float = image.astype(np.float64)
    
    dark_channel = get_dark_channel(image_float, radius)
    
    atmospheric_light = estimate_atmospheric_light(image_float, dark_channel)
    
    # 避免除零错误
    atmospheric_light = np.maximum(atmospheric_light, 1e-6)
    
    normalized_image = image_float / atmospheric_light
    
    transmission = 1 - omega * get_dark_channel(normalized_image, radius)
    
    if use_guided_filter:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float64) / 255.0
        transmission = guided_filter(gray_image, transmission, guided_radius, guided_epsilon)
    
    dehazed_image = recover_radiance(image_float, transmission, atmospheric_light, t0)
    
    return dehazed_image


def dehaze_simple(image, omega=0.95, t0=0.1, radius=7):
    """
    简化版暗通道先验去雾算法（不使用导向滤波）
    
    Args:
        image: 输入BGR图像
        omega: 去雾强度 (0-1)
        t0: 最小透射率阈值
        radius: 暗通道计算半径
        
    Returns:
        去雾后的BGR图像
    """
    return dehaze(image, omega, t0, radius, use_guided_filter=False)


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("使用方法: python dark_channel_prior.py <input_image> [output_image]")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else 'dehazed_result.jpg'
    
    image = cv2.imread(input_path)
    if image is None:
        print(f"无法读取图像: {input_path}")
        sys.exit(1)
    
    print("正在去雾...")
    dehazed_image = dehaze(image)
    
    cv2.imwrite(output_path, dehazed_image)
    print(f"去雾完成，结果已保存到: {output_path}")
