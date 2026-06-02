"""
CV课程作业（2）：边缘检测
实现：Sobel、拉普拉斯、DoG、LoG、Canny 五种边缘检测算子
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rcParams

rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False


def load_image(path, gray=True):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"无法加载图像: {path}")
    if gray:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


# ==================== Sobel 算子 ====================

def sobel_edge(img):
    """
    Sobel 算子：计算 x 方向梯度、y 方向梯度、梯度幅度、梯度角度
    使用 Scharr 核的 3x3 Sobel（cv2.CV_64F 避免截断负梯度）
    """
    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)

    # 梯度幅度
    magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    # 梯度角度 (弧度)
    angle = np.arctan2(sobel_y, sobel_x)

    # 归一化到 [0, 255] 用于显示
    sobel_x_abs = np.uint8(np.abs(sobel_x) / np.max(np.abs(sobel_x)) * 255)
    sobel_y_abs = np.uint8(np.abs(sobel_y) / np.max(np.abs(sobel_y)) * 255)
    magnitude_abs = np.uint8(magnitude / np.max(magnitude) * 255)

    return {
        'x': sobel_x, 'y': sobel_y,
        'magnitude': magnitude,
        'angle': angle,
        'x_display': sobel_x_abs,
        'y_display': sobel_y_abs,
        'magnitude_display': magnitude_abs,
    }


# ==================== 拉普拉斯算子 ====================

def laplacian_edge(img):
    """拉普拉斯算子：二阶导数，对噪声敏感，需要先平滑"""
    lap = cv2.Laplacian(img, cv2.CV_64F, ksize=3)
    lap_abs = np.uint8(np.abs(lap) / np.max(np.abs(lap)) * 255)
    return lap_abs


# ==================== LoG (Laplacian of Gaussian) ====================

def log_edge(img, gauss_ksize=5, sigma=1.0):
    """
    LoG：先高斯平滑抑制噪声，再施加拉普拉斯算子
    等价于对高斯函数的拉普拉斯核做一次卷积
    """
    blurred = cv2.GaussianBlur(img, (gauss_ksize, gauss_ksize), sigma)
    log_result = cv2.Laplacian(blurred, cv2.CV_64F, ksize=3)
    log_abs = np.uint8(np.abs(log_result) / np.max(np.abs(log_result)) * 255)
    return log_abs


# ==================== DoG (Difference of Gaussians) ====================

def dog_edge(img, sigma1=1.0, sigma2=2.0):
    """
    DoG：两个不同尺度的高斯平滑结果之差
    DoG = G(x,y,σ1) - G(x,y,σ2)，是 LoG 的高效近似
    """
    g1 = cv2.GaussianBlur(img, (0, 0), sigma1)  # ksize=0 自动从 sigma 计算
    g2 = cv2.GaussianBlur(img, (0, 0), sigma2)
    dog = g1.astype(np.float32) - g2.astype(np.float32)
    dog_abs = np.uint8(np.abs(dog) / np.max(np.abs(dog)) * 255)
    return dog_abs


# ==================== Canny 算子 ====================

def canny_edge(img, low=50, high=150):
    """
    Canny 算子：多阶段流程
    1. 高斯平滑 (5x5)
    2. 梯度计算 (Sobel)
    3. 非极大值抑制 (NMS)
    4. 双阈值 + 边缘连接 (滞后阈值)
    """
    return cv2.Canny(img, low, high)


# ==================== 可视化 ====================

def show_sobel_results(img, sobel):
    """Sobel 结果：2x2 子图"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    axes[0, 0].imshow(sobel['x_display'], cmap='gray')
    axes[0, 0].set_title('Sobel X 方向梯度', fontsize=13, fontweight='bold')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(sobel['y_display'], cmap='gray')
    axes[0, 1].set_title('Sobel Y 方向梯度', fontsize=13, fontweight='bold')
    axes[0, 1].axis('off')

    axes[1, 0].imshow(sobel['magnitude_display'], cmap='gray')
    axes[1, 0].set_title('Sobel 梯度幅度', fontsize=13, fontweight='bold')
    axes[1, 0].axis('off')

    # 梯度角度用彩色显示 (hsv 映射)
    angle_display = (sobel['angle'] + np.pi) / (2 * np.pi)  # 归一化到 [0,1]
    angle_colored = plt.cm.hsv(angle_display)[:, :, :3]
    # 用梯度幅度调制角度颜色饱和度
    mag_norm = sobel['magnitude'] / np.max(sobel['magnitude'])
    angle_colored = angle_colored * mag_norm[:, :, np.newaxis]
    axes[1, 1].imshow(angle_colored)
    axes[1, 1].set_title('Sobel 梯度角度 (色相=方向, 亮度=幅度)', fontsize=13, fontweight='bold')
    axes[1, 1].axis('off')

    fig.suptitle('Sobel 算子边缘检测结果', fontsize=16, fontweight='bold')
    plt.tight_layout()
    return fig


def show_all_edges(img, sobel_mag, laplacian, dog, log, canny):
    """所有方法对比：2x3 子图"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    items = [
        (axes[0, 0], img, '原始图像', None),
        (axes[0, 1], sobel_mag, 'Sobel (梯度幅度)', 'gray'),
        (axes[0, 2], laplacian, '拉普拉斯 (Laplacian)', 'gray'),
        (axes[1, 0], dog, 'DoG (高斯差分)', 'gray'),
        (axes[1, 1], log, 'LoG (高斯拉普拉斯)', 'gray'),
        (axes[1, 2], canny, 'Canny', 'gray'),
    ]
    for ax, im, title, cmap in items:
        ax.imshow(im, cmap=cmap)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.axis('off')

    fig.suptitle('五种边缘检测算子对比', fontsize=16, fontweight='bold')
    plt.tight_layout()
    return fig


def show_comprehensive(img, sobel, laplacian, dog, log, canny):
    """综合分析大图：3x4 布局"""
    fig = plt.figure(figsize=(20, 16))

    positions = [
        (img, '原始图像', 0, None),
        (sobel['x_display'], 'Sobel X 梯度', 1, 'gray'),
        (sobel['y_display'], 'Sobel Y 梯度', 2, 'gray'),
        (sobel['magnitude_display'], 'Sobel 梯度幅度', 3, 'gray'),

        (laplacian, 'Laplacian (拉普拉斯)', 4, 'gray'),
        (log, 'LoG (高斯拉普拉斯)', 5, 'gray'),
        (dog, 'DoG (高斯差分)', 6, 'gray'),
        (canny, 'Canny', 7, 'gray'),

        (img, '原始图像', 8, None),
        (laplacian, 'Laplacian 边缘', 9, 'gray'),
        (canny, 'Canny 边缘', 10, 'gray'),
        (sobel['magnitude_display'], 'Sobel 边缘', 11, 'gray'),
    ]

    for im, title, idx, cmap in positions:
        ax = fig.add_subplot(3, 4, idx + 1)
        ax.imshow(im, cmap=cmap)
        ax.set_title(title, fontsize=11)
        ax.axis('off')

    fig.suptitle('边缘检测综合分析', fontsize=18, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    return fig


# ==================== 主流程 ====================

def main(image_path):
    img = load_image(image_path, gray=True)
    print(f"图像尺寸: {img.shape}")

    # ---- 1. Sobel 算子 ----
    sobel = sobel_edge(img)
    fig1 = show_sobel_results(img, sobel)
    fig1.savefig('output_1_sobel.png', dpi=150, bbox_inches='tight')
    print("1/6 Sobel 算子完成")

    # ---- 2. 拉普拉斯算子 ----
    laplacian = laplacian_edge(img)
    print("2/6 拉普拉斯算子完成")

    # ---- 3. DoG ----
    dog = dog_edge(img, sigma1=1.0, sigma2=2.0)
    print("3/6 DoG 完成")

    # ---- 4. LoG ----
    log = log_edge(img, gauss_ksize=5, sigma=1.0)
    print("4/6 LoG 完成")

    # ---- 5. Canny ----
    canny = canny_edge(img, low=50, high=150)
    print("5/6 Canny 完成")

    # ---- 6. 综合对比 ----
    fig2 = show_all_edges(img, sobel['magnitude_display'], laplacian, dog, log, canny)
    fig2.savefig('output_2_all_edges.png', dpi=150, bbox_inches='tight')
    print("6/6 综合对比完成")

    # ---- 额外：综合分析大图 ----
    fig3 = show_comprehensive(img, sobel, laplacian, dog, log, canny)
    fig3.savefig('output_3_comprehensive.png', dpi=150, bbox_inches='tight')
    print("综合分析图完成")

    plt.show()
    print("所有结果已保存")


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        print("请提供图像路径: python main.py <image_path>")
        sys.exit(1)
    main(path)
