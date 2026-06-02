# CV 课程作业（2）：边缘检测

Python + OpenCV 实现五种经典边缘检测算子。

## 功能

| # | 功能 | 算法 |
|---|------|------|
| 1 | Sobel 算子 | 一阶梯度（X方向、Y方向、幅度、角度） |
| 2 | Laplacian 算子 | 二阶导数边缘检测 |
| 3 | DoG (Difference of Gaussians) | 高斯差分（LoG 的高效近似） |
| 4 | LoG (Laplacian of Gaussian) | 先高斯平滑再拉普拉斯 |
| 5 | Canny 算子 | 多阶段优化（NMS + 双阈值 + 滞后连接） |

## 运行

```bash
pip install -r requirements.txt
python main.py <image_path>
```

## 输出

| 文件 | 内容 |
|------|------|
| `output_1_sobel.png` | Sobel 四象限（X梯度 / Y梯度 / 梯度幅度 / 梯度角度） |
| `output_2_all_edges.png` | 2×3 五种算子 + 原始图像对比 |
| `output_3_comprehensive.png` | 3×4 综合分析大图 |

## 依赖

- Python 3.8+
- OpenCV (`opencv-python`)
- NumPy
- Matplotlib
