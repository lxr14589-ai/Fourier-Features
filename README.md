# Fourier Features Image Reconstruction

使用 MLP 学习二维坐标到 RGB 颜色的映射，并通过 Fourier Features 提升图像细节重建能力。

## 功能

- 从原图中心裁剪 `128×128` 区域
- 使用完整像素坐标作为训练数据，不使用插值或降采样
- 可切换普通二维坐标和 Fourier Features
- 自动使用 CUDA GPU；没有 CUDA 时回退到 CPU
- 使用 MSE Loss、Adam、学习率调度和 Early Stopping
- 每次运行单独生成一个 CSV 日志文件

## 环境安装

建议使用 Python 3.13 虚拟环境：

```powershell
python -m venv venv
& .\venv\Scripts\Activate.ps1
python -m pip install numpy matplotlib pillow
```

CUDA 版 PyTorch（NVIDIA GPU）：

```powershell
python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
```

## 运行

确保 `test.jpg` 位于项目根目录，然后运行：

```powershell
& .\venv\Scripts\python.exe .\1D-Baseline.py
```

程序会显示原图、重建图和训练 Loss 曲线，并在终端输出最终 Loss 与 MSE。

## 参数配置

所有主要参数位于 `1D-Baseline.py` 文件开头：

```python
USE_FOURIER_FEATURES = True
NUM_FREQUENCIES = 128
SIGMA = 5.0
CROP_SIZE = 128
LEARNING_RATE = 0.01
MAX_EPOCHS = 30000
```

将 `USE_FOURIER_FEATURES` 改为 `False` 即可使用普通二维坐标输入。

## 日志

每次运行都会生成独立文件：

```text
training_log_YYYYMMDD_HHMMSS_microseconds.csv
```

日志记录运行参数、实际训练轮数、最终 Loss 和重建 MSE。
