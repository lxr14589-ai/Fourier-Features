import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image
from datetime import datetime
from pathlib import Path
import csv

# ==================== 可调参数 ====================
SEED = 42
USE_FOURIER_FEATURES = True
NUM_FREQUENCIES = 128
SIGMA = 100
CROP_SIZE = 256
LEARNING_RATE = 0.01
MAX_EPOCHS = 30000
EARLY_STOPPING_PATIENCE = 2000
MIN_DELTA = 1e-6
LR_FACTOR = 0.5
LR_PATIENCE = 200
MIN_LEARNING_RATE = 1e-5
PRINT_INTERVAL = 500

# 固定随机种子
torch.manual_seed(SEED)

# 自动选择 GPU；没有 CUDA 时回退到 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))

# 1. 从原图中心裁剪 128x128，使用全部像素作为训练数据
image = Image.open("test.jpg").convert("RGB")
CROP_SIZE = int(CROP_SIZE)
if image.width < CROP_SIZE or image.height < CROP_SIZE:
    raise ValueError(f"test.jpg 的尺寸必须至少为 {CROP_SIZE}x{CROP_SIZE}")

left = (image.width - CROP_SIZE) // 2
top = (image.height - CROP_SIZE) // 2
image = image.crop((left, top, left + CROP_SIZE, top + CROP_SIZE))
pixels = np.asarray(image, dtype=np.float32) / 255.0
height, width, _ = pixels.shape

# 使用完整 128x128 网格，共 16384 个训练样本
x = np.arange(width, dtype=np.float32) / (width - 1)
y = np.arange(height, dtype=np.float32) / (height - 1)
xx, yy = np.meshgrid(x, y)
coords = np.stack([xx, yy], axis=-1)

# 2. 根据开关决定是否使用 Fourier Features
X_coordinates = torch.from_numpy(coords.reshape(-1, 2)).float().to(device)
if USE_FOURIER_FEATURES:
    B = (SIGMA * torch.randn(NUM_FREQUENCIES, 2)).to(device)

    def fourier_features(coordinates):
        angles = 2 * torch.pi * (coordinates @ B.T)
        return torch.cat([torch.cos(angles), torch.sin(angles)], dim=1)

    X = fourier_features(X_coordinates)
    input_dim = 2 * NUM_FREQUENCIES
    print(f"Using Fourier Features: {input_dim} dimensions, sigma={SIGMA}")
else:
    X = X_coordinates
    input_dim = 2
    print("Using standard 2D coordinates")

Y = torch.from_numpy(pixels.reshape(-1, 3)).float()

# 2. 定义 MLP
model = nn.Sequential(
    nn.Linear(input_dim, 128),
    nn.ReLU(),
    nn.Linear(128, 128),
    nn.ReLU(),
    nn.Linear(128, 128),
    nn.ReLU(),
    nn.Linear(128, 3)
).to(device)

Y = Y.to(device)

# 3. 损失函数和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
# 当 Loss 长时间没有改善时自动降低学习率
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=LR_FACTOR,
    patience=LR_PATIENCE,
    min_lr=MIN_LEARNING_RATE
)

# 4. 训练
loss_history = []
best_loss = float("inf")
best_state = None
epochs_without_improvement = 0

for epoch in range(MAX_EPOCHS):
    optimizer.zero_grad()
    prediction = model(X)
    loss = criterion(prediction, Y)
    loss.backward()
    optimizer.step()
    current_loss = loss.item()
    scheduler.step(current_loss)
    loss_history.append(current_loss)

    if current_loss < best_loss - MIN_DELTA:
        best_loss = current_loss
        best_state = {
            name: parameter.detach().clone()
            for name, parameter in model.state_dict().items()
        }
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1

    if (epoch + 1) % PRINT_INTERVAL == 0:
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch + 1}/{MAX_EPOCHS}, Loss: {current_loss:.6f}, LR: {current_lr:.6f}")

    if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
        print(f"Early stopping at epoch {epoch + 1}; best loss: {best_loss:.6f}")
        break

if best_state is not None:
    model.load_state_dict(best_state)

# 5. 在完整 128x128 网格上预测，重建图片
# 训练和重建使用同一个 B 矩阵进行 Fourier 映射
with torch.no_grad():
    X_test = fourier_features(X_coordinates) if USE_FOURIER_FEATURES else X_coordinates
    prediction = model(X_test)

reconstructed = prediction.cpu().numpy()
reconstructed = reconstructed.reshape(height, width, 3)
reconstructed = np.clip(reconstructed, 0, 1)

# 计算重建图像与原图之间的均方误差（MSE）
mse = np.mean((pixels - reconstructed) ** 2)
print(f"Reconstruction MSE: {mse:.8f}")

# 每次运行生成独立的日志文件
run_time = datetime.now().astimezone()
run_timestamp = run_time.strftime("%Y%m%d_%H%M%S_%f")
log_file = Path(f"training_log_{run_timestamp}.csv")
log_row = {
    "timestamp": run_time.isoformat(timespec="seconds"),
    "device": str(device),
    "torch_version": torch.__version__,
    "use_fourier_features": USE_FOURIER_FEATURES,
    "num_frequencies": NUM_FREQUENCIES if USE_FOURIER_FEATURES else 0,
    "sigma": SIGMA if USE_FOURIER_FEATURES else 0,
    "crop_size": CROP_SIZE,
    "learning_rate": LEARNING_RATE,
    "max_epochs": MAX_EPOCHS,
    "actual_epochs": len(loss_history),
    "final_loss": float(best_loss),
    "mse": float(mse),
}

with log_file.open("a", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=log_row.keys())
    writer.writeheader()
    writer.writerow(log_row)
print(f"Log saved to: {log_file}")

# 6. 显示原图和重建图
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.imshow(pixels)
plt.title("Original")
plt.axis("off")
plt.subplot(1, 2, 2)
plt.imshow(reconstructed)
plt.title("MLP Reconstruction")
plt.axis("off")
plt.show()

# 7. 显示训练 Loss 曲线
plt.figure(figsize=(8, 4))
plt.plot(loss_history)
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Training Loss")
plt.grid(True)
plt.show()
