# -*- coding: utf-8 -*-
import os
import time
import glob
import math  # [新增] 需要用到 math 库计算 log
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
from fractions import Fraction
from tensorboardX import SummaryWriter
from torch.nn.parallel import DataParallel
from PIL import Image
from sklearn.model_selection import train_test_split

# === 依赖本地文件 (model.py, utils.py) ===
from model import DeepJSCC, ratio2filtersize
from utils import image_normalization, set_seed


# ==========================================
# 1. 配置参数
# ==========================================
def config_parser():
    import argparse
    parser = argparse.ArgumentParser()

    # --- 数据集路径 ---
    parser.add_argument('--data_root', default='./data/military_object_dataset/train/images', type=str,
                        help='Path to the Military dataset root folder')

    # --- 训练参数 ---
    parser.add_argument('--snr_list', default=['4'], nargs='+', help='List of SNR values')
    parser.add_argument('--ratio_list', default=['1/24'], nargs='+', help='List of compression ratios')
    parser.add_argument('--out', default='./out_military', type=str, help='Output directory')
    parser.add_argument('--image_size', default=256, type=int, help='Resize images to this size')
    parser.add_argument('--batch_size', default=16, type=int, help='Batch size')
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs')
    parser.add_argument('--init_lr', default=1e-4, type=float, help='Initial learning rate')

    parser.add_argument('--channel', default='AWGN', type=str, choices=['AWGN', 'Rayleigh'])
    parser.add_argument('--device', default='cuda:0', type=str)
    parser.add_argument('--parallel', action='store_true', help='Use DataParallel for multiple GPUs')
    parser.add_argument('--disable_tqdm', action='store_true')
    parser.add_argument('--seed', default=42, type=int)

    return parser.parse_args()


# ==========================================
# 2. 数据集类
# ==========================================
class MilitaryDataset(Dataset):
    def __init__(self, root, transform=None, train=True, test_split=0.1, seed=42):
        self.transform = transform
        self.files = []
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff']

        print(f"[{'Train' if train else 'Test'}] Scanning images in: {root}")
        for ext in extensions:
            self.files.extend(glob.glob(os.path.join(root, '**', ext), recursive=True))

        self.files = sorted(list(set(self.files)))
        self.files = [f for f in self.files if os.path.getsize(f) > 0]

        if len(self.files) == 0:
            raise RuntimeError(f"No images found in {root}. Please check the path.")

        train_files, val_files = train_test_split(self.files, test_size=test_split, random_state=seed)
        self.data = train_files if train else val_files
        print(f"Found {len(self.data)} images for {'training' if train else 'testing'}.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading {img_path}: {e}, skipping...")
            return self.__getitem__((idx + 1) % len(self))

        if self.transform:
            image = self.transform(image)
        return image, 0


# ==========================================
# 3. 训练与评估函数 (已修改：增加PSNR计算)
# ==========================================
def train_epoch(model, optimizer, param, data_loader):
    model.train()
    epoch_loss = 0
    epoch_psnr = 0  # [新增] 累积PSNR

    loop = tqdm(data_loader, disable=param['disable_tqdm'], leave=False)

    for i, (images, _) in enumerate(loop):
        images = images.to(param['device'])
        optimizer.zero_grad()
        outputs = model(images)

        # 反归一化 (通常还原到 0-255)
        outputs_denorm = image_normalization('denormalization')(outputs)
        images_denorm = image_normalization('denormalization')(images)

        # 计算 Loss
        if hasattr(model, 'loss'):
            loss = model.loss(images_denorm, outputs_denorm)
        elif hasattr(model, 'module') and hasattr(model.module, 'loss'):
            loss = model.module.loss(images_denorm, outputs_denorm)
        else:
            criterion = nn.MSELoss()
            loss = criterion(images_denorm, outputs_denorm)

        loss.backward()
        optimizer.step()

        # [新增] 计算当前 Batch 的 PSNR
        # PSNR = 10 * log10(MAX^2 / MSE)
        # 假设 denormalization 后是 0-255，则 MAX=255
        mse_val = loss.item()
        if mse_val > 0:
            batch_psnr = 10 * math.log10(255 ** 2 / mse_val)
        else:
            batch_psnr = 100  # 防止 MSE=0 的情况

        epoch_loss += mse_val
        epoch_psnr += batch_psnr

        # 更新进度条
        loop.set_description(f"Train Loss: {mse_val:.2f} | PSNR: {batch_psnr:.2f}")

    avg_loss = epoch_loss / len(data_loader)
    avg_psnr = epoch_psnr / len(data_loader)

    return avg_loss, avg_psnr, optimizer  # [修改] 返回增加 PSNR


def evaluate_epoch(model, param, data_loader):
    model.eval()
    epoch_loss = 0
    epoch_psnr = 0  # [新增]

    criterion = nn.MSELoss()

    with torch.no_grad():
        for images, _ in data_loader:
            images = images.to(param['device'])
            outputs = model(images)

            outputs_denorm = image_normalization('denormalization')(outputs)
            images_denorm = image_normalization('denormalization')(images)

            if hasattr(model, 'loss'):
                loss = model.loss(images_denorm, outputs_denorm)
            elif hasattr(model, 'module') and hasattr(model.module, 'loss'):
                loss = model.module.loss(images_denorm, outputs_denorm)
            else:
                loss = criterion(images_denorm, outputs_denorm)

            # [新增] 计算 PSNR
            mse_val = loss.item()
            if mse_val > 0:
                batch_psnr = 10 * math.log10(255 ** 2 / mse_val)
            else:
                batch_psnr = 100

            epoch_loss += mse_val
            epoch_psnr += batch_psnr

    avg_loss = epoch_loss / len(data_loader)
    avg_psnr = epoch_psnr / len(data_loader)

    return avg_loss, avg_psnr  # [修改] 返回增加 PSNR


# ==========================================
# 4. 主流程 (已修改：打印和记录PSNR)
# ==========================================
def train_pipeline(params):
    print(f"\n{'=' * 40}")
    print(f"Running: SNR={params['snr']}, Ratio={params['ratio']}, ImageSize={params['image_size']}")
    print(f"{'=' * 40}")

    # 1. 准备数据
    transform = transforms.Compose([
        transforms.Resize((params['image_size'], params['image_size'])),
        transforms.ToTensor()
    ])

    try:
        train_dataset = MilitaryDataset(root=params['data_root'], transform=transform, train=True, seed=params['seed'])
        test_dataset = MilitaryDataset(root=params['data_root'], transform=transform, train=False, seed=params['seed'])
    except Exception as e:
        print(f"Dataset Init Failed: {e}")
        return

    train_loader = DataLoader(train_dataset, shuffle=True,
                              batch_size=params['batch_size'], num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, shuffle=False,
                             batch_size=params['batch_size'], num_workers=4, pin_memory=True)

    # 2. 初始化模型
    if len(train_dataset) == 0: return
    sample_img, _ = train_dataset[0]

    c = ratio2filtersize(sample_img, params['ratio'])
    print(f"Calculated Inner Channel (C): {c}")

    model = DeepJSCC(c=c, channel_type=params['channel'], snr=params['snr'])

    device = torch.device(params['device'] if torch.cuda.is_available() else 'cpu')
    if params['parallel'] and torch.cuda.device_count() > 1:
        model = DataParallel(model)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=params['init_lr'], weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=8, verbose=True)

    phaser = f"MILITARY_{c}_{params['snr']}_{params['ratio']:.2f}_{params['channel']}_{params['image_size']}"
    root_log_dir = os.path.join(params['out'], 'logs', phaser)
    root_ckpt_dir = os.path.join(params['out'], 'checkpoints', phaser)
    os.makedirs(root_ckpt_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=root_log_dir)

    best_loss = float('inf')

    # 3. 训练循环
    for epoch in range(params['epochs']):
        start_time = time.time()

        # [修改] 接收返回的 PSNR
        train_loss, train_psnr, optimizer = train_epoch(model, optimizer, params, train_loader)
        val_loss, val_psnr = evaluate_epoch(model, params, test_loader)

        # [新增] 写入 TensorBoard
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Val', val_loss, epoch)
        writer.add_scalar('PSNR/Train', train_psnr, epoch)  # 新增
        writer.add_scalar('PSNR/Val', val_psnr, epoch)  # 新增
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)

        scheduler.step(val_loss)

        duration = time.time() - start_time

        # [修改] 打印信息包含 PSNR
        print(f"Epoch {epoch + 1}/{params['epochs']} | Time: {duration:.1f}s | "
              f"Train Loss: {train_loss:.2f} | Train PSNR: {train_psnr:.2f} dB | "
              f"Val Loss: {val_loss:.2f} | Val PSNR: {val_psnr:.2f} dB")

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), os.path.join(root_ckpt_dir, 'best_model.pth'))

        if optimizer.param_groups[0]['lr'] < 1e-6:
            print("Learning rate too small, stop training.")
            break

    writer.close()
    print(f"Training Finished. Best Model saved to {root_ckpt_dir}/best_model.pth")


def main():
    args = config_parser()
    args.snr_list = list(map(float, args.snr_list))
    args.ratio_list = list(map(lambda x: float(Fraction(x)), args.ratio_list))
    params = vars(args)
    set_seed(params['seed'])

    for ratio in params['ratio_list']:
        for snr in params['snr_list']:
            current_params = params.copy()
            current_params['ratio'] = ratio
            current_params['snr'] = snr
            train_pipeline(current_params)


if __name__ == '__main__':
    main()