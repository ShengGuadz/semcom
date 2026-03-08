#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
共享工具函数
包含数据集加载、指标计算、JPEG压缩等通用功能
"""

import os
import io
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import matplotlib.pyplot as plt


class KodakDataset(Dataset):
    """简单的Kodak数据集加载器"""

    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform

        # 支持的图像格式
        valid_extensions = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.tif', '.tiff')

        # 获取所有图像文件
        self.image_paths = []
        if os.path.exists(root):
            for file in os.listdir(root):
                if file.lower().endswith(valid_extensions):
                    self.image_paths.append(os.path.join(root, file))

        self.image_paths.sort()  # 确保顺序一致

        if not self.image_paths:
            raise ValueError(f"No valid images found in {root}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]

        # 加载图像
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            raise RuntimeError(f"Error loading image {image_path}: {e}")

        # 获取原始图像尺寸（用于计算原始大小）
        original_size = os.path.getsize(image_path)

        # 应用变换
        if self.transform:
            image_tensor = self.transform(image)
        else:
            image_tensor = image

        return image_tensor, image_path, original_size


def calculate_psnr(img1, img2, max_val=255.0):
    """计算PSNR (Peak Signal-to-Noise Ratio)"""
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    psnr = 20 * torch.log10(max_val / torch.sqrt(mse))
    return psnr


def calculate_ssim(img1, img2, max_val=255.0):
    """简化版SSIM计算"""
    # 计算均值
    mu1 = torch.mean(img1)
    mu2 = torch.mean(img2)

    # 计算方差和协方差
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = torch.mean(img1 * img1) - mu1_sq
    sigma2_sq = torch.mean(img2 * img2) - mu2_sq
    sigma12 = torch.mean(img1 * img2) - mu1_mu2

    # SSIM常数
    c1 = (0.01 * max_val) ** 2
    c2 = (0.03 * max_val) ** 2

    # 计算SSIM
    ssim = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / \
           ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))

    return ssim


def calculate_metrics(original, reconstructed):
    """计算PSNR和SSIM指标"""
    # 转换到0-255范围
    original_255 = (original * 255).clamp(0, 255)
    reconstructed_255 = (reconstructed * 255).clamp(0, 255)

    # 计算PSNR
    mse = torch.mean((original_255 - reconstructed_255) ** 2)
    if mse == 0:
        psnr = float('inf')
    else:
        psnr = 20 * torch.log10(255.0 / torch.sqrt(mse))

    # 计算SSIM
    ssim = calculate_ssim(original_255, reconstructed_255, max_val=255.0)

    return psnr.item() if torch.is_tensor(psnr) else psnr, \
        ssim.item() if torch.is_tensor(ssim) else ssim


def compress_jpeg(image_tensor, quality=75):
    """
    将图像tensor压缩为JPEG格式
    
    Args:
        image_tensor: torch.Tensor, shape (C, H, W), range [0, 1]
        quality: int, JPEG质量 (1-100)
    
    Returns:
        jpeg_size: JPEG压缩后的字节数
        decompressed_tensor: 解压后的图像tensor
    """
    # 转换为PIL图像
    image_np = (image_tensor.cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    pil_image = Image.fromarray(image_np)
    
    # 压缩为JPEG
    buffer = io.BytesIO()
    pil_image.save(buffer, format='JPEG', quality=quality)
    jpeg_size = buffer.tell()
    
    # 解压
    buffer.seek(0)
    decompressed_pil = Image.open(buffer).convert('RGB')
    decompressed_np = np.array(decompressed_pil).astype(np.float32) / 255.0
    decompressed_tensor = torch.from_numpy(decompressed_np).permute(2, 0, 1)
    
    return jpeg_size, decompressed_tensor


def save_comparison_images(original, reconstructed, jpeg_recon, save_path, 
                          metrics_deepjscc, metrics_jpeg, sizes):
    """
    保存原图、DeepJSCC重建和JPEG重建的对比图
    
    Args:
        original: 原始图像tensor
        reconstructed: DeepJSCC重建图像
        jpeg_recon: JPEG重建图像
        save_path: 保存路径
        metrics_deepjscc: DeepJSCC的指标 (psnr, ssim)
        metrics_jpeg: JPEG的指标 (psnr, ssim)
        sizes: 大小信息字典
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 转换为numpy格式 (H, W, C)
    orig_np = original.cpu().permute(1, 2, 0).numpy().clip(0, 1)
    recon_np = reconstructed.cpu().permute(1, 2, 0).numpy().clip(0, 1)
    jpeg_np = jpeg_recon.cpu().permute(1, 2, 0).numpy().clip(0, 1)
    
    # 显示原图
    axes[0].imshow(orig_np)
    axes[0].set_title(f'Original\nSize: {sizes["original_kb"]:.2f} KB', fontsize=12)
    axes[0].axis('off')
    
    # 显示DeepJSCC重建
    psnr_dj, ssim_dj = metrics_deepjscc
    axes[1].imshow(recon_np)
    axes[1].set_title(
        f'DeepJSCC\n'
        f'Size: {sizes["semantic_kb"]:.2f} KB (CR: {sizes["semantic_cr"]:.2f}x)\n'
        f'PSNR: {psnr_dj:.2f} dB, SSIM: {ssim_dj:.4f}',
        fontsize=12
    )
    axes[1].axis('off')
    
    # 显示JPEG重建
    psnr_jpg, ssim_jpg = metrics_jpeg
    axes[2].imshow(jpeg_np)
    axes[2].set_title(
        f'JPEG (Q={sizes.get("jpeg_quality", 75)})\n'
        f'Size: {sizes["jpeg_kb"]:.2f} KB (CR: {sizes["jpeg_cr"]:.2f}x)\n'
        f'PSNR: {psnr_jpg:.2f} dB, SSIM: {ssim_jpg:.4f}',
        fontsize=12
    )
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def calculate_bpp(feature_size_bytes, image_height, image_width):
    """
    计算 bits per pixel (bpp)
    
    Args:
        feature_size_bytes: 特征的字节数
        image_height: 图像高度
        image_width: 图像宽度
    
    Returns:
        bpp: bits per pixel
    """
    total_bits = feature_size_bytes * 8
    total_pixels = image_height * image_width
    return total_bits / total_pixels


def print_comparison_table(results_list):
    """打印对比表格"""
    print("\n" + "=" * 120)
    print("Compression Comparison: DeepJSCC vs JPEG")
    print("=" * 120)
    print(f"{'Image':<10} | {'Method':<10} | {'Size (KB)':<12} | {'CR':<8} | {'BPP':<8} | "
          f"{'PSNR (dB)':<12} | {'SSIM':<8}")
    print("-" * 120)
    
    for result in results_list:
        image_name = result['image_name']
        
        # DeepJSCC
        print(f"{image_name:<10} | {'DeepJSCC':<10} | "
              f"{result['semantic_kb']:>11.2f} | {result['semantic_cr']:>7.2f} | "
              f"{result['semantic_bpp']:>7.3f} | {result['psnr_deepjscc']:>11.2f} | "
              f"{result['ssim_deepjscc']:>7.4f}")
        
        # JPEG
        print(f"{'':10} | {'JPEG':<10} | "
              f"{result['jpeg_kb']:>11.2f} | {result['jpeg_cr']:>7.2f} | "
              f"{result['jpeg_bpp']:>7.3f} | {result['psnr_jpeg']:>11.2f} | "
              f"{result['ssim_jpeg']:>7.4f}")
        print("-" * 120)
