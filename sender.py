#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
发送端脚本 - DeepJSCC Sender
负责：
1. 加载图像和模型
2. 进行语义编码
3. 量化特征
4. JPEG压缩对比
5. 保存传输数据
"""

import os
import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
import time
import pickle
from model import DeepJSCC
from utils_common import KodakDataset, compress_jpeg, calculate_bpp


def quantize_features(features, num_bits=8):
    """
    量化特征到指定比特数
    
    Args:
        features: torch.Tensor, 特征张量
        num_bits: int, 量化比特数
    
    Returns:
        quantized: 量化后的特征 (整数)
        scale: 量化比例因子
        zero_point: 零点
    """
    # 计算量化参数
    min_val = features.min().item()
    max_val = features.max().item()
    
    # 计算scale和zero_point
    qmin = 0
    qmax = 2 ** num_bits - 1
    
    scale = (max_val - min_val) / (qmax - qmin)
    zero_point = qmin - min_val / scale
    
    # 量化
    quantized = torch.clamp(torch.round(features / scale + zero_point), qmin, qmax)
    
    return quantized.to(torch.uint8 if num_bits == 8 else torch.int16), scale, zero_point


def dequantize_features(quantized, scale, zero_point):
    """
    反量化特征
    
    Args:
        quantized: 量化后的特征
        scale: 量化比例因子
        zero_point: 零点
    
    Returns:
        features: 反量化后的特征
    """
    return (quantized.float() - zero_point) * scale


class DeepJSCCSender:
    """DeepJSCC发送端"""
    
    def __init__(self, model_path, c=8, snr=19, device='cuda:0', 
                 quantization_bits=8, jpeg_quality=75):
        """
        初始化发送端
        
        Args:
            model_path: 模型权重路径
            c: 内部通道数
            snr: 信噪比
            device: 设备
            quantization_bits: 量化比特数
            jpeg_quality: JPEG质量
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.quantization_bits = quantization_bits
        self.jpeg_quality = jpeg_quality
        
        # 创建和加载模型
        print(f"Creating model on {self.device}...")
        self.model = DeepJSCC(c=c, channel_type='AWGN', snr=snr)
        self.model = self.model.to(self.device)
        
        # 加载权重
        print(f"Loading model weights from {model_path}...")
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # 处理可能的DataParallel格式
        if isinstance(checkpoint, dict) and any(key.startswith('module.') for key in checkpoint.keys()):
            new_checkpoint = {}
            for key, value in checkpoint.items():
                new_key = key[7:] if key.startswith('module.') else key
                new_checkpoint[new_key] = value
            checkpoint = new_checkpoint
        
        self.model.load_state_dict(checkpoint)
        self.model.eval()
        
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Model loaded successfully! Total parameters: {total_params:,}")
    
    def encode_image(self, image_tensor):
        """
        对图像进行语义编码
        
        Args:
            image_tensor: 输入图像tensor, shape (C, H, W)
        
        Returns:
            features: 编码后的特征
        """
        with torch.no_grad():
            # 添加batch维度
            if image_tensor.dim() == 3:
                image_tensor = image_tensor.unsqueeze(0)
            
            image_tensor = image_tensor.to(self.device)
            
            # 使用encoder编码
            features = self.model.encoder(image_tensor)
            
            # 通过信道（可选，取决于是否要模拟信道噪声）
            # 在发送端通常不需要通过信道，直接量化即可
            # features = self.model.channel(features)
            
        return features
    
    def process_image(self, image_tensor, image_path, original_size):
        """
        处理单张图像
        
        Args:
            image_tensor: 图像tensor
            image_path: 图像路径
            original_size: 原始文件大小（字节）
        
        Returns:
            transmission_data: 传输数据字典
        """
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        print(f"\nProcessing {image_name}...")
        
        # 1. 语义编码
        start_time = time.time()
        features = self.encode_image(image_tensor)
        encoding_time = (time.time() - start_time) * 1000
        
        print(f"  Encoding time: {encoding_time:.2f} ms")
        print(f"  Feature shape: {features.shape}")
        
        # 2. 量化特征
        start_time = time.time()
        quantized_features, scale, zero_point = quantize_features(
            features, self.quantization_bits
        )
        quantization_time = (time.time() - start_time) * 1000
        
        print(f"  Quantization time: {quantization_time:.2f} ms")
        
        # 计算量化后的特征大小
        semantic_size = quantized_features.numel() * (self.quantization_bits // 8)
        semantic_size += 8 * 2  # scale和zero_point的大小（float32 * 2）
        
        # 3. JPEG压缩
        start_time = time.time()
        jpeg_size, _ = compress_jpeg(image_tensor, quality=self.jpeg_quality)
        jpeg_time = (time.time() - start_time) * 1000
        
        print(f"  JPEG compression time: {jpeg_time:.2f} ms")
        
        # 4. 计算压缩率和bpp
        _, H, W = image_tensor.shape
        
        semantic_kb = semantic_size / 1024
        jpeg_kb = jpeg_size / 1024
        original_kb = original_size / 1024
        
        semantic_cr = original_size / semantic_size
        jpeg_cr = original_size / jpeg_size
        
        semantic_bpp = calculate_bpp(semantic_size, H, W)
        jpeg_bpp = calculate_bpp(jpeg_size, H, W)
        
        print(f"  Original size: {original_kb:.2f} KB")
        print(f"  DeepJSCC size: {semantic_kb:.2f} KB (CR: {semantic_cr:.2f}x, BPP: {semantic_bpp:.3f})")
        print(f"  JPEG size: {jpeg_kb:.2f} KB (CR: {jpeg_cr:.2f}x, BPP: {jpeg_bpp:.3f})")
        
        # 5. 准备传输数据
        transmission_data = {
            'image_name': image_name,
            'image_tensor': image_tensor.cpu(),  # 保存原始图像用于接收端计算指标
            
            # 语义特征相关
            'quantized_features': quantized_features.cpu(),
            'scale': scale,
            'zero_point': zero_point,
            'feature_shape': features.shape,
            'semantic_size': semantic_size,
            'semantic_kb': semantic_kb,
            'semantic_cr': semantic_cr,
            'semantic_bpp': semantic_bpp,
            
            # JPEG相关
            'jpeg_quality': self.jpeg_quality,
            'jpeg_size': jpeg_size,
            'jpeg_kb': jpeg_kb,
            'jpeg_cr': jpeg_cr,
            'jpeg_bpp': jpeg_bpp,
            
            # 原始信息
            'original_size': original_size,
            'original_kb': original_kb,
            'image_shape': image_tensor.shape,
            
            # 时间信息
            'encoding_time_ms': encoding_time,
            'quantization_time_ms': quantization_time,
            'jpeg_time_ms': jpeg_time
        }
        
        return transmission_data
    
    def process_dataset(self, data_loader, output_dir):
        """
        处理整个数据集
        
        Args:
            data_loader: 数据加载器
            output_dir: 输出目录
        
        Returns:
            all_transmission_data: 所有图像的传输数据列表
        """
        os.makedirs(output_dir, exist_ok=True)
        all_transmission_data = []
        
        print(f"\n{'='*80}")
        print(f"Starting encoding process...")
        print(f"{'='*80}")
        
        total_start = time.time()
        
        for idx, (image_tensor, image_path, original_size) in enumerate(data_loader):
            # 移除batch维度
            image_tensor = image_tensor.squeeze(0)
            image_path = image_path[0]
            original_size = original_size.item()
            
            # 处理图像
            transmission_data = self.process_image(image_tensor, image_path, original_size)
            all_transmission_data.append(transmission_data)
            
            # 保存单个图像的传输数据
            save_path = os.path.join(output_dir, f"{transmission_data['image_name']}_transmission.pkl")
            with open(save_path, 'wb') as f:
                pickle.dump(transmission_data, f)
            
            print(f"  Saved to: {save_path}")
        
        total_time = time.time() - total_start
        
        # 保存汇总信息
        summary = {
            'all_data': all_transmission_data,
            'total_images': len(all_transmission_data),
            'total_processing_time': total_time,
            'quantization_bits': self.quantization_bits,
            'jpeg_quality': self.jpeg_quality,
            'device': str(self.device)
        }
        
        summary_path = os.path.join(output_dir, 'transmission_summary.pkl')
        with open(summary_path, 'wb') as f:
            pickle.dump(summary, f)
        
        print(f"\n{'='*80}")
        print(f"Encoding completed!")
        print(f"{'='*80}")
        print(f"Total images: {len(all_transmission_data)}")
        print(f"Total time: {total_time:.2f} seconds")
        print(f"Average time per image: {total_time/len(all_transmission_data):.2f} seconds")
        print(f"All data saved to: {output_dir}")
        
        # 打印统计信息
        self.print_statistics(all_transmission_data)
        
        return all_transmission_data
    
    def print_statistics(self, all_data):
        """打印统计信息"""
        print(f"\n{'='*80}")
        print("Compression Statistics")
        print(f"{'='*80}")
        
        # DeepJSCC统计
        semantic_sizes = [d['semantic_kb'] for d in all_data]
        semantic_crs = [d['semantic_cr'] for d in all_data]
        semantic_bpps = [d['semantic_bpp'] for d in all_data]
        
        print(f"\nDeepJSCC:")
        print(f"  Avg Size: {np.mean(semantic_sizes):.2f} ± {np.std(semantic_sizes):.2f} KB")
        print(f"  Avg CR: {np.mean(semantic_crs):.2f}x ± {np.std(semantic_crs):.2f}x")
        print(f"  Avg BPP: {np.mean(semantic_bpps):.3f} ± {np.std(semantic_bpps):.3f}")
        
        # JPEG统计
        jpeg_sizes = [d['jpeg_kb'] for d in all_data]
        jpeg_crs = [d['jpeg_cr'] for d in all_data]
        jpeg_bpps = [d['jpeg_bpp'] for d in all_data]
        
        print(f"\nJPEG (Quality={self.jpeg_quality}):")
        print(f"  Avg Size: {np.mean(jpeg_sizes):.2f} ± {np.std(jpeg_sizes):.2f} KB")
        print(f"  Avg CR: {np.mean(jpeg_crs):.2f}x ± {np.std(jpeg_crs):.2f}x")
        print(f"  Avg BPP: {np.mean(jpeg_bpps):.3f} ± {np.std(jpeg_bpps):.3f}")
        
        # 时间统计
        encoding_times = [d['encoding_time_ms'] for d in all_data]
        print(f"\nTiming:")
        print(f"  Avg Encoding Time: {np.mean(encoding_times):.2f} ± {np.std(encoding_times):.2f} ms")


def main():
    # ========== 配置参数 ==========
    MODEL_PATH = r"D:\pythonproject\Deep-JSCC-PyTorch-main\out\checkpoints\CIFAR10_8_19.0_0.17_AWGN_22h13m53s_on_Jun_07_2024\epoch_998.pkl"
    KODAK_PATH = r"D:\pythonproject\Deep-JSCC-PyTorch-main\data\kodak"
    OUTPUT_DIR = "./transmission_data"
    
    # 模型参数
    C = 8  # 内部通道数
    SNR = 19  # 信噪比
    DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    # 压缩参数
    QUANTIZATION_BITS = 8  # 量化比特数 (8或16)
    JPEG_QUALITY = 75  # JPEG质量 (1-100)
    
    # ========== 打印配置信息 ==========
    print("=" * 80)
    print("DeepJSCC Sender - Encoding and Transmission")
    print("=" * 80)
    print(f"Model path: {MODEL_PATH}")
    print(f"Kodak path: {KODAK_PATH}")
    print(f"Output dir: {OUTPUT_DIR}")
    print(f"Device: {DEVICE}")
    print(f"Inner channels (c): {C}")
    print(f"SNR: {SNR} dB")
    print(f"Quantization bits: {QUANTIZATION_BITS}")
    print(f"JPEG quality: {JPEG_QUALITY}")
    
    # 检查文件路径
    if not os.path.exists(MODEL_PATH):
        print(f"\nError: Model file not found: {MODEL_PATH}")
        return
    
    if not os.path.exists(KODAK_PATH):
        print(f"\nError: Kodak dataset not found: {KODAK_PATH}")
        return
    
    # 准备数据集
    print(f"\nLoading Kodak dataset...")
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])
    
    dataset = KodakDataset(root=KODAK_PATH, transform=transform)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    
    print(f"Dataset loaded: {len(dataset)} images")
    
    # 创建发送端
    sender = DeepJSCCSender(
        model_path=MODEL_PATH,
        c=C,
        snr=SNR,
        device=DEVICE,
        quantization_bits=QUANTIZATION_BITS,
        jpeg_quality=JPEG_QUALITY
    )
    
    # 处理数据集
    all_transmission_data = sender.process_dataset(data_loader, OUTPUT_DIR)
    
    print(f"\n{'='*80}")
    print("Sender process completed!")
    print(f"Transmission data saved to: {OUTPUT_DIR}")
    print("You can now run the receiver script to decode and evaluate.")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
