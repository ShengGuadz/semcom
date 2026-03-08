#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
接收端脚本 - DeepJSCC Receiver
负责：
1. 加载传输数据
2. 反量化特征
3. 解码重建图像
4. JPEG解压和评估
5. 计算和对比质量指标
6. 生成对比可视化
"""

import os
import torch
import numpy as np
import time
import pickle
from model import DeepJSCC
from utils_common import (
    calculate_metrics, 
    compress_jpeg,
    save_comparison_images,
    print_comparison_table
)


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


class DeepJSCCReceiver:
    """DeepJSCC接收端"""
    
    def __init__(self, model_path, c=8, snr=19, device='cuda:0'):
        """
        初始化接收端
        
        Args:
            model_path: 模型权重路径
            c: 内部通道数
            snr: 信噪比
            device: 设备
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
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
    
    def decode_features(self, features):
        """
        解码特征重建图像
        
        Args:
            features: 编码特征
        
        Returns:
            reconstructed: 重建的图像
        """
        with torch.no_grad():
            features = features.to(self.device)
            
            # 使用decoder解码
            reconstructed = self.model.decoder(features)
            
        return reconstructed
    
    def process_transmission_data(self, transmission_data, output_dir):
        """
        处理单个传输数据
        
        Args:
            transmission_data: 传输数据字典
            output_dir: 输出目录
        
        Returns:
            result: 评估结果字典
        """
        image_name = transmission_data['image_name']
        print(f"\nProcessing {image_name}...")
        
        # 1. 获取原始图像
        original_tensor = transmission_data['image_tensor']
        
        # 2. 反量化语义特征
        start_time = time.time()
        quantized_features = transmission_data['quantized_features']
        scale = transmission_data['scale']
        zero_point = transmission_data['zero_point']
        feature_shape = transmission_data['feature_shape']
        
        # 反量化并reshape到正确的形状
        dequantized_features = dequantize_features(quantized_features, scale, zero_point)
        dequantized_features = dequantized_features.reshape(feature_shape)
        
        dequantization_time = (time.time() - start_time) * 1000
        print(f"  Dequantization time: {dequantization_time:.2f} ms")
        
        # 3. DeepJSCC解码重建
        start_time = time.time()
        reconstructed = self.decode_features(dequantized_features)
        decoding_time = (time.time() - start_time) * 1000
        
        # 移除batch维度
        reconstructed = reconstructed.squeeze(0).cpu()
        print(f"  Decoding time: {decoding_time:.2f} ms")
        
        # 4. JPEG解压重建
        start_time = time.time()
        jpeg_quality = transmission_data['jpeg_quality']
        _, jpeg_reconstructed = compress_jpeg(original_tensor, quality=jpeg_quality)
        jpeg_decompress_time = (time.time() - start_time) * 1000
        print(f"  JPEG decompression time: {jpeg_decompress_time:.2f} ms")
        
        # 5. 计算DeepJSCC的质量指标
        start_time = time.time()
        psnr_deepjscc, ssim_deepjscc = calculate_metrics(original_tensor, reconstructed)
        metric_time_dj = (time.time() - start_time) * 1000
        
        # 6. 计算JPEG的质量指标
        start_time = time.time()
        psnr_jpeg, ssim_jpeg = calculate_metrics(original_tensor, jpeg_reconstructed)
        metric_time_jpg = (time.time() - start_time) * 1000
        
        print(f"  Metrics calculation time: {metric_time_dj + metric_time_jpg:.2f} ms")
        
        print(f"\n  Quality Comparison:")
        print(f"    DeepJSCC - PSNR: {psnr_deepjscc:.2f} dB, SSIM: {ssim_deepjscc:.4f}")
        print(f"    JPEG     - PSNR: {psnr_jpeg:.2f} dB, SSIM: {ssim_jpeg:.4f}")
        
        # 7. 保存对比图像
        sizes = {
            'original_kb': transmission_data['original_kb'],
            'semantic_kb': transmission_data['semantic_kb'],
            'semantic_cr': transmission_data['semantic_cr'],
            'jpeg_kb': transmission_data['jpeg_kb'],
            'jpeg_cr': transmission_data['jpeg_cr'],
            'jpeg_quality': jpeg_quality
        }
        
        save_path = os.path.join(output_dir, 'comparisons', f'{image_name}_comparison.png')
        save_comparison_images(
            original_tensor,
            reconstructed,
            jpeg_reconstructed,
            save_path,
            (psnr_deepjscc, ssim_deepjscc),
            (psnr_jpeg, ssim_jpeg),
            sizes
        )
        print(f"  Comparison image saved to: {save_path}")
        
        # 8. 准备结果
        result = {
            'image_name': image_name,
            
            # 大小和压缩率信息
            'original_kb': transmission_data['original_kb'],
            'semantic_kb': transmission_data['semantic_kb'],
            'semantic_cr': transmission_data['semantic_cr'],
            'semantic_bpp': transmission_data['semantic_bpp'],
            'jpeg_kb': transmission_data['jpeg_kb'],
            'jpeg_cr': transmission_data['jpeg_cr'],
            'jpeg_bpp': transmission_data['jpeg_bpp'],
            
            # DeepJSCC质量指标
            'psnr_deepjscc': psnr_deepjscc,
            'ssim_deepjscc': ssim_deepjscc,
            
            # JPEG质量指标
            'psnr_jpeg': psnr_jpeg,
            'ssim_jpeg': ssim_jpeg,
            
            # 时间信息
            'dequantization_time_ms': dequantization_time,
            'decoding_time_ms': decoding_time,
            'jpeg_decompress_time_ms': jpeg_decompress_time,
            'total_time_ms': dequantization_time + decoding_time,
            
            # 编码端时间（从传输数据获取）
            'encoding_time_ms': transmission_data['encoding_time_ms'],
            'quantization_time_ms': transmission_data['quantization_time_ms']
        }
        
        return result
    
    def process_all_data(self, transmission_dir, output_dir):
        """
        处理所有传输数据
        
        Args:
            transmission_dir: 传输数据目录
            output_dir: 输出目录
        
        Returns:
            all_results: 所有结果列表
        """
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'comparisons'), exist_ok=True)
        
        # 加载汇总信息
        summary_path = os.path.join(transmission_dir, 'transmission_summary.pkl')
        if not os.path.exists(summary_path):
            print(f"Error: Transmission summary not found: {summary_path}")
            return []
        
        with open(summary_path, 'rb') as f:
            summary = pickle.load(f)
        
        all_transmission_data = summary['all_data']
        
        print(f"\n{'='*80}")
        print(f"Starting decoding process...")
        print(f"{'='*80}")
        print(f"Total images to process: {len(all_transmission_data)}")
        
        all_results = []
        total_start = time.time()
        
        for transmission_data in all_transmission_data:
            result = self.process_transmission_data(transmission_data, output_dir)
            all_results.append(result)
        
        total_time = time.time() - total_start
        
        print(f"\n{'='*80}")
        print(f"Decoding completed!")
        print(f"{'='*80}")
        print(f"Total images: {len(all_results)}")
        print(f"Total time: {total_time:.2f} seconds")
        print(f"Average time per image: {total_time/len(all_results):.2f} seconds")
        
        # 保存结果
        self.save_results(all_results, output_dir, summary)
        
        # 打印对比表格
        print_comparison_table(all_results)
        
        # 打印统计分析
        self.print_analysis(all_results)
        
        return all_results
    
    def save_results(self, all_results, output_dir, summary):
        """保存评估结果"""
        # 保存详细结果
        results_path = os.path.join(output_dir, 'evaluation_results.pkl')
        with open(results_path, 'wb') as f:
            pickle.dump({
                'results': all_results,
                'summary': summary
            }, f)
        
        # 保存文本报告
        report_path = os.path.join(output_dir, 'evaluation_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("DeepJSCC vs JPEG Compression Evaluation Report\n")
            f.write("=" * 100 + "\n\n")
            
            f.write("Configuration:\n")
            f.write(f"  Quantization bits: {summary.get('quantization_bits', 'N/A')}\n")
            f.write(f"  JPEG quality: {summary.get('jpeg_quality', 'N/A')}\n")
            f.write(f"  Total images: {summary.get('total_images', 'N/A')}\n")
            f.write(f"  Device: {summary.get('device', 'N/A')}\n\n")
            
            f.write("Detailed Results:\n")
            f.write("-" * 100 + "\n")
            f.write(f"{'Image':<12} | {'Method':<10} | {'Size(KB)':<10} | {'CR':<8} | "
                   f"{'BPP':<8} | {'PSNR(dB)':<10} | {'SSIM':<8}\n")
            f.write("-" * 100 + "\n")
            
            for result in all_results:
                # DeepJSCC
                f.write(f"{result['image_name']:<12} | {'DeepJSCC':<10} | "
                       f"{result['semantic_kb']:>9.2f} | {result['semantic_cr']:>7.2f} | "
                       f"{result['semantic_bpp']:>7.3f} | {result['psnr_deepjscc']:>9.2f} | "
                       f"{result['ssim_deepjscc']:>7.4f}\n")
                # JPEG
                f.write(f"{'':12} | {'JPEG':<10} | "
                       f"{result['jpeg_kb']:>9.2f} | {result['jpeg_cr']:>7.2f} | "
                       f"{result['jpeg_bpp']:>7.3f} | {result['psnr_jpeg']:>9.2f} | "
                       f"{result['ssim_jpeg']:>7.4f}\n")
                f.write("-" * 100 + "\n")
            
            # 统计信息
            f.write("\n\nStatistical Summary:\n")
            f.write("=" * 100 + "\n")
            
            self._write_statistics(f, all_results)
        
        print(f"\nResults saved:")
        print(f"  Results: {results_path}")
        print(f"  Report: {report_path}")
    
    def _write_statistics(self, f, results):
        """写入统计信息到文件"""
        # DeepJSCC统计
        psnr_dj = [r['psnr_deepjscc'] for r in results]
        ssim_dj = [r['ssim_deepjscc'] for r in results]
        size_dj = [r['semantic_kb'] for r in results]
        cr_dj = [r['semantic_cr'] for r in results]
        bpp_dj = [r['semantic_bpp'] for r in results]
        
        f.write("\nDeepJSCC:\n")
        f.write(f"  PSNR: {np.mean(psnr_dj):.2f} ± {np.std(psnr_dj):.2f} dB "
               f"(min: {np.min(psnr_dj):.2f}, max: {np.max(psnr_dj):.2f})\n")
        f.write(f"  SSIM: {np.mean(ssim_dj):.4f} ± {np.std(ssim_dj):.4f} "
               f"(min: {np.min(ssim_dj):.4f}, max: {np.max(ssim_dj):.4f})\n")
        f.write(f"  Size: {np.mean(size_dj):.2f} ± {np.std(size_dj):.2f} KB\n")
        f.write(f"  CR: {np.mean(cr_dj):.2f}x ± {np.std(cr_dj):.2f}x\n")
        f.write(f"  BPP: {np.mean(bpp_dj):.3f} ± {np.std(bpp_dj):.3f}\n")
        
        # JPEG统计
        psnr_jpg = [r['psnr_jpeg'] for r in results]
        ssim_jpg = [r['ssim_jpeg'] for r in results]
        size_jpg = [r['jpeg_kb'] for r in results]
        cr_jpg = [r['jpeg_cr'] for r in results]
        bpp_jpg = [r['jpeg_bpp'] for r in results]
        
        f.write("\nJPEG:\n")
        f.write(f"  PSNR: {np.mean(psnr_jpg):.2f} ± {np.std(psnr_jpg):.2f} dB "
               f"(min: {np.min(psnr_jpg):.2f}, max: {np.max(psnr_jpg):.2f})\n")
        f.write(f"  SSIM: {np.mean(ssim_jpg):.4f} ± {np.std(ssim_jpg):.4f} "
               f"(min: {np.min(ssim_jpg):.4f}, max: {np.max(ssim_jpg):.4f})\n")
        f.write(f"  Size: {np.mean(size_jpg):.2f} ± {np.std(size_jpg):.2f} KB\n")
        f.write(f"  CR: {np.mean(cr_jpg):.2f}x ± {np.std(cr_jpg):.2f}x\n")
        f.write(f"  BPP: {np.mean(bpp_jpg):.3f} ± {np.std(bpp_jpg):.3f}\n")
        
        # 对比
        f.write("\nComparison (DeepJSCC vs JPEG):\n")
        psnr_diff = np.array(psnr_dj) - np.array(psnr_jpg)
        ssim_diff = np.array(ssim_dj) - np.array(ssim_jpg)
        size_ratio = np.array(size_dj) / np.array(size_jpg)
        
        f.write(f"  PSNR difference: {np.mean(psnr_diff):.2f} ± {np.std(psnr_diff):.2f} dB\n")
        f.write(f"  SSIM difference: {np.mean(ssim_diff):.4f} ± {np.std(ssim_diff):.4f}\n")
        f.write(f"  Size ratio: {np.mean(size_ratio):.2f}x ± {np.std(size_ratio):.2f}x\n")
        
        # 统计哪个方法更好
        dj_better_psnr = sum(1 for d in psnr_diff if d > 0)
        dj_better_ssim = sum(1 for d in ssim_diff if d > 0)
        dj_smaller_size = sum(1 for r in size_ratio if r < 1)
        
        f.write(f"\nPerformance Summary:\n")
        f.write(f"  DeepJSCC achieves higher PSNR in {dj_better_psnr}/{len(results)} images\n")
        f.write(f"  DeepJSCC achieves higher SSIM in {dj_better_ssim}/{len(results)} images\n")
        f.write(f"  DeepJSCC achieves smaller size in {dj_smaller_size}/{len(results)} images\n")
    
    def print_analysis(self, results):
        """打印分析结果"""
        print(f"\n{'='*80}")
        print("Performance Analysis")
        print(f"{'='*80}")
        
        # DeepJSCC统计
        psnr_dj = [r['psnr_deepjscc'] for r in results]
        ssim_dj = [r['ssim_deepjscc'] for r in results]
        size_dj = [r['semantic_kb'] for r in results]
        cr_dj = [r['semantic_cr'] for r in results]
        
        print(f"\nDeepJSCC Performance:")
        print(f"  PSNR: {np.mean(psnr_dj):.2f} ± {np.std(psnr_dj):.2f} dB")
        print(f"  SSIM: {np.mean(ssim_dj):.4f} ± {np.std(ssim_dj):.4f}")
        print(f"  Size: {np.mean(size_dj):.2f} ± {np.std(size_dj):.2f} KB")
        print(f"  CR: {np.mean(cr_dj):.2f}x ± {np.std(cr_dj):.2f}x")
        
        # JPEG统计
        psnr_jpg = [r['psnr_jpeg'] for r in results]
        ssim_jpg = [r['ssim_jpeg'] for r in results]
        size_jpg = [r['jpeg_kb'] for r in results]
        cr_jpg = [r['jpeg_cr'] for r in results]
        
        print(f"\nJPEG Performance:")
        print(f"  PSNR: {np.mean(psnr_jpg):.2f} ± {np.std(psnr_jpg):.2f} dB")
        print(f"  SSIM: {np.mean(ssim_jpg):.4f} ± {np.std(ssim_jpg):.4f}")
        print(f"  Size: {np.mean(size_jpg):.2f} ± {np.std(size_jpg):.2f} KB")
        print(f"  CR: {np.mean(cr_jpg):.2f}x ± {np.std(cr_jpg):.2f}x")
        
        # 对比分析
        print(f"\nComparative Analysis:")
        psnr_diff = np.mean(psnr_dj) - np.mean(psnr_jpg)
        ssim_diff = np.mean(ssim_dj) - np.mean(ssim_jpg)
        size_ratio = np.mean(size_dj) / np.mean(size_jpg)
        
        print(f"  PSNR difference: {psnr_diff:+.2f} dB "
              f"({'DeepJSCC better' if psnr_diff > 0 else 'JPEG better'})")
        print(f"  SSIM difference: {ssim_diff:+.4f} "
              f"({'DeepJSCC better' if ssim_diff > 0 else 'JPEG better'})")
        print(f"  Size ratio (DJ/JPEG): {size_ratio:.2f}x "
              f"({'DeepJSCC smaller' if size_ratio < 1 else 'JPEG smaller'})")


def main():
    # ========== 配置参数 ==========
    MODEL_PATH = r"D:\pythonproject\Deep-JSCC-PyTorch-main\out\checkpoints\CIFAR10_8_19.0_0.17_AWGN_22h13m53s_on_Jun_07_2024\epoch_998.pkl"
    TRANSMISSION_DIR = "./transmission_data"  # 发送端生成的数据目录
    OUTPUT_DIR = "./receiver_results"
    
    # 模型参数
    C = 8  # 内部通道数
    SNR = 19  # 信噪比
    DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    # ========== 打印配置信息 ==========
    print("=" * 80)
    print("DeepJSCC Receiver - Decoding and Evaluation")
    print("=" * 80)
    print(f"Model path: {MODEL_PATH}")
    print(f"Transmission data dir: {TRANSMISSION_DIR}")
    print(f"Output dir: {OUTPUT_DIR}")
    print(f"Device: {DEVICE}")
    print(f"Inner channels (c): {C}")
    print(f"SNR: {SNR} dB")
    
    # 检查文件路径
    if not os.path.exists(MODEL_PATH):
        print(f"\nError: Model file not found: {MODEL_PATH}")
        return
    
    if not os.path.exists(TRANSMISSION_DIR):
        print(f"\nError: Transmission data not found: {TRANSMISSION_DIR}")
        print("Please run sender.py first to generate transmission data.")
        return
    
    # 创建接收端
    receiver = DeepJSCCReceiver(
        model_path=MODEL_PATH,
        c=C,
        snr=SNR,
        device=DEVICE
    )
    
    # 处理所有传输数据
    all_results = receiver.process_all_data(TRANSMISSION_DIR, OUTPUT_DIR)
    
    print(f"\n{'='*80}")
    print("Receiver process completed!")
    print(f"All results saved to: {OUTPUT_DIR}")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
