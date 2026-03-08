import os
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as calculate_psnr
from skimage.metrics import structural_similarity as calculate_ssim

# 导入你提供的逻辑类
from sender_hailo import ConfigSender, Sender
from receiver_hailo import ConfigReceiver, Receiver


def add_awgn_to_feature(feature_tensor, snr_db):
    """
    对语义特征添加加性高斯白噪声 (AWGN)
    假设特征已经过功率归一化 (Power Normalization)
    """
    snr = 10 ** (snr_db / 10.0)
    # 计算噪声标准差 (假设信号功率为1)
    std = np.sqrt(1.0 / snr)
    noise = torch.randn_like(feature_tensor) * std
    return feature_tensor + noise


def run_evaluation():
    # 1. 初始化
    snr_list = [-10,-5,0, 5, 10, 15, 20]
    results = {
        'snr': snr_list,
        'sem_psnr': [], 'sem_ssim': [],
        'trad_psnr': [], 'trad_ssim': []
    }

    sender = Sender(ConfigSender())
    receiver = Receiver(ConfigReceiver())

    print(f"Starting evaluation on {len(sender.test_loader)} images...")

    for snr in snr_list:
        print(f"\nEvaluating SNR = {snr} dB")
        receiver.config.snr = snr  # 更新接收端的信噪比配置用于 JPEG 位错误模拟

        tmp_sem_psnr, tmp_sem_ssim = [], []
        tmp_trad_psnr, tmp_trad_ssim = [], []

        for i, image_tensor in enumerate(tqdm(sender.test_loader)):
            # --- 编码端 (Sender) ---
            # feature_q: 量化后的特征, img_processed: 原始RGB图像(0-1)
            feature_q, min_v, max_v, rotated, img_processed = sender.encode(image_tensor)

            # 准备原始图像 (用于对比)
            orig_img = (img_processed * 255).clip(0, 255).astype('uint8')
            # 这里的 orig_img 是 RGB 格式

            # --- 语义通信模拟 (Semantic) ---
            # 1. 反量化回 Tensor
            feat_tensor = receiver._dequantize(feature_q, min_v, max_v)
            # 2. 模拟信道噪声 (JSCC 核心)
            feat_noisy = add_awgn_to_feature(feat_tensor, snr)

            # 3. 语义解码
            with torch.no_grad():
                # 模拟 receiver_hailo.py 中的解码逻辑，但使用带噪声的特征
                if feat_noisy.dim() == 3: feat_noisy = feat_noisy.unsqueeze(0)
                recon = receiver.model.decoder(feat_noisy)
                recon_np = recon.squeeze(0).cpu().numpy().transpose(1, 2, 0)
                recon_np = (recon_np * 255).clip(0, 255).astype(np.uint8)
                # 保持 RGB 进行计算
                sem_img = recon_np

                # 处理旋转恢复 (如果需要)
                if rotated == 1:
                    sem_img = np.transpose(sem_img, (1, 0, 2))

            # --- 传统通信模拟 (JPEG) ---
            # 1. JPEG 编码
            img_bgr = cv2.cvtColor(orig_img, cv2.COLOR_RGB2BGR)
            jpeg_data = cv2.imencode('.jpg', img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 100])[1].tobytes()

            # 2. 模拟位错误
            corr_jpeg = receiver._simulate_bit_errors(jpeg_data, snr)

            # 3. JPEG 解码
            trad_img_bgr = cv2.imdecode(np.frombuffer(corr_jpeg, np.uint8), cv2.IMREAD_COLOR)
            if trad_img_bgr is None:  # 如果解码失败（在高噪声下常见）
                trad_img = np.zeros_like(orig_img)
            else:
                trad_img = cv2.cvtColor(trad_img_bgr, cv2.COLOR_BGR2RGB)
                if rotated == 1 and trad_img.shape != orig_img.shape:
                    trad_img = np.transpose(trad_img, (1, 0, 2))

            # --- 计算指标 ---
            # 确保尺寸一致
            if sem_img.shape != orig_img.shape:
                sem_img = cv2.resize(sem_img, (orig_img.shape[1], orig_img.shape[0]))

            tmp_sem_psnr.append(calculate_psnr(orig_img, sem_img))
            tmp_sem_ssim.append(calculate_ssim(orig_img, sem_img, channel_axis=2))

            tmp_trad_psnr.append(calculate_psnr(orig_img, trad_img))
            tmp_trad_ssim.append(calculate_ssim(orig_img, trad_img, channel_axis=2))

        # 记录平均值
        results['sem_psnr'].append(np.mean(tmp_sem_psnr))
        results['sem_ssim'].append(np.mean(tmp_sem_ssim))
        results['trad_psnr'].append(np.mean(tmp_trad_psnr))
        results['trad_ssim'].append(np.mean(tmp_trad_ssim))

    sender.close()
    return results


def plot_curves(results):
    snr = results['snr']

    plt.figure(figsize=(12, 5))

    # Plot PSNR
    plt.subplot(1, 2, 1)
    plt.plot(snr, results['sem_psnr'], 'ro-', label='Semantic (DeepJSCC)')
    plt.plot(snr, results['trad_psnr'], 'bs--', label='Traditional (JPEG)')
    plt.xlabel('SNR (dB)')
    plt.ylabel('PSNR (dB)')
    plt.title('SNR vs PSNR')
    plt.grid(True)
    plt.legend()

    # Plot SSIM
    plt.subplot(1, 2, 2)
    plt.plot(snr, results['sem_ssim'], 'ro-', label='Semantic (DeepJSCC)')
    plt.plot(snr, results['trad_ssim'], 'bs--', label='Traditional (JPEG)')
    plt.xlabel('SNR (dB)')
    plt.ylabel('SSIM')
    plt.title('SNR vs SSIM')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig('performance_curves.png')
    print("\nCurves saved as 'performance_curves.png'")
    plt.show()


if __name__ == '__main__':
    data_results = run_evaluation()
    plot_curves(data_results)