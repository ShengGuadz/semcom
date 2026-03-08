import os
import torch
import numpy as np
import cv2
import zlib
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as calculate_psnr
from skimage.metrics import structural_similarity as calculate_ssim

# === 导入你提供的 Hailo 发送端和 Torch 接收端 ===
from sender_hailo import ConfigSender, Sender
from receiver_hailo import ConfigReceiver, Receiver

# 配置设备 (接收端解码通常在CPU或GPU，Hailo只负责发送端推理)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def add_awgn_to_feature(feature_tensor, snr_db):
    """
    对语义特征添加加性高斯白噪声 (AWGN)
    注意：feature_tensor 应该是 PyTorch Tensor
    """
    # 1. 计算信号功率 (P_signal = E[x^2])
    signal_power = torch.mean(feature_tensor ** 2)

    # 2. 根据 SNR 计算噪声功率
    # P_noise = P_signal / 10^(SNR/10)
    noise_power = signal_power / (10 ** (snr_db / 10.0))

    # 3. 生成噪声并添加
    noise_std = torch.sqrt(noise_power)
    noise = torch.randn_like(feature_tensor) * noise_std

    return feature_tensor + noise


def find_matching_jpeg(image_bgr, target_size_bytes):
    """
    二分查找：寻找 JPEG 质量因子，使得 JPEG 大小最接近 target_size_bytes
    """
    low, high = 1, 100
    best_quality = 1
    best_jpeg_data = None
    min_diff = float('inf')

    while low <= high:
        mid = (low + high) // 2
        quality = max(1, mid)

        # 编码
        encoded_data = cv2.imencode('.jpg', image_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])[1].tobytes()
        current_size = len(encoded_data)

        diff = abs(current_size - target_size_bytes)

        if diff < min_diff:
            min_diff = diff
            best_quality = quality
            best_jpeg_data = encoded_data

        if current_size > target_size_bytes:
            high = mid - 1
        else:
            low = mid + 1

    return best_jpeg_data, best_quality


def run_evaluation_hailo():
    # === 配置评估参数 ===
    snr_list = [-10,-5,0, 5, 10, 15, 20]  # 测试的信噪比列表 (dB)

    results = {
        'snr': snr_list,
        'sem_psnr': [], 'sem_ssim': [],
        'trad_psnr': [], 'trad_ssim': [],
        'avg_size_kb': []
    }

    # === 初始化 ===
    print("Initializing Sender (Hailo) and Receiver (Torch)...")
    sender = Sender(ConfigSender())
    receiver = Receiver(ConfigReceiver())

    # 确保接收端模型处于评估模式
    receiver.model.eval()

    print(f"Starting evaluation on {len(sender.test_loader)} images...")

    for snr in snr_list:
        print(f"\nEvaluating SNR = {snr} dB")

        tmp_sem_psnr, tmp_sem_ssim = [], []
        tmp_trad_psnr, tmp_trad_ssim = [], []
        tmp_size_kb = []

        # 遍历测试集
        # sender.test_loader 返回的是 Tensor (B, C, H, W)
        for i, image_tensor in enumerate(tqdm(sender.test_loader)):
            # -------------------------------------------
            # 1. 准备 Ground Truth (原始图像)
            # -------------------------------------------
            # 获取原始图像 Numpy RGB [H, W, 3] 用于后续指标计算
            orig_img_np = image_tensor.squeeze(0).permute(1, 2, 0).numpy()
            orig_img_np = (orig_img_np * 255).clip(0, 255).astype('uint8')

            # 准备 BGR 格式用于 OpenCV JPEG 编码
            orig_img_bgr = cv2.cvtColor(orig_img_np, cv2.COLOR_RGB2BGR)

            # -------------------------------------------
            # 2. 语义发送端 (Hailo)
            # -------------------------------------------
            # 调用 sender_hailo.py 中的 encode
            # 返回: 量化特征(uint8), 最小值, 最大值, 是否旋转标记, 处理后的输入图
            feature_q, min_v, max_v, rotated, _ = sender.encode(image_tensor)

            # -------------------------------------------
            # 3. 熵编码 (计算数据量)
            # -------------------------------------------
            # 转为 bytes 并使用 zlib 压缩
            feature_bytes = feature_q.tobytes()
            compressed_feature = zlib.compress(feature_bytes, level=9)

            semantic_size_bytes = len(compressed_feature)
            tmp_size_kb.append(semantic_size_bytes / 1024.0)

            # -------------------------------------------
            # 4. 信道模拟 (模拟 DeepJSCC 特性)
            # -------------------------------------------
            # 反量化 (Numpy uint8 -> Torch Tensor float)
            feature_tensor = receiver._dequantize(feature_q, min_v, max_v)

            # 添加 AWGN 噪声
            feature_noisy = add_awgn_to_feature(feature_tensor, snr)

            # -------------------------------------------
            # 5. 语义接收端 (Torch Decoder)
            # -------------------------------------------
            # 确保维度是 NCHW
            if feature_noisy.dim() == 3:
                feature_noisy = feature_noisy.unsqueeze(0)

            with torch.no_grad():
                # 解码
                recon = receiver.model.decoder(feature_noisy)

                # 转为 Numpy HWC RGB
                recon_np = recon.squeeze(0).cpu().numpy()
                recon_np = np.transpose(recon_np, (1, 2, 0))
                recon_np = (recon_np * 255).clip(0, 255).astype(np.uint8)

                # === 关键：处理 Hailo 引入的旋转 ===
                # 如果发送端做了旋转 (H,W -> W,H)，解码出来的图也是 (W,H)
                # 我们需要转置回去以便与原图 (H,W) 对比
                if rotated == 1:
                    # transpose(1, 0, 2) 对应 swap(H, W)
                    recon_np = np.transpose(recon_np, (1, 0, 2))

                sem_img_rgb = recon_np

            # -------------------------------------------
            # 6. 传统通信对比 (JPEG + 误码)
            # -------------------------------------------
            # 寻找大小相近的 JPEG
            jpeg_data_clean, quality = find_matching_jpeg(orig_img_bgr, semantic_size_bytes)

            # 模拟数字信道误码 (Bit Errors)
            jpeg_data_noisy = receiver._simulate_bit_errors(jpeg_data_clean, snr)

            # 解码 JPEG
            trad_img_bgr = cv2.imdecode(np.frombuffer(jpeg_data_noisy, np.uint8), cv2.IMREAD_COLOR)

            if trad_img_bgr is None:
                # 解码失败 (高误码率下)，用黑图代替
                trad_img_rgb = np.zeros_like(orig_img_np)
            else:
                trad_img_rgb = cv2.cvtColor(trad_img_bgr, cv2.COLOR_BGR2RGB)

            # -------------------------------------------
            # 7. 尺寸对齐 & 计算指标
            # -------------------------------------------
            # Hailo 模型如果强制 Resize 了输入，解码出的图可能与原图尺寸不一致
            # 我们将解码图 Resize 回原图尺寸，计算 End-to-End 的质量

            if sem_img_rgb.shape != orig_img_np.shape:
                sem_img_rgb = cv2.resize(sem_img_rgb, (orig_img_np.shape[1], orig_img_np.shape[0]))

            if trad_img_rgb.shape != orig_img_np.shape:
                trad_img_rgb = cv2.resize(trad_img_rgb, (orig_img_np.shape[1], orig_img_np.shape[0]))

            # PSNR
            sem_p = calculate_psnr(orig_img_np, sem_img_rgb)
            trad_p = calculate_psnr(orig_img_np, trad_img_rgb)

            # SSIM (RGB图需指定channel_axis)
            sem_s = calculate_ssim(orig_img_np, sem_img_rgb, channel_axis=2)
            trad_s = calculate_ssim(orig_img_np, trad_img_rgb, channel_axis=2)

            tmp_sem_psnr.append(sem_p)
            tmp_sem_ssim.append(sem_s)
            tmp_trad_psnr.append(trad_p)
            tmp_trad_ssim.append(trad_s)

        # 记录平均值
        results['sem_psnr'].append(np.mean(tmp_sem_psnr))
        results['sem_ssim'].append(np.mean(tmp_sem_ssim))
        results['trad_psnr'].append(np.mean(tmp_trad_psnr))
        results['trad_ssim'].append(np.mean(tmp_trad_ssim))
        results['avg_size_kb'].append(np.mean(tmp_size_kb))

        print(f"  -> Semantic (Hailo): PSNR={results['sem_psnr'][-1]:.2f}, SSIM={results['sem_ssim'][-1]:.4f}")
        print(f"  -> Traditional (JPEG): PSNR={results['trad_psnr'][-1]:.2f}, SSIM={results['trad_ssim'][-1]:.4f}")
        print(f"  -> Avg Size: {results['avg_size_kb'][-1]:.2f} KB")

    # 释放 Hailo 资源
    sender.close()

    return results


def plot_curves(results):
    snr = results['snr']

    plt.figure(figsize=(14, 6))

    # --- Plot 1: PSNR ---
    plt.subplot(1, 2, 1)
    plt.plot(snr, results['sem_psnr'], 'r-o', linewidth=2, label='DeepJSCC (Hailo)')
    plt.plot(snr, results['trad_psnr'], 'b--s', linewidth=2, label='JPEG (Traditional)')
    plt.xlabel('SNR (dB)', fontsize=12)
    plt.ylabel('PSNR (dB)', fontsize=12)
    plt.title(f'PSNR Performance (Avg Size: {np.mean(results["avg_size_kb"]):.1f} KB)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)

    # --- Plot 2: SSIM ---
    plt.subplot(1, 2, 2)
    plt.plot(snr, results['sem_ssim'], 'r-o', linewidth=2, label='DeepJSCC (Hailo)')
    plt.plot(snr, results['trad_ssim'], 'b--s', linewidth=2, label='JPEG (Traditional)')
    plt.xlabel('SNR (dB)', fontsize=12)
    plt.ylabel('SSIM', fontsize=12)
    plt.title('SSIM Performance', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)

    plt.tight_layout()
    save_path = 'hailo_performance_evaluation.png'
    plt.savefig(save_path, dpi=300)
    print(f"\nCurves saved to: {os.path.abspath(save_path)}")
    # plt.show() # 如果在无界面服务器上运行，请注释掉此行


if __name__ == '__main__':
    print("=== DeepJSCC Hailo Evaluation Script ===")
    data_results = run_evaluation_hailo()
    plot_curves(data_results)