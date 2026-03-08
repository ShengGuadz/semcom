"""
DeepJSCC 接收端 Flask 应用
提供Web界面、Socket服务器和图像解码功能
"""
import socket
import time
import pickle
import cv2
import webbrowser
from flask import Flask, render_template
from flask_socketio import SocketIO
import threading
import eventlet
import glob
import os
import numpy as np

eventlet.monkey_patch()

from receiver_jscc import ConfigReceiver, Receiver, calculate_psnr, calculate_ssim

# ================= 配置增强 =================
# 请确保此路径指向接收端的真实数据集，且文件名排序与发送端完全一致
DATASET_DIR = "./data/kodak/"
# DATASET_DIR = "./data/military_test/"
# ===========================================

# 初始化Flask应用
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# 初始化接收端
receiver = Receiver(ConfigReceiver())

def get_ground_truth_images(data_dir):
    """加载本地数据集所有路径，确保顺序与发送端一致"""
    if not os.path.exists(data_dir):
        print(f"Error: Dataset directory {data_dir} not found!")
        return []
    # 使用 sorted 确保顺序与发送端一致
    image_paths = sorted(glob.glob(os.path.join(data_dir, '*.*')))
    print(f"Found {len(image_paths)} images in {data_dir}")
    return image_paths

@app.route('/receiver')
def show_receiver():
    """显示接收端界面"""
    return render_template('receiver_deepjscc.html')

def receiver_server():
    """
    接收端服务器主循环
    监听Socket连接，接收数据并解码
    """
    # 建立Socket服务器
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((ConfigReceiver.host, ConfigReceiver.port))
    s.listen(5)

    print(f"\nReceiver listening on {ConfigReceiver.host}:{ConfigReceiver.port}")

    # === 预加载本地图片路径 ===
    gt_image_paths = get_ground_truth_images(DATASET_DIR)
    # ========================

    print("Waiting for sender connection...")
    conn, addr = s.accept()
    print(f"Connected by {addr}")

    counter = 1

    # 统计列表
    traditional_psnr_list = []
    semantic_psnr_list = []
    traditional_ssim_list = []
    semantic_ssim_list = []
    traditional_kb_list = []
    semantic_kb_list = []
    packet_kb_list = []

    while True:
        try:
            # ============ 接收数据 ============
            recv_start_time = time.time()

            data_length_bytes = conn.recv(4)
            if not data_length_bytes:
                print("Connection closed by sender")
                break

            data_length = int.from_bytes(data_length_bytes, byteorder='big')
            packet_kb_list.append(data_length / 1024.0)

            print(f"\n{'='*60}")
            print(f"[{time.time():.2f}] Receiving image {counter}")

            received = bytearray()
            while len(received) < data_length:
                chunk = conn.recv(min(4096, data_length - len(received)))
                if not chunk:
                    break
                received.extend(chunk)

            recv_time = time.time() - recv_start_time
            print(f"Reception time: {recv_time:.3f}s")

            # ============ 解析数据 ============
            data = pickle.loads(received)
            # 获取发送端传来的序号 (sender 发送的 counter 也是从1开始)
            img_idx = data['counter'] - 1

            # ============ 加载本地 Ground Truth ============
            ground_truth_img = None
            if img_idx < len(gt_image_paths):
                # 使用 cv2.imread 保持与接收到的 cv2 图片格式一致 (BGR)
                ground_truth_img = cv2.imread(gt_image_paths[img_idx])
                if ground_truth_img is None:
                    print(f"Warning: Failed to load local image: {gt_image_paths[img_idx]}")
                else:
                    # 确保尺寸匹配 (DeepJSCC 可能会resize，或者发送端做了resize)
                    # 这里的逻辑是：以发送过来的图像尺寸为准，如果本地图太大，可能需要resize
                    # 但为了严格对比，通常应该保证数据预处理一致。
                    pass
            else:
                print(f"Warning: Image index {img_idx} out of range for local dataset.")

            # ============ 解码阶段 ============
            print(f"[{time.time():.2f}] Decoding image {counter}...")
            decode_start_time = time.time()

            # 注意：这里返回的 jpeg_image_clean 是发送端压缩过的无噪图，不要用它做 Ground Truth
            semantic_image, jpeg_image_clean, feature_quantized, traditional_image_size = \
                receiver._decode(data)

            decode_time = time.time() - decode_start_time
            print(f"Decoding time: {decode_time*1000:.3f}ms")

            # ============ 生成传统通信结果 (JPEG + 模拟误码) ============
            # 1. 提取JPEG字节数组
            jpeg_data_clean = data['image']

            # 2. 根据SNR模拟比特错误
            corrupted_jpeg_data = receiver._simulate_bit_errors(
                jpeg_data_clean,
                ConfigReceiver.snr,
                bandwidth=100e6
            )

            # 3. 解码 *受损* 的JPEG数据
            noisy_jpeg_image = cv2.imdecode(
                np.frombuffer(corrupted_jpeg_data, np.uint8),
                cv2.IMREAD_COLOR
            )

            if noisy_jpeg_image is None:
                print("JPEG decoding failed due to channel errors.")
                # 如果解码失败，用全黑图代替，或者用严重模糊的图
                noisy_jpeg_image = np.zeros_like(semantic_image)
            else:
                # 确保尺寸一致 (有些JPEG解码可能会有微小差异，或者如果本地图尺寸不同)
                if noisy_jpeg_image.shape != semantic_image.shape:
                     noisy_jpeg_image = cv2.resize(noisy_jpeg_image, (semantic_image.shape[1], semantic_image.shape[0]))

            # ============ 关键修正：准备用于计算指标的参考图 ============
            target_ref_image = None

            if ground_truth_img is not None:
                print("Using LOCAL DATASET image as Ground Truth.")
                target_ref_image = ground_truth_img

                # 如果本地原图尺寸和处理后的尺寸不一致（例如发送端做了Resize），需要缩放本地原图
                if target_ref_image.shape != semantic_image.shape:
                     target_ref_image = cv2.resize(target_ref_image, (semantic_image.shape[1], semantic_image.shape[0]))
            else:
                print("Warning: Using SENT JPEG (Clean) as Ground Truth (Not Recommended for fair comparison).")
                target_ref_image = jpeg_image_clean

            # ============ 保存图像 ============
            semantic_image_path = f"{receiver.config.save_dir_semantic}{counter}.png"
            traditional_image_path = f"{receiver.config.save_dir_traditional}{counter}.png"

            cv2.imwrite(semantic_image_path, semantic_image)
            cv2.imwrite(traditional_image_path, noisy_jpeg_image)

            # ============ 计算质量指标 ============
            print(f"[{time.time():.2f}] Computing quality metrics...")

            # 对比：本地原图 vs 传统(加噪JPEG)
            traditional_psnr = calculate_psnr(target_ref_image, noisy_jpeg_image)
            # 对比：本地原图 vs 语义(JSCC)
            semantic_psnr = calculate_psnr(target_ref_image, semantic_image)

            # SSIM
            try:
                traditional_ssim = calculate_ssim(target_ref_image, noisy_jpeg_image)
                semantic_ssim = calculate_ssim(target_ref_image, semantic_image)
            except Exception as e:
                print(f"SSIM calculation failed: {e}")
                traditional_ssim = 0.0
                semantic_ssim = 0.0

            print(f"Traditional (JPEG+Noise vs Raw) - PSNR: {traditional_psnr:.2f} dB, SSIM: {traditional_ssim:.4f}")
            print(f"Semantic    (DeepJSCC   vs Raw) - PSNR: {semantic_psnr:.2f} dB, SSIM: {semantic_ssim:.4f}")

            # ... (后续的统计列表 append, 计算 compression ratio, socket emit 等代码保持不变) ...

            # 这里的 image_size 要用原图的大小，而不是 JPEG 大小
            original_image_size = data['original_image_size'] # 这是发送端计算的JPEG大小，如果要算压缩率可能需要用bmp大小

            # 修正压缩比计算：通常压缩比 = 未压缩图大小 / 传输大小
            # 未压缩大小 (KB) = H * W * 3 / 1024
            # raw_size_kb = (target_ref_image.shape[0] * target_ref_image.shape[1] * 3) / 1024
            #
            semantic_feature_size = feature_quantized.nbytes // 1024
            traditional_compression_ratio = round(
                (original_image_size - traditional_image_size) * 100 / original_image_size,
                2
            )
            semantic_compression_ratio = round(
                (original_image_size - semantic_feature_size) * 100 / original_image_size,
                2
            )
            # 吞吐量（假设固定带宽 100 Mbps = 102400 Kbps）
            bandwidth_kbps = 102400
            traditional_throughput = round(bandwidth_kbps / traditional_image_size, 2)
            semantic_throughput = round(bandwidth_kbps / max(semantic_feature_size, 1), 2)

            print(f"Compression - Traditional: {traditional_compression_ratio}%, "
                  f"Semantic: {semantic_compression_ratio}%")
            print(f"Throughput - Traditional: {traditional_throughput} imgs/s, "
                  f"Semantic: {semantic_throughput} imgs/s")

            traditional_kb_list.append(float(traditional_image_size))
            semantic_kb_list.append(float(semantic_feature_size))
            traditional_psnr_list.append(traditional_psnr)
            semantic_psnr_list.append(semantic_psnr)
            traditional_ssim_list.append(traditional_ssim)
            semantic_ssim_list.append(semantic_ssim)

            # ============ 推送到前端 ============
            socketio.emit('message', {
                'traditional_image_url': traditional_image_path,
                'semantic_image_url': semantic_image_path,
                'traditional_image_size': traditional_image_size, # JPEG大小
                'semantic_feature_size': semantic_feature_size,
                'traditional_psnr': round(traditional_psnr, 2),
                'semantic_psnr': round(semantic_psnr, 2),
                'traditional_ssim': round(traditional_ssim, 4),
                'semantic_ssim': round(semantic_ssim, 4),
                'traditional_compression_ratio': traditional_compression_ratio, # 修正显示
                'semantic_compression_ratio': semantic_compression_ratio,     # 修正显示
                'traditional_throughput': traditional_throughput, # 这里可以保留你原来的计算逻辑
                'semantic_throughput': semantic_throughput,
                'snr': ConfigReceiver.snr,
            })

            # ... (ACK 发送和 counter+=1 保持不变) ...

            ack = (1).to_bytes(4, byteorder='big')
            conn.sendall(ack)
            counter += 1

        except Exception as e:
            print(f"Error processing image: {e}")
            import traceback
            traceback.print_exc()
            break
    # ======== 修复点：确保以下代码块缩进回退，与 while 关键字对齐 ========
    print(f"\n{'=' * 20} FINAL AVERAGE RESULTS {'=' * 20}")

    if traditional_psnr_list:
        avg_trad_psnr = np.mean(traditional_psnr_list)
        avg_sem_psnr = np.mean(semantic_psnr_list)
        avg_trad_ssim = np.mean(traditional_ssim_list)
        avg_sem_ssim = np.mean(semantic_ssim_list)
        avg_trad_kb = np.mean(traditional_kb_list)
        avg_sem_kb = np.mean(semantic_kb_list)

        print(f"Total Images Processed: {len(traditional_psnr_list)}")
        print(f"Average PSNR - Traditional: {avg_trad_psnr:.2f} dB")
        print(f"Average PSNR - Semantic:    {avg_sem_psnr:.2f} dB")
        print(f"Average SSIM - Traditional: {avg_trad_ssim:.4f}")
        print(f"Average SSIM - Semantic:    {avg_sem_ssim:.4f}")
        print(f"Average Size - JPEG:        {avg_trad_kb:.2f} KB")
        print(f"Average Size - Semantic:    {avg_sem_kb:.2f} KB")

        if avg_trad_kb > 0:
            saving = (1.0 - avg_sem_kb / avg_trad_kb) * 100.0
            print(f"Data Saving (Semantic vs JPEG): {saving:.2f}%")
    else:
        print("No images were processed.")

    print(f"{'=' * 63}\n")
    # ... (循环结束后的平均值计算保持不变) ...
    conn.close()
    s.close()


if __name__ == '__main__':
    # 启动接收端服务器线程
    threading.Thread(target=receiver_server, daemon=True).start()

    # 打开浏览器
    # webbrowser.open("http://127.0.0.1:5004/receiver")

    # 启动Flask应用
    print("\nStarting DeepJSCC Receiver...")
    print(f"Device: {ConfigReceiver.device}")
    print(f"Model: c={ConfigReceiver.c}, SNR={ConfigReceiver.snr}dB")
    print(f"Web interface: http://127.0.0.1:5004/receiver")

    socketio.run(app, host='0.0.0.0', port=5004, debug=False)
