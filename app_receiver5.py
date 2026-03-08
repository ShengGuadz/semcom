"""
DeepJSCC 接收端 Flask 应用
集成熵解码 + 分包接收 + 全链路时延统计
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
import zlib

eventlet.monkey_patch()

from receiver_jscc import ConfigReceiver, Receiver, calculate_psnr, calculate_ssim

# ================= 配置增强 =================
DATASET_DIR = "./data/kodak/"
# ===========================================

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")
receiver = Receiver(ConfigReceiver())


def get_ground_truth_images(data_dir):
    if not os.path.exists(data_dir):
        return []
    return sorted(glob.glob(os.path.join(data_dir, '*.*')))


@app.route('/receiver')
def show_receiver():
    return render_template('receiver5.html')


def print_summary_report(metrics):
    """新增：打印平均指标统计函数"""
    num_imgs = len(metrics)
    avg = lambda key: sum(m[key] for m in metrics) / num_imgs

    print(f"\n" + "=" * 20 + f" FINAL STATISTICS (Excluding 1st Image, Total: {num_imgs}) " + "=" * 20)

    # 创建对比表格格式
    print(f"{'Metric':<15} | {'Traditional (JPEG)':<20} | {'Semantic (JSCC)':<20}")
    print("-" * 60)
    print(f"{'Avg PSNR (dB)':<15} | {avg('jpeg_psnr'):<20.2f} | {avg('sem_psnr'):<20.2f}")
    print(f"{'Avg SSIM':<15} | {avg('jpeg_ssim'):<20.4f} | {avg('sem_ssim'):<20.4f}")
    print(f"{'Avg Size (KB)':<15} | {avg('jpeg_size'):<20.2f} | {avg('sem_size'):<20.2f}")
    print(f"{'Avg Latency (ms)':<15} | {avg('jpeg_latency') * 1000:<20.2f} | {avg('sem_latency') * 1000:<20.2f}")
    print("=" * 75)
def receive_packet(conn):
    """接收数据包并计算真实的物理传输耗时"""
    # 1. 接收长度
    data_length_bytes = conn.recv(4)
    if not data_length_bytes:
        return None, 0, 0
    data_length = int.from_bytes(data_length_bytes, byteorder='big')

    # 2. 接收主体数据
    received = bytearray()
    while len(received) < data_length:
        chunk = conn.recv(min(4096, data_length - len(received)))
        if not chunk:
            break
        received.extend(chunk)

    # 记录收到完整包的时间戳
    arrival_time = time.time()

    # 3. 发送 ACK
    ack = (1).to_bytes(4, byteorder='big')
    conn.sendall(ack)

    # 4. 解析数据并计算差值geic
    data_dict = pickle.loads(received)

    # 真实传输时延 = 接收完成时刻 - 发送开始时刻
    if 'send_timestamp' in data_dict:
        net_delay = arrival_time - data_dict['send_timestamp']
    else:
        net_delay = 0

    return data_dict, net_delay, data_length

def receiver_server():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((ConfigReceiver.host, ConfigReceiver.port))
    s.listen(5)

    print(f"\nReceiver listening on {ConfigReceiver.host}:{ConfigReceiver.port}")
    gt_image_paths = get_ground_truth_images(DATASET_DIR)
    conn, addr = s.accept()
    print(f"Connected by {addr}")

    counter = 1
    all_metrics = []

    while True:
        try:
            print(f"\n{'=' * 60}")
            print(f"Waiting for image {counter}...")

            # ============ 接收 Packet 1 (JPEG) ============
            data_jpeg, rx_time_jpeg, len_jpeg = receive_packet(conn)
            if data_jpeg is None: break

            # 获取发送端传来的 JPEG 编码时间
            sender_time_jpeg = data_jpeg.get('encode_time', 0)

            # ============ 接收 Packet 2 (Semantic) ============
            data_sem, rx_time_sem, len_sem = receive_packet(conn)
            if data_sem is None: break

            # 获取发送端传来的 语义 编码时间
            sender_time_sem = data_sem.get('encode_time', 0)

            # 合并数据
            data = {}
            data.update(data_jpeg)
            data.update(data_sem)
            img_idx = data['counter'] - 1

            # ============ 1. 语义解码处理时延 (Receiver Processing) ============
            print(f"[{time.time():.2f}] Semantic Decoding...")
            rec_sem_start = time.time()

            # 熵解码 (Zlib Decompress)
            compressed_feature, min_val, max_val, feature_shape, feature_dtype = data['feature']
            decompressed_bytes = zlib.decompress(compressed_feature)
            feature_quantized = np.frombuffer(decompressed_bytes, dtype=feature_dtype)
            feature_quantized = feature_quantized.reshape(feature_shape)
            data['feature'] = [feature_quantized, min_val, max_val, feature_shape]

            # 深度学习解码 (Decoder)
            semantic_image, jpeg_image_clean, _, traditional_image_size = receiver._decode(data)

            # 计算接收端处理时间
            receiver_time_sem = time.time() - rec_sem_start

            # ============ 2. 传统解码处理时延 (Receiver Processing) ============
            print(f"[{time.time():.2f}] JPEG Decoding...")
            rec_jpeg_start = time.time()

            jpeg_data_clean = data['image']
            corrupted_jpeg_data = receiver._simulate_bit_errors(
                jpeg_data_clean, ConfigReceiver.snr, bandwidth=100e6
            )
            # 模拟信道后解码
            noisy_jpeg_image = cv2.imdecode(np.frombuffer(corrupted_jpeg_data, np.uint8), cv2.IMREAD_COLOR)

            if noisy_jpeg_image is None:
                noisy_jpeg_image = np.zeros_like(semantic_image)
            elif noisy_jpeg_image.shape != semantic_image.shape:
                noisy_jpeg_image = cv2.resize(noisy_jpeg_image, (semantic_image.shape[1], semantic_image.shape[0]))

            receiver_time_jpeg = time.time() - rec_jpeg_start

            # ============ 计算总时延 (Total Latency) ============
            # 总时延 = 发送端处理 + 网络传输(接收耗时) + 接收端处理
            total_time_jpeg = sender_time_jpeg + rx_time_jpeg + receiver_time_jpeg
            total_time_sem = sender_time_sem + rx_time_sem + receiver_time_sem

            print(f"Total JPEG Latency: {total_time_jpeg * 1000:.2f} ms")
            print(f"  - Sender: {sender_time_jpeg * 1000:.2f} ms")
            print(f"  - Network: {rx_time_jpeg * 1000:.2f} ms")
            print(f"  - Receiver: {receiver_time_jpeg * 1000:.2f} ms")

            print(f"Total Semantic Latency: {total_time_sem * 1000:.2f} ms")
            print(f"  - Sender: {sender_time_sem * 1000:.2f} ms")
            print(f"  - Network: {rx_time_sem * 1000:.2f} ms")
            print(f"  - Receiver: {receiver_time_sem * 1000:.2f} ms")

            # ============ 准备参考图 ============
            ground_truth_img = None
            if img_idx < len(gt_image_paths):
                ground_truth_img = cv2.imread(gt_image_paths[img_idx])
            target_ref_image = ground_truth_img if ground_truth_img is not None else jpeg_image_clean
            if target_ref_image.shape != semantic_image.shape:
                target_ref_image = cv2.resize(target_ref_image, (semantic_image.shape[1], semantic_image.shape[0]))

            # 保存与计算指标
            semantic_image_path = f"{receiver.config.save_dir_semantic}{counter}.png"
            traditional_image_path = f"{receiver.config.save_dir_traditional}{counter}.png"
            cv2.imwrite(semantic_image_path, semantic_image)
            cv2.imwrite(traditional_image_path, noisy_jpeg_image)

            traditional_psnr = calculate_psnr(target_ref_image, noisy_jpeg_image)
            semantic_psnr = calculate_psnr(target_ref_image, semantic_image)
            try:
                traditional_ssim = calculate_ssim(target_ref_image, noisy_jpeg_image)
                semantic_ssim = calculate_ssim(target_ref_image, semantic_image)
            except:
                traditional_ssim, semantic_ssim = 0.0, 0.0

            # 大小计算
            original_image_size = data['original_image_size']
            semantic_feature_size = len(compressed_feature) / 1024.0
            if counter > 2:
                all_metrics.append({
                    'jpeg_psnr': traditional_psnr,
                    'sem_psnr': semantic_psnr,
                    'jpeg_ssim': traditional_ssim,
                    'sem_ssim': semantic_ssim,
                    'jpeg_size': traditional_image_size,
                    'sem_size': semantic_feature_size,
                    'jpeg_latency': total_time_jpeg,
                    'sem_latency': total_time_sem
                })

            # ============ 推送到前端 ============
            socketio.emit('message', {
                'traditional_image_url': traditional_image_path,
                'semantic_image_url': semantic_image_path,
                'traditional_image_size': traditional_image_size,
                'semantic_feature_size': round(semantic_feature_size, 2),
                'traditional_psnr': round(traditional_psnr, 2),
                'semantic_psnr': round(semantic_psnr, 2),
                'traditional_ssim': round(traditional_ssim, 4),
                'semantic_ssim': round(semantic_ssim, 4),
                'traditional_compression_ratio': round(
                    (original_image_size - traditional_image_size) * 100 / original_image_size, 2),
                'semantic_compression_ratio': round(
                    (original_image_size - semantic_feature_size) * 100 / original_image_size, 2),
                'snr': ConfigReceiver.snr,

                # === 详细时延数据 ===
                # JPEG
                'time_jpeg_sender': sender_time_jpeg,
                'time_jpeg_net': rx_time_jpeg,
                'time_jpeg_receiver': receiver_time_jpeg,
                'time_jpeg_total': total_time_jpeg,

                # Semantic
                'time_sem_sender': sender_time_sem,
                'time_sem_net': rx_time_sem,
                'time_sem_receiver': receiver_time_sem,
                'time_sem_total': total_time_sem
            })

            counter += 1

        except Exception as e:
            print(f"Error processing image: {e}")
            import traceback
            traceback.print_exc()
            break
    if len(all_metrics) > 0:
        print_summary_report(all_metrics)

    conn.close()
    s.close()


if __name__ == '__main__':
    threading.Thread(target=receiver_server, daemon=True).start()
    print(f"Receiver Web Interface: http://127.0.0.1:5004/receiver")
    socketio.run(app, host='0.0.0.0', port=5004, debug=False)