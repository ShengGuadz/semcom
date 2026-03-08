"""
app_receiver_ptq.py
DeepJSCC 接收端 Flask 应用 (PTQ Int8 量化版)
专门用于加载 receiver_ptq.py 和 deepjscc_int8.pth
"""
import socket
import time
import pickle
import cv2
import threading
import eventlet
import numpy as np
from flask import Flask, render_template
from flask_socketio import SocketIO

# === 关键修改：导入 PTQ 版本的配置和接收类 ===
from receiver_ptq import ConfigReceiverPTQ, ReceiverPTQ, calculate_psnr, calculate_ssim
from quant_model import QuantizableDeepJSCC

eventlet.monkey_patch()

# 初始化Flask应用
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# === 初始化 PTQ 接收端 ===
receiver = ReceiverPTQ(ConfigReceiverPTQ())


@app.route('/receiver')
def show_receiver():
    """显示接收端界面 (复用原有的HTML模板)"""
    return render_template('receiver_deepjscc.html')


def receiver_server():
    """
    接收端服务器主循环
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    # 使用 ConfigReceiverPTQ 配置
    try:
        s.bind((ConfigReceiverPTQ.host, ConfigReceiverPTQ.port))
        s.listen(5)
    except OSError as e:
        print(f"Error binding port: {e}")
        return

    print(f"\nReceiver listening on {ConfigReceiverPTQ.host}:{ConfigReceiverPTQ.port}")
    print("Waiting for sender connection...")

    conn, addr = s.accept()
    print(f"Connected by {addr}")

    counter = 1

    # ============ 新增：初始化指标累积列表 ============
    traditional_psnr_list = []
    semantic_psnr_list = []
    traditional_ssim_list = []
    semantic_ssim_list = []
    traditional_kb_list = []  # JPEG payload KB/张
    semantic_kb_list = []  # 语义特征 payload KB/张
    packet_kb_list = []  # 实际socket整包 KB/张

    while True:
        try:
            # ============ 接收数据 ============
            recv_start_time = time.time()
            data_length_bytes = conn.recv(4)
            if not data_length_bytes:
                print("Connection closed by sender")
                break

            data_length = int.from_bytes(data_length_bytes, byteorder='big')
            # 记录数据包大小
            packet_kb_list.append(data_length / 1024.0)

            print(f"\n{'=' * 60}")
            print(f"[{time.time():.2f}] Receiving image {counter}")

            received = bytearray()
            while len(received) < data_length:
                chunk = conn.recv(min(4096, data_length - len(received)))
                if not chunk:
                    break
                received.extend(chunk)

            # ============ 解析数据 ============
            data = pickle.loads(received)
            counter = data['counter']

            # ============ 解码阶段 (PTQ) ============
            print(f"[{time.time():.2f}] Decoding image {counter}...")
            decode_start_time = time.time()

            # 调用 receiver_ptq 中的解码逻辑
            semantic_image, original_image, feature_quantized, traditional_image_size = \
                receiver._decode(data)

            decode_time = time.time() - decode_start_time
            print(f"Decoding time: {decode_time * 1000:.3f}ms")

            # ============ 生成传统通信结果对比 ============
            # 1. 提取JPEG字节流
            jpeg_data = data['image']

            # 2. 模拟误码 (使用 ReceiverPTQ 中的方法)
            corrupted_jpeg_data = receiver._simulate_bit_errors(
                jpeg_data,
                ConfigReceiverPTQ.snr,
                bandwidth=100e6
            )

            # 3. 解码受损 JPEG
            traditional_image = cv2.imdecode(
                np.frombuffer(corrupted_jpeg_data, np.uint8),
                cv2.IMREAD_COLOR
            )

            if traditional_image is None:
                print("警告：JPEG解码失败，回退至像素域加噪")
                traditional_image = receiver._add_gaussian_noise(original_image, ConfigReceiverPTQ.snr)

            # ============ 保存图像 ============
            semantic_image_path = f"{receiver.config.save_dir_semantic}{counter}.png"
            traditional_image_path = f"{receiver.config.save_dir_traditional}{counter}.png"

            cv2.imwrite(semantic_image_path, semantic_image)
            cv2.imwrite(traditional_image_path, traditional_image)

            # ============ 计算指标 ============
            traditional_psnr = calculate_psnr(original_image, traditional_image)
            semantic_psnr = calculate_psnr(original_image, semantic_image)

            try:
                traditional_ssim = calculate_ssim(original_image, traditional_image)
                semantic_ssim = calculate_ssim(original_image, semantic_image)
            except:
                traditional_ssim = 0.0
                semantic_ssim = 0.0

            print(f"Traditional - PSNR: {traditional_psnr:.2f} dB, SSIM: {traditional_ssim:.4f}")
            print(f"Semantic (Int8) - PSNR: {semantic_psnr:.2f} dB, SSIM: {semantic_ssim:.4f}")

            # ============ 累积每张图的指标 ============
            traditional_psnr_list.append(traditional_psnr)
            semantic_psnr_list.append(semantic_psnr)
            traditional_ssim_list.append(traditional_ssim)
            semantic_ssim_list.append(semantic_ssim)

            # ============ 计算传输参数 ============
            original_image_size = data['original_image_size']
            semantic_feature_size = feature_quantized.nbytes // 1024

            # 累积大小指标
            traditional_kb_list.append(float(traditional_image_size))
            semantic_kb_list.append(float(semantic_feature_size))

            # 压缩比
            trad_comp = round((original_image_size - traditional_image_size) * 100 / original_image_size, 2)
            sem_comp = round((original_image_size - semantic_feature_size) * 100 / original_image_size, 2)

            # 吞吐量估算
            bandwidth_kbps = 102400
            trad_throughput = round(bandwidth_kbps / max(traditional_image_size, 1), 2)
            sem_throughput = round(bandwidth_kbps / max(semantic_feature_size, 1), 2)

            # ============ 推送前端 ============
            socketio.emit('message', {
                'traditional_image_url': traditional_image_path,
                'semantic_image_url': semantic_image_path,
                'traditional_image_size': traditional_image_size,
                'semantic_feature_size': semantic_feature_size,
                'traditional_psnr': round(traditional_psnr, 2),
                'semantic_psnr': round(semantic_psnr, 2),
                'traditional_ssim': round(traditional_ssim, 4),
                'semantic_ssim': round(semantic_ssim, 4),
                'traditional_compression_ratio': trad_comp,
                'semantic_compression_ratio': sem_comp,
                'traditional_throughput': trad_throughput,
                'semantic_throughput': sem_throughput,
            })

            # 发送ACK
            ack = (1).to_bytes(4, byteorder='big')
            conn.sendall(ack)
            counter += 1

        except Exception as e:
            print(f"Error processing image: {e}")
            import traceback
            traceback.print_exc()
            break

    # ============ 新增：计算并显示平均指标 (在连接断开后) ============
    if traditional_psnr_list:  # 确保有数据才计算
        avg_traditional_psnr = sum(traditional_psnr_list) / len(traditional_psnr_list)
        avg_semantic_psnr = sum(semantic_psnr_list) / len(semantic_psnr_list)
        avg_traditional_ssim = sum(traditional_ssim_list) / len(traditional_ssim_list)
        avg_semantic_ssim = sum(semantic_ssim_list) / len(semantic_ssim_list)
        avg_trad_kb = sum(traditional_kb_list) / len(traditional_kb_list)
        avg_sem_kb = sum(semantic_kb_list) / len(semantic_kb_list)
        avg_pkt_kb = sum(packet_kb_list) / len(packet_kb_list)

        print(f"\n{'=' * 60}")
        print(f"TOTAL IMAGES PROCESSED: {len(traditional_psnr_list)}")
        print(f"AVERAGE QUALITY METRICS:")
        print(f"Traditional - PSNR: {avg_traditional_psnr:.2f} dB, SSIM: {avg_traditional_ssim:.4f}")
        print(f"Semantic - PSNR: {avg_semantic_psnr:.2f} dB, SSIM: {avg_semantic_ssim:.4f}")
        print("AVERAGE BANDWIDTH SIZE (KB per image):")
        print(f"JPEG payload: {avg_trad_kb:.2f} KB/img")
        print(f"Semantic payload (feature only): {avg_sem_kb:.2f} KB/img")
        print(f"Actual socket packet (JPEG+Semantic): {avg_pkt_kb:.2f} KB/img")

        if avg_trad_kb > 0:
            saving = (1.0 - avg_sem_kb / avg_trad_kb) * 100.0
            print(f"Semantic vs JPEG payload saving: {saving:.2f}%")
        print(f"{'=' * 60}")

    conn.close()
    s.close()
    print("\nReceiver stopped.")


if __name__ == '__main__':
    threading.Thread(target=receiver_server, daemon=True).start()

    print("\nStarting DeepJSCC Receiver (PTQ Version)...")
    print(f"Device: {ConfigReceiverPTQ.device}")
    print(f"Web interface: http://127.0.0.1:5004/receiver")

    # 注意：端口保持 5004，与原版一致
    socketio.run(app, host='0.0.0.0', port=5004, debug=False)