"""
DeepJSCC 发送端 Flask 应用
集成熵编码 + 分包传输 + 时延统计 (Sender Processing Time)
"""
import socket
import torch
import cv2
import time
import pickle
import webbrowser
import zlib
import numpy as np
from flask import Flask, render_template
from flask_socketio import SocketIO
import threading
import eventlet

eventlet.monkey_patch()

from sender_jscc import ConfigSender, Sender

# 初始化Flask应用
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# 初始化发送端
sender = Sender(ConfigSender())


@app.route('/sender')
def show_sender():
    """显示发送端界面"""
    return render_template('sender_deepjscc.html')


def send_packet(sock, data_dict):
    """辅助函数：发送单个数据包并等待ACK"""
    # 序列化
    bytes_data = pickle.dumps(data_dict)
    data_length = len(bytes_data)

    # 记录发送开始时间
    tx_start = time.time()

    # 发送长度
    sock.sendall(data_length.to_bytes(4, byteorder='big'))
    # 发送数据
    sock.sendall(bytes_data)

    # 计算纯传输耗时 (写入缓冲区时间)
    tx_time = time.time() - tx_start

    # 等待接收端确认 (ACK)
    ack = sock.recv(4)

    return tx_time, data_length


def sender_server():
    """
    发送端服务器主循环
    """
    # 建立Socket连接
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.connect((ConfigSender.host, ConfigSender.port))
        print(f"Connected to receiver at {ConfigSender.host}:{ConfigSender.port}")
    except Exception as e:
        print(f"Failed to connect to receiver: {e}")
        return

    counter = 1

    with torch.no_grad():
        for batch_idx, input_image in enumerate(sender.test_loader):
            try:
                print(f"\n{'=' * 60}")
                print(f"Processing image {counter}")

                input_image = input_image.to(sender.config.device)
                B, C, H, W = input_image.shape

                # ============ 1. 语义编码处理时延 ============
                print(f"[{time.time():.2f}] Semantic Encoding...")
                semantic_start_time = time.time()

                # Encoder
                feature = sender.model.encoder(input_image)
                # Quantize
                feature_quantized, min_val, max_val = sender._quantize(feature)
                feature_shape = feature.shape
                # Entropy (Zlib)
                feature_bytes = feature_quantized.tobytes()
                compressed_feature = zlib.compress(feature_bytes, level=9)

                # 计算语义总处理时间
                semantic_proc_time = time.time() - semantic_start_time

                # ============ 2. JPEG 编码处理时延 ============
                print(f"[{time.time():.2f}] JPEG Encoding (Search)...")
                target_size = len(compressed_feature)

                # 准备图像数据
                image_np = input_image[0].cpu().numpy()
                image_np = (image_np * 255).clip(0, 255).astype('uint8')
                image_np = image_np.transpose(1, 2, 0)
                image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

                # 保存原始图像
                original_image_path = f"{sender.config.sent_dir}{counter}.png"
                cv2.imwrite(original_image_path, image_bgr)
                time.sleep(0.3)  # 必须保留，防止文件未写完

                # 开始计时 JPEG 处理
                jpeg_proc_start = time.time()

                low, high = 1, 100
                best_jpeg_data = None
                min_diff = float('inf')

                # 二分查找最佳质量
                while low <= high:
                    mid = (low + high) // 2
                    quality = max(1, mid)
                    temp_data = cv2.imencode('.jpg', image_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])[1].tobytes()
                    temp_size = len(temp_data)
                    diff = abs(temp_size - target_size)
                    if diff < min_diff:
                        min_diff = diff
                        best_jpeg_data = temp_data
                    if temp_size > target_size:
                        high = mid - 1
                    else:
                        low = mid + 1

                original_image_data = best_jpeg_data
                jpeg_proc_time = time.time() - jpeg_proc_start

                # === 大小统计 ===
                original_image_size = len(original_image_data) // 1024
                semantic_feature_size = len(compressed_feature) / 1024.0

                print(f"JPEG Time: {jpeg_proc_time * 1000:.2f}ms | Semantic Time: {semantic_proc_time * 1000:.2f}ms")

                # ============ 分包封装与发送 ============

                # --- 包 1: JPEG (包含编码时间) ---
                data_jpeg = {
                    'type': 'traditional',
                    'counter': counter,
                    'image': original_image_data,
                    'original_image_size': original_image_size,
                    'image_size': (H, W),
                    'encode_time': jpeg_proc_time,
                    'send_timestamp': time.time()# <--- 新增：发送端处理时延
                }

                print(f"[{time.time():.2f}] Sending Packet 1 (JPEG)...")
                tx_time_jpeg, len_jpeg = send_packet(s, data_jpeg)

                # --- 包 2: Semantic (包含编码时间) ---
                data_semantic = {
                    'type': 'semantic',
                    'counter': counter,
                    'feature': [compressed_feature, min_val, max_val, feature_shape, feature_quantized.dtype],
                    'encode_time': semantic_proc_time,
                    'send_timestamp': time.time()# <--- 新增：发送端处理时延
                }

                print(f"[{time.time():.2f}] Sending Packet 2 (Semantic)...")
                tx_time_semantic, len_semantic = send_packet(s, data_semantic)

                # ============ 推送到发送端前端 ============
                socketio.emit('message', {
                    'original_image_url': original_image_path,
                    'original_image_size': original_image_size,
                    'semantic_feature_size': round(semantic_feature_size, 2),
                    'encode_time': semantic_proc_time,
                    'encode_time_jpeg': jpeg_proc_time,
                    'tx_time_jpeg': tx_time_jpeg,
                    'tx_time_semantic': tx_time_semantic
                })

                counter += 1

            except Exception as e:
                print(f"Error processing image {counter}: {e}")
                import traceback
                traceback.print_exc()
                break

    s.close()
    print("\nTransmission completed.")


if __name__ == '__main__':
    threading.Thread(target=sender_server, daemon=True).start()
    print(f"Sender Web Interface: http://127.0.0.1:5003/sender")
    socketio.run(app, host='0.0.0.0', port=5003, debug=False)