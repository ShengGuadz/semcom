"""
DeepJSCC 发送端 - UDP 版本
实现应用层分包 (Application-Level Fragmentation)
"""
import socket
import torch
import cv2
import time
import pickle
import zlib
import numpy as np
import struct
import math
from flask import Flask, render_template
from flask_socketio import SocketIO
import threading
import eventlet

eventlet.monkey_patch()

from sender_jscc import ConfigSender, Sender

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# 初始化发送端
sender = Sender(ConfigSender())

# === UDP 配置 ===
# 使用 1400 字节作为 payload 大小，留出空间给 IP/UDP 头，避免 IP 层分片
UDP_PAYLOAD_SIZE = 1400
TARGET_IP = ConfigSender.host
TARGET_PORT = ConfigSender.port


@app.route('/sender')
def show_sender():
    return render_template('sender_deepjscc.html')


def send_udp_packet(sock, data_bytes, image_id):
    """
    将大数据切片并通过 UDP 发送
    协议头 (12 bytes): [Image_ID (4B), Seq_ID (4B), Total_Packets (4B)]
    """
    total_len = len(data_bytes)
    total_packets = math.ceil(total_len / UDP_PAYLOAD_SIZE)

    print(f"UDP Sending: {total_len / 1024:.2f} KB in {total_packets} packets")

    for i in range(total_packets):
        start = i * UDP_PAYLOAD_SIZE
        end = min(start + UDP_PAYLOAD_SIZE, total_len)
        chunk = data_bytes[start:end]

        # 打包头部 (!III 表示 3个无符号整数，大端序)
        header = struct.pack('!III', image_id, i, total_packets)
        packet = header + chunk

        sock.sendto(packet, (TARGET_IP, TARGET_PORT))

        # === 关键流控 ===
        # UDP发太快会把缓冲区撑爆导致本地丢包，必须微小延时
        if i % 10 == 0:
            time.sleep(0.001)


def sender_server():
    # 改为 SOCK_DGRAM (UDP)
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    # 增加发送缓冲区大小，防止本地丢包
    s.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1024 * 1024)

    counter = 1

    # 统计
    total_semantic_time = 0.0
    total_jpeg_time = 0.0
    total_entropy_time = 0.0
    stats_count = 0

    with torch.no_grad():
        for batch_idx, input_image in enumerate(sender.test_loader):
            try:
                print(f"\n{'=' * 60}")
                print(f"Processing image {counter}")

                input_image = input_image.to(sender.config.device)
                B, C, H, W = input_image.shape

                # 1. 语义编码
                start_time = time.time()
                feature = sender.model.encoder(input_image)
                encode_time = time.time() - start_time
                total_semantic_time += encode_time

                # 2. 量化
                feature_quantized, min_val, max_val = sender._quantize(feature)
                feature_shape = feature.shape

                # 3. 熵编码 (Zlib)
                entropy_start = time.time()
                if isinstance(feature_quantized, torch.Tensor):
                    # 如果是 Tensor，转为 torch.uint8
                    if feature_quantized.dtype != torch.uint8:
                        # 确保数据在 0-255 之间 (量化函数通常已经做到了，这里是双保险)
                        feature_quantized = feature_quantized.to(torch.uint8)
                    feature_np = feature_quantized.cpu().numpy()
                else:
                    # 如果是 Numpy，转为 np.uint8
                    feature_np = feature_quantized.astype(np.uint8)

                dtype_str = str(feature_np.dtype)
                feature_bytes = feature_np.tobytes()
                compressed_feature = zlib.compress(feature_bytes, level=9)

                entropy_time = time.time() - entropy_start
                total_entropy_time += entropy_time

                # 4. 动态 JPEG (匹配大小)
                target_size = len(compressed_feature)
                image_np = input_image[0].cpu().numpy()
                image_np = (image_np * 255).clip(0, 255).astype('uint8').transpose(1, 2, 0)
                image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

                # 保存用于Web显示
                original_image_path = f"{sender.config.sent_dir}{counter}.png"
                cv2.imwrite(original_image_path, image_bgr)

                # 二分查找匹配 JPEG 大小
                jpeg_start_time = time.time()
                low, high = 1, 100
                best_jpeg_data = None
                min_diff = float('inf')

                while low <= high:
                    mid = (low + high) // 2
                    quality = max(1, mid)
                    temp_data = cv2.imencode('.jpg', image_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])[1].tobytes()
                    if abs(len(temp_data) - target_size) < min_diff:
                        min_diff = abs(len(temp_data) - target_size)
                        best_jpeg_data = temp_data

                    if len(temp_data) > target_size:
                        high = mid - 1
                    else:
                        low = mid + 1

                original_image_data = best_jpeg_data
                jpeg_encode_time = time.time() - jpeg_start_time
                total_jpeg_time += jpeg_encode_time
                stats_count += 1

                # 5. 封装数据包
                original_image_size = len(original_image_data) // 1024
                semantic_feature_size = len(compressed_feature) / 1024.0

                data = {
                    'counter': counter,
                    'feature': [compressed_feature, min_val, max_val, feature_shape, dtype_str],
                    'image': original_image_data,  # 这里的 JPEG 也会被拆包发送
                    'original_image_size': original_image_size,
                    'image_size': (H, W),
                }

                # 6. 序列化并 UDP 发送
                print(f"[{time.time():.2f}] Serializing & Sending via UDP...")
                bytes_data = pickle.dumps(data)

                send_start = time.time()
                send_udp_packet(s, bytes_data, counter)
                send_time = time.time() - send_start

                # 推送前端
                socketio.emit('message', {
                    'original_image_url': original_image_path,
                    'semantic_image_url': original_image_path,
                    'original_image_size': original_image_size,
                    'semantic_feature_size': round(semantic_feature_size, 2),
                    'compression_ratio': round(original_image_size / max(semantic_feature_size, 0.1), 2),
                    'encode_time': encode_time + entropy_time,
                    'encode_time_jpeg': jpeg_encode_time,
                })

                print(f"Sent Image {counter}. Waiting...")
                # time.sleep(0.5)  # 控制发送速率，模拟视频流间隔
                counter += 1

            except Exception as e:
                print(f"Error: {e}")
                import traceback
                traceback.print_exc()
                break

    s.close()


if __name__ == '__main__':
    threading.Thread(target=sender_server, daemon=True).start()
    print("\nStarting UDP Sender...")
    print(f"Target: {ConfigSender.host}:{ConfigSender.port}")
    print(f"Web interface: http://127.0.0.1:5003/sender")
    socketio.run(app, host='0.0.0.0', port=5003, debug=False)