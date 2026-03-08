# -*- coding: utf-8 -*-
"""
DeepJSCC-CSI 发送端 Flask 应用
提供 Web 界面和 Socket 通信功能
"""

import socket
import torch
import cv2
import time
import pickle
from flask import Flask, render_template
from flask_socketio import SocketIO
import threading
import eventlet
import webbrowser

eventlet.monkey_patch()

from sender_csi import ConfigSenderCSI, SenderCSI

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# 初始化发送端
sender = SenderCSI(ConfigSenderCSI())


@app.route('/sender_csi')
def show_sender_csi():
    """显示 JSCC-CSI 发送端界面"""
    # 可以直接复用 sender_deepjscc.html 模板，也可以单独做 sender_csi.html
    return render_template('sender_deepjscc.html')


def sender_csi_server():
    """
    JSCC-CSI 发送端服务器主循环：
    逐张图像编码、量化并通过 Socket 发送
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.connect((ConfigSenderCSI.host, ConfigSenderCSI.port))
        print(f"[Sender-CSI] Connected to receiver at {ConfigSenderCSI.host}:{ConfigSenderCSI.port}")
    except Exception as e:
        print(f"[Sender-CSI] Failed to connect to receiver: {e}")
        return

    counter = 1

    with torch.no_grad():
        for _, input_image in enumerate(sender.test_loader):
            try:
                print(f"\n{'=' * 60}")
                print(f"[Sender-CSI] Processing image {counter}")

                input_image = input_image.to(sender.config.device)
                start_time = time.time()

                # 编码 + 量化
                feature_quantized, min_val, max_val, feature_shape, image_bgr = \
                    sender.encode_one_image(input_image, num_bits=8)
                encode_time = time.time() - start_time
                print(f"[Sender-CSI] Encoding+quantization time: {encode_time * 1000:.3f} ms")

                # 保存原始图像
                original_image_path = f"{sender.config.sent_dir}{counter}.png"
                cv2.imwrite(original_image_path, image_bgr)

                # JPEG 编码（用于传统链路对比）
                jpeg_start_time = time.time()
                original_image_data = cv2.imencode(
                    '.jpg', image_bgr,
                    [cv2.IMWRITE_JPEG_QUALITY, 100]
                )[1].tobytes()
                jpeg_encode_time = time.time() - jpeg_start_time

                original_image_size = len(original_image_data) // 1024  # KB
                semantic_feature_size = feature_quantized.nbytes // 1024  # KB

                print(f"[Sender-CSI] Original image size: {original_image_size} KB")
                print(f"[Sender-CSI] Semantic feature size: {semantic_feature_size} KB")

                # 封装数据包
                data = {
                    'counter': counter,
                    'feature': [feature_quantized, min_val, max_val, feature_shape],
                    'image': original_image_data,
                    'original_image_size': original_image_size,
                    'image_size': image_bgr.shape[:2],
                }

                bytes_data = pickle.dumps(data)
                data_length = len(bytes_data)

                # 发送长度 + 数据
                s.sendall(data_length.to_bytes(4, byteorder='big'))
                s.sendall(bytes_data)

                send_time = time.time() - start_time - encode_time
                print(f"[Sender-CSI] Transmission time: {send_time:.3f} s")
                print(f"[Sender-CSI] Data size: {data_length / 1024:.2f} KB")

                socketio.emit('message', {
                    'original_image_url': original_image_path,
                    'semantic_image_url': original_image_path,
                    'original_image_size': original_image_size,
                    'semantic_feature_size': semantic_feature_size,
                    'compression_ratio': round(
                        original_image_size / max(semantic_feature_size, 1), 2),
                    'encode_time': round(encode_time, 3),
                    'encode_time_jpeg': round(jpeg_encode_time, 3),
                })

                # 等待接收端 ACK
                ack = s.recv(4)
                if ack:
                    print(f"[Sender-CSI] Received ACK: {int.from_bytes(ack, byteorder='big')}")

                counter += 1
                time.sleep(0.5)

            except Exception as e:
                print(f"[Sender-CSI] Error processing image {counter}: {e}")
                import traceback
                traceback.print_exc()
                break

    s.close()
    print("[Sender-CSI] Transmission completed.")


if __name__ == '__main__':
    threading.Thread(target=sender_csi_server, daemon=True).start()
    # Web 前端地址（和 Flask 端口、路由保持一致）
    url = "http://127.0.0.1:62000/sender_csi"

    # 终端提示信息，便于直接点击
    print("\nStarting DeepJSCC-CSI Sender...")
    print(f"Web interface: {url}")
    print("Press CTRL+C to stop.")

    # 如需自动打开浏览器，可以取消下一行注释
    # webbrowser.open(url)
    # 这里可以自己打开浏览器访问 /sender_csi
    socketio.run(app, host='0.0.0.0', port=62000, debug=False)
