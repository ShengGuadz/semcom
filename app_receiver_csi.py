# -*- coding: utf-8 -*-
"""
DeepJSCC-CSI 接收端 Flask 应用
提供 Web 界面、Socket 服务器和 CSI-aware 图像解码功能
"""

import socket
import time
import pickle
import cv2
import numpy as np
from flask import Flask, render_template
from flask_socketio import SocketIO
import threading
import eventlet
import webbrowser

eventlet.monkey_patch()

from receiver_csi import ConfigReceiverCSI, ReceiverCSI
from receiver_jscc import calculate_psnr, calculate_ssim  # 直接复用原有指标函数:contentReference[oaicite:2]{index=2}

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

receiver = ReceiverCSI(ConfigReceiverCSI())


@app.route('/receiver_csi')
def show_receiver_csi():
    """显示 JSCC-CSI 接收端界面"""
    # 同样可以复用 receiver_deepjscc.html
    return render_template('receiver_deepjscc.html')


def receiver_csi_server():
    """JSCC-CSI 接收端 Socket 服务器主循环"""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((ConfigReceiverCSI.host, ConfigReceiverCSI.port))
    s.listen(5)

    print(f"[Receiver-CSI] Listening on {ConfigReceiverCSI.host}:{ConfigReceiverCSI.port}")
    conn, addr = s.accept()
    print(f"[Receiver-CSI] Connected by {addr}")

    counter = 1
    traditional_psnr_list, semantic_psnr_list = [], []
    traditional_ssim_list, semantic_ssim_list = [], []
    # 新增：大小统计（单位：KB）
    traditional_kb_list, semantic_kb_list, packet_kb_list = [], [], []

    while True:
        try:
            data_length_bytes = conn.recv(4)
            if not data_length_bytes:
                print("[Receiver-CSI] Connection closed by sender")
                break
            data_length = int.from_bytes(data_length_bytes, byteorder='big')
            # 整个 socket 包的大小（含 JPEG + 语义特征）
            packet_kb_list.append(data_length / 1024.0)

            received = bytearray()
            while len(received) < data_length:
                chunk = conn.recv(min(4096, data_length - len(received)))
                if not chunk:
                    break
                received.extend(chunk)

            data = pickle.loads(received)
            counter = data['counter']
            print(f"\n{'=' * 60}")
            print(f"[Receiver-CSI] Decoding image {counter}")

            # --------- 语义解码（CSI-aware）---------
            sem_start = time.time()
            semantic_image, original_image, feature_quantized, traditional_image_size = \
                receiver.decode(data)
            decode_time = time.time() - sem_start
            print(f"[Receiver-CSI] Semantic decoding time: {decode_time * 1000:.3f} ms")

            # --------- 传统 JPEG 链路 （带误码模拟）---------
            jpeg_data = data['image']
            corrupted_jpeg_data = receiver._simulate_bit_errors(
                jpeg_data, ConfigReceiverCSI.snr
            )
            traditional_image = cv2.imdecode(
                np.frombuffer(corrupted_jpeg_data, np.uint8),
                cv2.IMREAD_COLOR
            )
            if traditional_image is None:
                print("[Receiver-CSI] JPEG decode failed, fallback to pixel-domain AWGN")
                traditional_image = receiver._add_gaussian_noise(original_image, ConfigReceiverCSI.snr)

            # --------- 保存图像 ---------
            semantic_image_path = f"{receiver.config.save_dir_semantic}{counter}.png"
            traditional_image_path = f"{receiver.config.save_dir_traditional}{counter}.png"
            cv2.imwrite(semantic_image_path, semantic_image)
            cv2.imwrite(traditional_image_path, traditional_image)

            # --------- 指标计算 ---------
            traditional_psnr = calculate_psnr(original_image, traditional_image)
            semantic_psnr = calculate_psnr(original_image, semantic_image)
            try:
                traditional_ssim = calculate_ssim(original_image, traditional_image)
                semantic_ssim = calculate_ssim(original_image, semantic_image)
            except Exception as e:
                print(f"[Receiver-CSI] SSIM calculation failed: {e}")
                traditional_ssim = 0.0
                semantic_ssim = 0.0

            traditional_psnr_list.append(traditional_psnr)
            semantic_psnr_list.append(semantic_psnr)
            traditional_ssim_list.append(traditional_ssim)
            semantic_ssim_list.append(semantic_ssim)

            print(f"[Receiver-CSI] Traditional - PSNR: {traditional_psnr:.2f} dB, SSIM: {traditional_ssim:.4f}")
            print(f"[Receiver-CSI] Semantic   - PSNR: {semantic_psnr:.2f} dB, SSIM: {semantic_ssim:.4f}")

            original_image_size = data['original_image_size']
            semantic_feature_size = feature_quantized.nbytes // 1024  # KB
            # JPEG 压缩后的大小（KB）：traditional_image_size 是 decode() 返回的值
            traditional_kb_list.append(float(traditional_image_size))
            # 语义特征比特流大小（KB）
            semantic_kb_list.append(float(semantic_feature_size))


            traditional_compression_ratio = round(
                (original_image_size - traditional_image_size) * 100 / original_image_size, 2
            )
            semantic_compression_ratio = round(
                (original_image_size - semantic_feature_size) * 100 / original_image_size, 2
            )

            bandwidth_kbps = 102400
            traditional_throughput = round(bandwidth_kbps / traditional_image_size, 2)
            semantic_throughput = round(bandwidth_kbps / max(semantic_feature_size, 1), 2)

            socketio.emit('message', {
                'traditional_image_url': traditional_image_path,
                'semantic_image_url': semantic_image_path,
                'traditional_image_size': traditional_image_size,
                'semantic_feature_size': semantic_feature_size,
                'traditional_psnr': round(traditional_psnr, 2),
                'semantic_psnr': round(semantic_psnr, 2),
                'traditional_ssim': round(traditional_ssim, 4),
                'semantic_ssim': round(semantic_ssim, 4),
                'traditional_compression_ratio': traditional_compression_ratio,
                'semantic_compression_ratio': semantic_compression_ratio,
                'traditional_throughput': traditional_throughput,
                'semantic_throughput': semantic_throughput,
            })

            # 回 ACK
            ack = (1).to_bytes(4, byteorder='big')
            conn.sendall(ack)
            counter += 1

        except Exception as e:
            print(f"[Receiver-CSI] Error processing image: {e}")
            import traceback
            traceback.print_exc()
            break

    s.close()
    print("[Receiver-CSI] Server closed.")
    # ===== 平均指标统计 =====
    if traditional_psnr_list:  # 防止空列表
        n = len(traditional_psnr_list)

        avg_trad_psnr = sum(traditional_psnr_list) / n
        avg_sem_psnr  = sum(semantic_psnr_list) / n
        avg_trad_ssim = sum(traditional_ssim_list) / n
        avg_sem_ssim  = sum(semantic_ssim_list) / n

        avg_trad_kb   = sum(traditional_kb_list) / n if traditional_kb_list else 0.0
        avg_sem_kb    = sum(semantic_kb_list) / n if semantic_kb_list else 0.0
        avg_packet_kb = sum(packet_kb_list) / n if packet_kb_list else 0.0

        print("\n" + "=" * 60)
        print(f"[Receiver-CSI] AVERAGE METRICS OVER {n} IMAGES")
        print(f"Traditional JPEG  : PSNR {avg_trad_psnr:.2f} dB, "
              f"SSIM {avg_trad_ssim:.4f}, size {avg_trad_kb:.2f} KB")
        print(f"Semantic (CSI-JSCC): PSNR {avg_sem_psnr:.2f} dB, "
              f"SSIM {avg_sem_ssim:.4f}, feature size {avg_sem_kb:.2f} KB")
        print(f"Socket packet (JPEG+feature) avg size: {avg_packet_kb:.2f} KB")

        if avg_trad_kb > 0 and avg_sem_kb > 0:
            saving = (1.0 - avg_sem_kb / avg_trad_kb) * 100.0
            print(f"Semantic vs JPEG payload saving: {saving:.2f}%")

    s.close()
    print("[Receiver-CSI] Server closed.")



if __name__ == '__main__':
    threading.Thread(target=receiver_csi_server, daemon=True).start()
    # Web 前端地址
    url = "http://127.0.0.1:63000/receiver_csi"

    # 终端提示信息
    print("\nStarting DeepJSCC-CSI Receiver...")
    print(f"Web interface: {url}")
    print("Press CTRL+C to stop.")

    # 如需自动打开浏览器，可以取消下一行注释
    # webbrowser.open(url)
    socketio.run(app, host='0.0.0.0', port=63000, debug=False)
