"""
app_sender_hailo.py
适配 Hailo 硬件加速的发送端主程序 (完整版: 含统计与打印)
"""
import eventlet

eventlet.monkey_patch()

import socket
import time
import pickle
import cv2
import threading
import torch
from flask import Flask, render_template
from flask_socketio import SocketIO

# 导入逻辑类 (确保 sender_hailo.py 和 hailo_infer.py 在同一目录)
from sender_hailo import ConfigSender, Sender

app = Flask(__name__)
# 禁用 Flask 默认的 HTTP 日志，保持控制台清爽
import logging

log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

socketio = SocketIO(app, cors_allowed_origins="*")

# 初始化发送端逻辑核心
sender = Sender(ConfigSender())


@app.route('/sender')
def show_sender():
    return render_template('sender_deepjscc.html')


def sender_server():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        print(f"Attempting to connect to receiver at {ConfigSender.host}:{ConfigSender.port}...")
        s.connect((ConfigSender.host, ConfigSender.port))
        print(f"Successfully connected to receiver!")
    except Exception as e:
        print(f"Connection failed: {e}")
        print("Please ensure the Receiver is running FIRST.")
        return

    counter = 1
    # === 统计变量初始化 ===
    total_sem_time = 0.0
    total_jpeg_time = 0.0
    stats_count = 0
    # ====================

    print("\nStarting transmission loop...")

    # 遍历加载器中的每一张图片
    for i, image_tensor in enumerate(sender.test_loader):
        try:
            print(f"\n{'=' * 40}\nProcessing Image {counter}")

            # --- 1. Hailo 语义编码 (计时) ---
            print(f"[{time.time():.2f}] JSCC Encoding image {counter} (Hailo)...")
            t0 = time.time()
            # 调用 sender_hailo.py 中的 encode
            feature_q, min_v, max_v, rotated, img_processed = sender.encode(image_tensor)
            enc_time = time.time() - t0

            total_sem_time += enc_time
            print(f"Encoding time: {enc_time * 1000:.3f} ms")
            print(f"Feature shape: {feature_q.shape}")

            # --- 2. JPEG 传统编码 (计时 & 对比用) ---
            # img_processed 是 float32 [0,1], 转回 uint8 [0,255] 用于显示和JPEG编码
            img_disp = (img_processed * 255).clip(0, 255).astype('uint8')
            img_disp = cv2.cvtColor(img_disp, cv2.COLOR_RGB2BGR)

            # 保存本地用于网页显示
            save_path = f"{sender.config.sent_dir}{counter}.png"
            cv2.imwrite(save_path, img_disp)

            # 执行 JPEG 编码
            t_jpg_start = time.time()
            jpeg_data = cv2.imencode('.jpg', img_disp, [cv2.IMWRITE_JPEG_QUALITY, 100])[1].tobytes()
            t_jpg_end = time.time()
            jpeg_enc_time = t_jpg_end - t_jpg_start

            total_jpeg_time += jpeg_enc_time
            stats_count += 1

            # --- [新增] 模仿原 Torch 代码的打印逻辑 ---
            original_image_size = len(jpeg_data) // 1024  # KB
            semantic_feature_size = feature_q.nbytes / 1024  # KB
            compression_ratio_x = original_image_size / max(semantic_feature_size, 1)  # 倍数

            print(f"Encoding time (JPEG): {jpeg_enc_time * 1000:.3f} ms")
            print(f"Original image size: {original_image_size} KB")
            print(f"Quantized feature size: {semantic_feature_size:.2f} KB")
            print(f"Compression ratio: {compression_ratio_x:.2f}x")
            # ------------------------------------------

            # --- 3. 封装数据包 ---
            # 协议格式必须与接收端 receiver_hailo.py 严格匹配
            data_map = {
                'counter': counter,
                'feature': [feature_q, min_v, max_v, feature_q.shape],
                'image': jpeg_data,
                'rotated': rotated,
                'original_image_size': original_image_size,
                'shape': feature_q.shape,
                'size': (img_processed.shape[0], img_processed.shape[1])
            }

            # --- 4. 发送数据 ---
            print(f"[{time.time():.2f}] Sending data...")
            bytes_data = pickle.dumps(data_map)
            length = len(bytes_data)

            # 先发长度
            s.sendall(length.to_bytes(4, 'big'))
            # 再发数据体
            s.sendall(bytes_data)

            print(f"Sent Packet Size: {length / 1024:.2f} KB | Rotated: {rotated}")

            # --- 5. 推送状态给前端 ---
            # 包含语义编码时间和JPEG时间，供前端展示对比
            socketio.emit('message', {
                'original_image_url': save_path,
                'encode_time': round(enc_time, 3),  # 语义耗时 (s)
                'encode_time_jpeg': round(jpeg_enc_time, 3),  # JPEG耗时 (s)
                'compression_ratio': round(compression_ratio_x, 2)
            })

            # --- 6. 等待确认 (ACK) ---
            ack = s.recv(4)
            if not ack:
                print("Receiver closed connection.")
                break
            # print(f"Received ACK: {int.from_bytes(ack, byteorder='big')}")

            counter += 1

        except Exception as e:
            print(f"Error during processing: {e}")
            import traceback
            traceback.print_exc()
            break

    # === 7. 最终统计报告 ===
    print(f"\n{'=' * 25} FINAL STATISTICS {'=' * 25}")
    if stats_count > 0:
        avg_semantic = (total_sem_time / stats_count) * 1000  # ms
        avg_jpeg = (total_jpeg_time / stats_count) * 1000  # ms

        print(f"Total Images Processed:       {stats_count}")
        print(f"Avg Semantic Encoding Time:   {avg_semantic:.4f} ms")
        print(f"Avg JPEG Encoding Time:       {avg_jpeg:.4f} ms")

        if avg_jpeg > 0:
            speedup = avg_jpeg / avg_semantic
            print(f"Speed Comparison (JPEG/Sem):  {speedup:.2f}x")
    else:
        print("No images processed successfully.")
    print(f"{'=' * 68}\n")
    # =====================

    s.close()
    sender.close()  # 释放 Hailo 资源
    print("Transmission completed. Hailo resources released.")


if __name__ == '__main__':
    # 启动后台发送线程
    threading.Thread(target=sender_server, daemon=True).start()

    # 打印访问地址
    print("\n" + "=" * 50)
    print("  DeepJSCC Sender (Hailo Edition) Running")
    print(f"  > Control Panel: http://{ConfigSender.host}:5003/sender")
    print(f"  > Local Access:  http://127.0.0.1:5003/sender")
    print("=" * 50 + "\n")

    # 启动 Web 服务
    socketio.run(app, host='0.0.0.0', port=5003, debug=False, use_reloader=False)