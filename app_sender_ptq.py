"""
app_sender_ptq.py
DeepJSCC 发送端 Flask 应用 (PTQ Int8 量化版)
专门用于加载 sender_ptq.py 和 deepjscc_int8.pth
"""
import socket
import torch
import cv2
import time
import pickle
import threading
import eventlet
from flask import Flask, render_template
from flask_socketio import SocketIO

# === 关键修改：导入 PTQ 版本的配置和发送类 ===
from sender_ptq import ConfigSenderPTQ, SenderPTQ
from quant_model import QuantizableDeepJSCC
# 必须进行 monkey_patch 以支持 socketio 的异步操作
eventlet.monkey_patch()

# 初始化Flask应用
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# === 初始化 PTQ 发送端 ===
# 这会自动加载 deepjscc_int8.pth
sender = SenderPTQ(ConfigSenderPTQ())


@app.route('/sender')
def show_sender():
    """显示发送端界面 (复用原有的HTML模板)"""
    return render_template('sender_deepjscc.html')


def sender_server():
    """
    发送端服务器主循环
    处理图像编码、量化和传输
    """
    # 建立Socket连接
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        # 使用 ConfigSenderPTQ 中的配置
        print(f"Connecting to receiver at {ConfigSenderPTQ.host}:{ConfigSenderPTQ.port}...")
        s.connect((ConfigSenderPTQ.host, ConfigSenderPTQ.port))
        print(f"Connected to receiver successfully!")
    except Exception as e:
        print(f"Failed to connect to receiver: {e}")
        print("Please ensure app_receiver_ptq.py is running first.")
        return

    counter = 1

    with torch.no_grad():
        for batch_idx, input_image in enumerate(sender.test_loader):
            try:
                # ============ 编码阶段 ============
                print(f"\n{'=' * 60}")
                print(f"Processing image {counter} (PTQ Int8 Mode)")

                # 移动到设备 (ConfigSenderPTQ.device 应该是 cpu)
                input_image = input_image.to(sender.config.device)
                B, C, H, W = input_image.shape

                # 编码
                print(f"[{time.time():.2f}] JSCC Encoding image {counter}...")
                start_time = time.time()

                # === 关键：调用量化模型的 encoder ===
                # sender.model 是 QuantizableDeepJSCC 对象
                feature = sender.model.encoder(input_image)

                encode_time = time.time() - start_time
                print(f"Encoding time: {encode_time * 1000:.3f}ms")
                print(f"Feature shape: {feature.shape}")

                # 量化 (传输层面的 uint8 量化)
                print(f"[{time.time():.2f}] Quantizing features for transmission...")
                feature_quantized, min_val, max_val = sender._quantize(feature)
                feature_shape = feature.shape

                # ============ 准备传输数据 ============
                # 将tensor转换为numpy图像用于显示和JPEG对比
                image_np = input_image[0].cpu().numpy()  # (C, H, W)
                image_np = (image_np * 255).clip(0, 255).astype('uint8')
                image_np = image_np.transpose(1, 2, 0)  # (H, W, C)
                image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

                # 保存原始图像
                original_image_path = f"{sender.config.sent_dir}{counter}.png"
                cv2.imwrite(original_image_path, image_bgr)

                # 编码为JPEG用于对比
                jpeg_start_time = time.time()
                original_image_data = cv2.imencode(
                    '.jpg', image_bgr,
                    [cv2.IMWRITE_JPEG_QUALITY, 100]
                )[1].tobytes()
                jpeg_encode_time = time.time() - jpeg_start_time

                # 计算大小
                original_image_size = len(original_image_data) // 1024  # KB
                semantic_feature_size = feature_quantized.nbytes // 1024  # KB

                print(f"Original image size: {original_image_size} KB")
                print(f"Semantic feature size: {semantic_feature_size} KB")

                # ============ 封装数据包 ============
                data = {
                    'counter': counter,
                    'feature': [feature_quantized, min_val, max_val, feature_shape],
                    'image': original_image_data,
                    'original_image_size': original_image_size,
                    'image_size': (H, W),
                }

                # ============ 发送数据 ============
                print(f"[{time.time():.2f}] Sending data...")
                bytes_data = pickle.dumps(data)
                data_length = len(bytes_data)

                # 发送数据长度
                s.sendall(data_length.to_bytes(4, byteorder='big'))
                # 发送数据
                s.sendall(bytes_data)

                send_time = time.time() - start_time - encode_time
                print(f"Data size: {data_length / 1024:.2f} KB")

                # ============ 推送到前端 ============
                socketio.emit('message', {
                    'original_image_url': original_image_path,
                    'semantic_image_url': original_image_path,
                    'original_image_size': original_image_size,
                    'semantic_feature_size': semantic_feature_size,
                    'compression_ratio': round(original_image_size / max(semantic_feature_size, 1), 2),
                    'encode_time': round(encode_time, 3),
                    'encode_time_jpeg': round(jpeg_encode_time, 3),
                })

                print(f"[{time.time():.2f}] Image {counter} sent")

                # 等待接收端确认
                ack = s.recv(4)
                if ack:
                    print(f"Received ACK: {int.from_bytes(ack, byteorder='big')}")

                counter += 1
                time.sleep(0.5)

            except Exception as e:
                print(f"Error processing image {counter}: {e}")
                import traceback
                traceback.print_exc()
                break

    s.close()
    print("\nTransmission completed.")


if __name__ == '__main__':
    # 启动发送端服务器线程
    threading.Thread(target=sender_server, daemon=True).start()

    print("\nStarting DeepJSCC Sender (PTQ Version)...")
    print(f"Device: {ConfigSenderPTQ.device}")
    print(f"Target Receiver: {ConfigSenderPTQ.host}:{ConfigSenderPTQ.port}")
    print(f"Web interface: http://127.0.0.1:5003/sender")

    # 注意：端口保持 5003，与原版一致，请不要同时运行原版 app_sender_deepjscc
    socketio.run(app, host='0.0.0.0', port=5003, debug=False)