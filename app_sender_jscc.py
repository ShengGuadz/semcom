"""
DeepJSCC 发送端 Flask 应用
提供Web界面和Socket通信功能
"""
import socket
import torch
import cv2
import time
import pickle
import webbrowser
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


def sender_server():
    """
    发送端服务器主循环
    处理图像编码、量化和传输
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
    # === 1. 初始化统计变量 ===
    total_semantic_time = 0.0  # 语义编码总耗时
    total_jpeg_time = 0.0  # JPEG编码总耗时
    stats_count = 0  # 成功处理的图片计数
    # =======================
    with torch.no_grad():
        for batch_idx, input_image in enumerate(sender.test_loader):
            try:
                # ============ 编码阶段 ============
                print(f"\n{'='*60}")
                print(f"Processing image {counter}")
                # start_time = time.time()

                # 移动到设备
                input_image = input_image.to(sender.config.device)
                B, C, H, W = input_image.shape
                
                # 编码
                print(f"[{time.time():.2f}] JSCC Encoding image {counter}...")
                start_time = time.time()
                feature = sender.model.encoder(input_image)
                encode_time = time.time() - start_time
                # === 累加语义编码耗时 ===
                total_semantic_time += encode_time
                # ======================
                print(f"Encoding time: {encode_time*1000:.3f}ms")
                # encode_time = encode_time * 1000
                # print(f"Encoding time: {encode_time:.3f}ms")
                print(f"Feature shape: {feature.shape}")

                # 量化
                print(f"[{time.time():.2f}] Quantizing features...")
                feature_quantized, min_val, max_val = sender._quantize(feature)
                feature_shape = feature.shape
                
                # ============ 准备传输数据 ============
                # 将tensor转换为numpy图像
                image_np = input_image[0].cpu().numpy()  # (C, H, W)
                image_np = (image_np * 255).clip(0, 255).astype('uint8')
                image_np = image_np.transpose(1, 2, 0)  # (H, W, C)
                image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
                
                # 保存原始图像
                original_image_path = f"{sender.config.sent_dir}{counter}.png"
                cv2.imwrite(original_image_path, image_bgr)#1.5
                
                # 编码为JPEG用于对比
                jpeg_start_time = time.time()
                original_image_data = cv2.imencode(
                    '.jpg', image_bgr,
                    [cv2.IMWRITE_JPEG_QUALITY, 100]
                )[1].tobytes()
                jpeg_encode_time = time.time() - jpeg_start_time
                # === 累加JPEG编码耗时 ===
                total_jpeg_time += jpeg_encode_time
                stats_count += 1  # 计数加1
                # ======================
                print(f"Encoding time (JPEG): {jpeg_encode_time*1000:.3f}ms")
                
                # 计算大小
                original_image_size = len(original_image_data) // 1024  # KB
                semantic_feature_size = feature_quantized.nbytes // 1024  # KB
                
                print(f"Original image size: {original_image_size} KB")
                print(f"Semantic feature size: {semantic_feature_size} KB")
                print(f"Compression ratio: {original_image_size / max(semantic_feature_size, 1):.2f}×")
                
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
                print(f"Transmission time: {send_time:.3f}s")



                
                # print(f"Transmission time: {send_time * 1000:.3f}ms")
                print(f"Data size: {data_length / 1024:.2f} KB")
                
                # ============ 推送到前端 ============
                socketio.emit('message', {
                    'original_image_url': original_image_path,
                    'semantic_image_url': original_image_path,
                    'original_image_size': original_image_size,
                    'semantic_feature_size': semantic_feature_size,
                    'compression_ratio': round(original_image_size / max(semantic_feature_size, 1), 2),
                    'encode_time': encode_time,
                    'encode_time_jpeg': jpeg_encode_time,
                })
                
                print(f"[{time.time():.2f}] Image {counter} sent and displayed")

                # 等待接收端确认
                ack = s.recv(4)
                if ack:
                    print(f"Received ACK: {int.from_bytes(ack, byteorder='big')}")

                counter += 1
                
                # 延时（可选）
                # time.sleep(0.5)
                
            except Exception as e:
                print(f"Error processing image {counter}: {e}")
                import traceback
                traceback.print_exc()
                break
        # === 3. 循环结束后计算并输出平均时延 ===
    print(f"\n{'=' * 25} RESULT {'=' * 25}")
    if stats_count > 0:
            avg_semantic = (total_semantic_time / stats_count) * 1000  # 转换为ms
            avg_jpeg = (total_jpeg_time / stats_count) * 1000  # 转换为ms

            print(f"Total Images: {stats_count}")
            print(f"Average Semantic Encoding Time: {avg_semantic:.4f} ms")
            print(f"Average JPEG Encoding Time:     {avg_jpeg:.4f} ms")
    else:
            print("No images processed successfully.")
    print(f"{'=' * 58}\n")
        # ====================================

    s.close()
    print("\nTransmission completed.")


if __name__ == '__main__':
    # 启动发送端服务器线程
    threading.Thread(target=sender_server, daemon=True).start()
    
    # 打开浏览器
    # webbrowser.open("http://127.0.0.1:5003/sender")
    
    # 启动Flask应用
    print("\nStarting DeepJSCC Sender...")
    print(f"Device: {ConfigSender.device}")
    print(f"Model: c={ConfigSender.c}, SNR={ConfigSender.snr}dB")
    print(f"Web interface: http://127.0.0.1:5003/sender")
    print(f"Connecting to receiver at {ConfigSender.host}:{ConfigSender.port}")
    
    socketio.run(app, host='0.0.0.0', port=5003, debug=False)
