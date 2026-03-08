import socket
import torch
import cv2
import time
import pickle
import os
import numpy as np
from flask import Flask
from flask_socketio import SocketIO
import threading
import eventlet

eventlet.monkey_patch()

from pyldpc import make_ldpc, encode
from sender_deepjscc import ConfigSender, Sender

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")
sender_instance = Sender(ConfigSender())

# LDPC 参数 (1/2 码率: n=1200, d_v=2, d_c=4)
N, D_V, D_C = 1200, 2, 4
H, G = make_ldpc(N, D_V, D_C, systematic=True, sparse=True)


def qam16_modulate(bits):
    """16QAM 调制实现"""
    bit_groups = bits.reshape(-1, 4)
    mapping = {tuple(map(int, bin(i)[2:].zfill(4))): complex((i % 4) * 2 - 3, (i // 4) * 2 - 3) for i in range(16)}
    return np.array([mapping[tuple(group)] for group in bit_groups])


def send_udp_segments(sock, data_dict, dest):
    """分片发送逻辑，防止 UDP 超过 MTU 并加入延迟"""
    raw_data = pickle.dumps(data_dict)
    segment_size = 8192  # 减小分片大小以适配网络
    total_parts = (len(raw_data) - 1) // segment_size + 1

    for i in range(total_parts):
        chunk = raw_data[i * segment_size: (i + 1) * segment_size]
        header = {'idx': data_dict['counter'], 'part': i, 'total': total_parts}
        sock.sendto(pickle.dumps({'header': header, 'payload': chunk}), dest)
        time.sleep(0.001)  # 分片间微小延迟，防止丢包


def sender_server():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    dest = (ConfigSender.host, ConfigSender.port)
    counter = 1

    # 获取数据集路径列表
    image_paths = sender_instance.test_loader.dataset.image_paths

    with torch.no_grad():
        for i, input_image in enumerate(sender_instance.test_loader):
            try:
                filename = os.path.basename(image_paths[i])  #
                input_image = input_image.to(ConfigSender.device)

                # 1. 语义路径编码及计时
                start_sem = time.time()
                feature = sender_instance.model.encoder(input_image)
                f_quant, f_min, f_max = sender_instance._quantize(feature)
                sem_enc_time = (time.time() - start_sem) * 1000  # ms

                # 2. 传统路径处理及计时
                img_np = (input_image[0].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

                start_jpeg = time.time()
                _, jpeg_bytes = cv2.imencode('.jpg', img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 50])
                jpeg_enc_time = (time.time() - start_jpeg) * 1000  # ms

                # LDPC 编码
                bits = np.unpackbits(np.frombuffer(jpeg_bytes, dtype=np.uint8))
                pad_len = (N // 2) - (len(bits) % (N // 2))
                bits_padded = np.append(bits, np.zeros(pad_len, dtype=np.uint8))
                encoded_bits = encode(G, bits_padded.reshape(-1, N // 2), snr=100).flatten()
                qam_syms = qam16_modulate(encoded_bits)

                # 3. 封装包 (不再发送 original_bgr)
                packet = {
                    'counter': counter,
                    'filename': filename,  # 传文件名
                    'semantic': [f_quant, f_min, f_max, feature.shape],
                    'traditional_qam': qam_syms,
                    'sem_enc_time': round(sem_enc_time, 3),  #
                    'jpeg_enc_time': round(jpeg_enc_time, 3),  #
                    'jpeg_size': len(jpeg_bytes) / 1024.0
                }

                send_udp_segments(s, packet, dest)
                print(f"Sent Image {counter}: {filename}. Sem: {sem_enc_time:.2f}ms, JPEG: {jpeg_enc_time:.2f}ms")

                socketio.emit('message', {
                    'idx': counter,
                    'sem_time': round(sem_enc_time, 3),
                    'jpeg_time': round(jpeg_enc_time, 3)
                })

                counter += 1
                time.sleep(1.0)  # 帧间间隔
            except Exception as e:
                print(f"Sender Error: {e}")
                break


if __name__ == '__main__':
    threading.Thread(target=sender_server, daemon=True).start()
    socketio.run(app, host='0.0.0.0', port=5003, debug=False)