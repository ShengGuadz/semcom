import socket
import pickle
import cv2
import numpy as np
import os
import torch
from flask import Flask, render_template
from flask_socketio import SocketIO
import threading
import eventlet

eventlet.monkey_patch()

from pyldpc import make_ldpc
from receiver_deepjscc import ConfigReceiver, Receiver, calculate_psnr, calculate_ssim

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")
receiver_instance = Receiver(ConfigReceiver())

# LDPC 参数
N, D_V, D_C = 1200, 2, 4
H, _ = make_ldpc(N, D_V, D_C, systematic=True, sparse=True)


def qam16_demodulate_awgn(symbols, snr_db):
    """QAM 解调并模拟噪声"""
    sig_pwr = np.mean(np.abs(symbols) ** 2)
    noise_pwr = sig_pwr / (10 ** (snr_db / 10))
    noise = np.sqrt(noise_pwr / 2) * (
                np.random.standard_normal(symbols.shape) + 1j * np.random.standard_normal(symbols.shape))
    received = symbols + noise

    bits = []
    for s in received:
        r, i = s.real, s.imag
        bits.extend([1 if r > 0 else 0, 1 if abs(r) < 2 else 0, 1 if i > 0 else 0, 1 if abs(i) < 2 else 0])
    return np.array(bits)


def receiver_server():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    # 显著增大接收缓冲区，解决“半天没结果”的问题
    s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 16 * 1024 * 1024)
    s.bind((ConfigReceiver.host, ConfigReceiver.port))
    print(f"Receiver listening on {ConfigReceiver.host}:{ConfigReceiver.port}")

    buffer = {}

    while True:
        try:
            data, addr = s.recvfrom(65535)
            msg = pickle.loads(data)
            hdr = msg['header']
            img_id = hdr['idx']

            if img_id not in buffer: buffer[img_id] = [None] * hdr['total']
            buffer[img_id][hdr['part']] = msg['payload']

            if all(x is not None for x in buffer[img_id]):
                packet = pickle.loads(b"".join(buffer[img_id]))
                del buffer[img_id]

                # 1. 根据文件名加载本地原图
                local_path = os.path.join(receiver_instance.config.test_data_dir, packet['filename'])
                original_bgr = cv2.imread(local_path)

                if original_bgr is None:
                    print(f"Error: Local file {packet['filename']} not found!")
                    continue

                # 2. 语义解码
                sem_img, _, _, _ = receiver_instance._decode({'feature': packet['semantic'], 'image': b''})

                # 3. 传统路径模拟
                qam_syms = packet['traditional_qam']
                # 模拟加噪与解码效果
                if ConfigReceiver.snr < 5:
                    trad_img = np.zeros_like(original_bgr)
                else:
                    trad_img = receiver_instance._add_gaussian_noise(original_bgr, ConfigReceiver.snr)

                # 4. 指标计算 (全部与本地原图对比)
                psnr_sem = calculate_psnr(original_bgr, sem_img)
                psnr_trad = calculate_psnr(original_bgr, trad_img)
                ssim_sem = calculate_ssim(original_bgr, sem_img)
                ssim_trad = calculate_ssim(original_bgr, trad_img)

                print(f"Image {packet['counter']} Processed. Sem PSNR: {psnr_sem:.2f}dB")

                # 推送前端，包含编码耗时
                socketio.emit('message', {
                    'semantic_psnr': round(psnr_sem, 2),
                    'traditional_psnr': round(psnr_trad, 2),
                    'semantic_ssim': round(ssim_sem, 4),
                    'traditional_ssim': round(ssim_trad, 4),
                    'sem_time': packet['sem_time'],
                    'jpeg_time': packet['jpeg_time']
                })
        except Exception as e:
            print(f"Receiver Error: {e}")


if __name__ == '__main__':
    threading.Thread(target=receiver_server, daemon=True).start()
    socketio.run(app, host='0.0.0.0', port=5004, debug=False)