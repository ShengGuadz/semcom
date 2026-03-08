"""
DeepJSCC 接收端 - UDP 版本
实现数据包重组与丢包检测
"""
import socket
import time
import pickle
import cv2
import zlib
import numpy as np
import struct
from flask import Flask, render_template
from flask_socketio import SocketIO
import threading
import eventlet
import os
import glob
from sender_jscc import ConfigSender, Sender
# 如果你有 pyldpc，可以把这个改为 True
USE_LDPC_SIM = False

eventlet.monkey_patch()
from receiver_jscc import ConfigReceiver, Receiver, calculate_psnr, calculate_ssim

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")
receiver = Receiver(ConfigReceiver())
DATASET_DIR = "./data/kodak/"

# 接收缓冲区： { image_id: { 'total': N, 'chunks': {seq_id: data}, 'last_seen': timestamp } }
reassembly_buffer = {}


def get_ground_truth_images(data_dir):
    if not os.path.exists(data_dir): return []
    return sorted(glob.glob(os.path.join(data_dir, '*.*')))


@app.route('/receiver')
def show_receiver():
    return render_template('receiver_deepjscc.html')


def receiver_server():
    # 改为 SOCK_DGRAM (UDP)
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind((ConfigReceiver.host, ConfigReceiver.port))
    # 增大接收缓冲区，非常重要！
    s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 10 * 1024 * 1024)
    # === 优化 1: 启动等待 (给接收端和前端留时间) ===
    print("Sender started. Waiting 5 seconds before transmission...")
    print("Please ensure Receiver is running and Web Page is open!")
    time.sleep(5)

    # === 优化 2: 网络层热身 (防止 ARP 丢包) ===
    print("Warming up network ARP cache...")
    try:
        # 发送一些无意义的小包，激活交换机/路由器的 MAC 表
        for _ in range(10):
            s.sendto(b'WARMUP_PACKET', (ConfigSender.host, ConfigSender.port))
            time.sleep(0.05)
    except Exception as e:
        print(f"Warmup warning: {e}")

    counter = 1
    print(f"\nUDP Receiver listening on {ConfigReceiver.host}:{ConfigReceiver.port}")

    gt_image_paths = get_ground_truth_images(DATASET_DIR)

    # 统计数据
    stats = {
        'psnr_trad': [], 'psnr_sem': [],
        'ssim_trad': [], 'ssim_sem': [],
        'size_trad': [], 'size_sem': []
    }

    last_cleanup_time = time.time()

    while True:
        try:
            # UDP 接收 (最大 64KB，我们发送的是 1400B)
            packet, addr = s.recvfrom(65535)

            # 1. 解析头部 (12 bytes)
            if len(packet) < 12: continue
            image_id, seq_id, total_packets = struct.unpack('!III', packet[:12])
            payload = packet[12:]

            # 2. 存入缓冲区
            if image_id not in reassembly_buffer:
                print(f"Start receiving Image {image_id} (Total packets: {total_packets})")
                reassembly_buffer[image_id] = {
                    'total': total_packets,
                    'chunks': {},
                    'last_seen': time.time()
                }

            # 记录数据块
            reassembly_buffer[image_id]['chunks'][seq_id] = payload
            reassembly_buffer[image_id]['last_seen'] = time.time()

            # 3. 检查是否收齐
            current_count = len(reassembly_buffer[image_id]['chunks'])

            if current_count == total_packets:
                print(f"Image {image_id} reassembled successfully!")

                # 重组数据
                chunks = reassembly_buffer[image_id]['chunks']
                full_data = b''.join([chunks[i] for i in range(total_packets)])

                # 清理缓冲区
                del reassembly_buffer[image_id]

                # === 开始处理完整数据 ===
                process_data(full_data, stats, gt_image_paths)

            # 4. 清理超时数据 (每1秒检查一次)
            if time.time() - last_cleanup_time > 1.0:
                cleanup_buffer()
                last_cleanup_time = time.time()

        except Exception as e:
            print(f"Error in main loop: {e}")


def cleanup_buffer():
    """清理超过 3 秒未收齐的残缺帧"""
    now = time.time()
    to_delete = []
    for img_id, info in reassembly_buffer.items():
        if now - info['last_seen'] > 3.0:
            print(f"Timeout: Dropping incomplete Image {img_id} ({len(info['chunks'])}/{info['total']} packets)")
            to_delete.append(img_id)

    for img_id in to_delete:
        del reassembly_buffer[img_id]


def process_data(bytes_data, stats, gt_image_paths):
    try:
        data = pickle.loads(bytes_data)
        counter = data['counter']

        # 提取数据
        compressed_feature, min_v, max_v, shape, dtype = data['feature']

        # 1. 熵解码
        decompressed = zlib.decompress(compressed_feature)
        feat_q = np.frombuffer(decompressed, dtype=dtype).reshape(shape)

        # 构造 decode 所需格式
        data['feature'] = [feat_q, min_v, max_v, shape]

        # 2. 语义解码 (含模拟信道噪声)
        sem_img, jpeg_clean, _, _ = receiver._decode(data)

        # 3. 传统解码 (含模拟比特错误)
        # 即使是UDP，pickle解包成功说明数据是完整的。
        # 为了对比，我们依然在这里模拟 "如果传统JPEG走同样的信道会发生什么"
        jpeg_bytes = data['image']
        corrupted_jpeg = receiver._simulate_bit_errors(jpeg_bytes, ConfigReceiver.snr)
        trad_img = cv2.imdecode(np.frombuffer(corrupted_jpeg, np.uint8), cv2.IMREAD_COLOR)

        if trad_img is None:
            trad_img = np.zeros_like(sem_img)
            print("JPEG decode failed (simulated).")
        else:
            if trad_img.shape != sem_img.shape:
                trad_img = cv2.resize(trad_img, (sem_img.shape[1], sem_img.shape[0]))

        # 4. 加载 GT
        gt_idx = counter - 1
        gt_img = None
        if gt_idx < len(gt_image_paths):
            gt_img = cv2.imread(gt_image_paths[gt_idx])
            if gt_img is not None and gt_img.shape != sem_img.shape:
                gt_img = cv2.resize(gt_img, (sem_img.shape[1], sem_img.shape[0]))
        if gt_img is None: gt_img = jpeg_clean

        # 5. 指标
        p_sem = calculate_psnr(gt_img, sem_img)
        p_trad = calculate_psnr(gt_img, trad_img)

        # 更新统计
        stats['psnr_sem'].append(p_sem)
        stats['psnr_trad'].append(p_trad)

        print(f"Img {counter}: Sem PSNR {p_sem:.2f} | Trad PSNR {p_trad:.2f}")

        # 前端推送
        # (此处省略部分路径保存代码，逻辑同 TCP 版本，仅做路径替换)
        sem_path = f"{receiver.config.save_dir_semantic}{counter}.png"
        trad_path = f"{receiver.config.save_dir_traditional}{counter}.png"
        cv2.imwrite(sem_path, sem_img)
        cv2.imwrite(trad_path, trad_img)

        socketio.emit('message', {
            'traditional_image_url': trad_path,
            'semantic_image_url': sem_path,
            'semantic_psnr': round(p_sem, 2),
            'traditional_psnr': round(p_trad, 2),
            'snr': ConfigReceiver.snr
        })

    except Exception as e:
        print(f"Error processing data: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    threading.Thread(target=receiver_server, daemon=True).start()
    print(f"Web interface: http://127.0.0.1:5004/receiver")
    socketio.run(app, host='0.0.0.0', port=5004, debug=False)