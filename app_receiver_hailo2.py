"""
app_receiver_hailo.py (修改版)
1. 加载本地 DATASET 作为 Ground Truth
2. 修复高 SNR 下 PSNR=100 的问题
3. 增加对齐逻辑：将本地原图 Resize 到语义图大小进行对比
"""
import socket
import time
import pickle
import cv2
import threading
from flask import Flask, render_template
from flask_socketio import SocketIO
import eventlet
import numpy as np
import glob
import os

eventlet.monkey_patch()

from receiver_hailo import ConfigReceiver, Receiver
from receiver_deepjscc import calculate_psnr, calculate_ssim

# ================= 配置区 =================
# 必须与发送端的数据集完全一致，且文件名排序一致
DATASET_DIR = "./data/kodak/"
# DATASET_DIR = "./data/military_test/"
# =========================================

app = Flask(__name__)
# 禁用日志干扰
import logging

log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

socketio = SocketIO(app, cors_allowed_origins="*")
receiver = Receiver(ConfigReceiver())


def get_ground_truth_images(data_dir):
    """加载本地数据集路径"""
    if not os.path.exists(data_dir):
        print(f"[Error] Dataset dir {data_dir} not found!")
        return []
    # 使用 sorted 确保和发送端顺序一致
    paths = sorted(glob.glob(os.path.join(data_dir, '*.*')))
    print(f"Loaded {len(paths)} local images from {data_dir}")
    return paths


@app.route('/receiver')
def show_receiver():
    return render_template('receiver_deepjscc2.html')


def receiver_server():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((ConfigReceiver.host, ConfigReceiver.port))
    s.listen(5)

    # 1. 预加载本地数据集
    gt_image_paths = get_ground_truth_images(DATASET_DIR)

    print(f"\nReceiver listening on {ConfigReceiver.host}:{ConfigReceiver.port}")
    print("Waiting for sender connection...")

    conn, addr = s.accept()
    print(f"Connected by: {addr}")

    # 统计变量
    counter = 1
    traditional_psnr_list = []
    semantic_psnr_list = []
    traditional_ssim_list = []
    semantic_ssim_list = []
    traditional_kb_list = []
    semantic_kb_list = []

    while True:
        try:
            # --- 接收数据 ---
            len_bytes = conn.recv(4)
            if not len_bytes:
                print("Sender disconnected.")
                break
            length = int.from_bytes(len_bytes, 'big')

            received = bytearray()
            while len(received) < length:
                chunk = conn.recv(min(4096, length - len(received)))
                if not chunk: break
                received.extend(chunk)

            # --- 解析数据 ---
            data = pickle.loads(received)
            # 发送端的 counter 从 1 开始
            img_idx = data['counter'] - 1
            rotated = data.get('rotated', 0)

            print(f"\n{'=' * 40}")
            print(f"[{time.time():.2f}] Processing Image {counter} (Rotated: {rotated})")

            # --- 1. 获取 Ground Truth (真值) ---
            ground_truth_img = None
            if img_idx < len(gt_image_paths):
                ground_truth_img = cv2.imread(gt_image_paths[img_idx])
                if ground_truth_img is None:
                    print(f"Warning: Could not read local file: {gt_image_paths[img_idx]}")
            else:
                print(f"Warning: Index {img_idx} out of range for local dataset.")

            # --- 2. 语义解码 ---
            # sem_img 已经处理过反旋转，方向是正确的
            # jpeg_clean 是发送端压缩过的无噪图（仅作备用，不做GT）
            sem_img, jpeg_clean, feat_q, trad_size_kb = receiver._decode(data)

            # --- 3. 传统通信模拟 (JPEG + 噪声) ---
            jpeg_data = data['image']  # 这是发送端发来的 JPEG 字节流

            # 模拟误码
            corrupted_jpeg = receiver._simulate_bit_errors(jpeg_data, ConfigReceiver.snr)

            # 解码受损 JPEG
            noisy_trad_img = cv2.imdecode(np.frombuffer(corrupted_jpeg, np.uint8), cv2.IMREAD_COLOR)

            # 如果 JPEG 解码成功且发生了旋转，需要转回来
            if noisy_trad_img is not None and rotated == 1:
                # 只有当维度不对时才转，防止逻辑重复
                if noisy_trad_img.shape != sem_img.shape:
                    noisy_trad_img = np.transpose(noisy_trad_img, (1, 0, 2))

            # 如果解码彻底失败（数据损坏太严重），用黑图代替，惩罚 PSNR
            if noisy_trad_img is None:
                print("JPEG decoding failed due to channel errors.")
                noisy_trad_img = np.zeros_like(sem_img)

            # --- 4. [关键] 尺寸对齐逻辑 ---
            # DeepJSCC 发送端可能为了适配模型做了 Resize。
            # 我们必须以 sem_img 的尺寸为基准，将 GT 和 noisy_trad_img 统一到该尺寸。

            target_h, target_w = sem_img.shape[:2]

            # 4.1 准备用于计算指标的 GT
            metric_gt = None
            if ground_truth_img is not None:
                # 如果本地原图尺寸与语义图不一致，进行缩放
                if ground_truth_img.shape[:2] != (target_h, target_w):
                    # print(f"Resizing GT from {ground_truth_img.shape[:2]} to {(target_h, target_w)}")
                    metric_gt = cv2.resize(ground_truth_img, (target_w, target_h))
                else:
                    metric_gt = ground_truth_img
            else:
                # 如果没找到本地图，被迫使用发送端的 Clean JPEG (不推荐)
                print("Using Transmitted JPEG as GT (Not accurate).")
                metric_gt = jpeg_clean
                if metric_gt.shape[:2] != (target_h, target_w):
                    metric_gt = cv2.resize(metric_gt, (target_w, target_h))

            # 4.2 准备用于计算指标的 Traditional Image
            # 同样需要 Resize，因为 JPEG 解码出来的可能尺寸有微小差异或旋转问题
            if noisy_trad_img.shape[:2] != (target_h, target_w):
                noisy_trad_img = cv2.resize(noisy_trad_img, (target_w, target_h))

            # --- 5. 保存图片 ---
            sem_path = f"{receiver.config.save_dir_semantic}{counter}.png"
            trad_path = f"{receiver.config.save_dir_traditional}{counter}.png"
            cv2.imwrite(sem_path, sem_img)
            cv2.imwrite(trad_path, noisy_trad_img)

            # --- 6. 计算指标 (对比 metric_gt) ---
            try:
                # 语义 vs 真值
                sem_psnr = calculate_psnr(metric_gt, sem_img)
                sem_ssim = calculate_ssim(metric_gt, sem_img)

                # 传统(噪声) vs 真值
                trad_psnr = calculate_psnr(metric_gt, noisy_trad_img)
                trad_ssim = calculate_ssim(metric_gt, noisy_trad_img)

                # 记录
                semantic_psnr_list.append(sem_psnr)
                traditional_psnr_list.append(trad_psnr)
                semantic_ssim_list.append(sem_ssim)
                traditional_ssim_list.append(trad_ssim)

            except Exception as e:
                print(f"Metric Error: {e}")
                sem_psnr, trad_psnr = 0, 0
                sem_ssim, trad_ssim = 0, 0

            # --- 7. 带宽与压缩率 ---
            # 真正的原图大小 (KB)
            raw_size_kb = (metric_gt.shape[0] * metric_gt.shape[1] * 3) / 1024.0
            sem_feat_kb = feat_q.nbytes / 1024.0
            trad_feat_kb = float(trad_size_kb)

            traditional_kb_list.append(trad_feat_kb)
            semantic_kb_list.append(sem_feat_kb)

            # 压缩率
            trad_cr = round(raw_size_kb / max(trad_feat_kb, 1), 2)
            sem_cr = round(raw_size_kb / max(sem_feat_kb, 1), 2)

            print(f"Metrics -> Sem:  PSNR={sem_psnr:.2f}, SSIM={sem_ssim:.4f}")
            print(f"           Trad: PSNR={trad_psnr:.2f}, SSIM={trad_ssim:.4f}")

            # --- 8. 推送前端 ---
            socketio.emit('message', {
                'traditional_image_url': trad_path,
                'semantic_image_url': sem_path,
                'traditional_image_size': trad_feat_kb,
                'semantic_feature_size': int(sem_feat_kb),
                'semantic_psnr': round(sem_psnr, 2),
                'traditional_psnr': round(trad_psnr, 2),
                'semantic_ssim': round(sem_ssim, 4),
                'traditional_ssim': round(trad_ssim, 4),
                'traditional_compression_ratio': trad_cr,
                'semantic_compression_ratio': sem_cr,
                'traditional_throughput': 0,  # 可选：保留原来的吞吐量计算
                'semantic_throughput': 0,
            })

            # ACK
            conn.sendall((1).to_bytes(4, 'big'))
            counter += 1

        except Exception as e:
            print(f"Main Loop Error: {e}")
            import traceback
            traceback.print_exc()
            break

    # --- 最终统计 ---
    print(f"\n{'=' * 25} RESULT {'=' * 25}")
    if len(semantic_psnr_list) > 0:
        avg_sem_p = sum(semantic_psnr_list) / len(semantic_psnr_list)
        avg_trad_p = sum(traditional_psnr_list) / len(traditional_psnr_list)
        avg_sem_s = sum(semantic_ssim_list) / len(semantic_ssim_list)
        avg_trad_s = sum(traditional_ssim_list) / len(traditional_ssim_list)

        print(f"Avg Semantic PSNR:    {avg_sem_p:.2f} dB")
        print(f"Avg Traditional PSNR: {avg_trad_p:.2f} dB")
        print(f"Avg Semantic SSIM:    {avg_sem_s:.4f}")
        print(f"Avg Traditional SSIM: {avg_trad_s:.4f}")
    print(f"{'=' * 58}\n")

    conn.close()
    s.close()


if __name__ == '__main__':
    # 启动接收线程
    threading.Thread(target=receiver_server, daemon=True).start()

    # 打印访问地址
    print("\n" + "=" * 50)
    print("  DeepJSCC Receiver (Hybrid) Running")
    print(f"  > Dashboard:    http://127.0.0.1:5004/receiver")
    print(f"  > Network Addr: {ConfigReceiver.host}:{ConfigReceiver.port}")
    print("=" * 50 + "\n")

    # 启动 Web 服务
    socketio.run(app, host='0.0.0.0', port=5004, debug=False)