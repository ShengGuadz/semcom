"""
app_receiver_hailo.py
适配 Hailo 发送端的接收端主程序 (完整版: 含吞吐量与压缩率统计)
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

eventlet.monkey_patch()

# [关键] 导入 Hailo 专用的接收逻辑 (处理旋转和维度)
# 同时导入通用的计算函数
from receiver_hailo import ConfigReceiver, Receiver
from receiver_jscc import calculate_psnr, calculate_ssim

app = Flask(__name__)
# 禁用 http 日志
import logging

log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

socketio = SocketIO(app, cors_allowed_origins="*")
receiver = Receiver(ConfigReceiver())


@app.route('/receiver')
def show_receiver():
    return render_template('receiver_deepjscc.html')


def receiver_server():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((ConfigReceiver.host, ConfigReceiver.port))
    s.listen(5)

    print(f"\nReceiver listening on {ConfigReceiver.host}:{ConfigReceiver.port}")
    print("Waiting for sender connection...")

    conn, addr = s.accept()
    print(f"Connected by: {addr}")

    counter = 1

    # ============ 统计列表初始化 ============
    traditional_psnr_list = []
    semantic_psnr_list = []
    traditional_ssim_list = []
    semantic_ssim_list = []

    # 新增：带宽与大小统计
    traditional_kb_list = []
    semantic_kb_list = []
    packet_kb_list = []
    # ======================================

    while True:
        try:
            # 1. 接收长度头
            len_bytes = conn.recv(4)
            if not len_bytes:
                print("Sender disconnected.")
                break
            length = int.from_bytes(len_bytes, 'big')

            # 记录实际网络包大小 (KB)
            packet_kb_list.append(length / 1024.0)

            # 2. 接收数据体
            received = bytearray()
            while len(received) < length:
                chunk = conn.recv(min(4096, length - len(received)))
                if not chunk: break
                received.extend(chunk)

            # 3. 反序列化
            data = pickle.loads(received)
            cnt = data['counter']
            rotated = data.get('rotated', 0)

            print(f"\n[{time.time():.2f}] Decoding Image {cnt} (Rotated: {rotated})...")

            # 4. 语义解码 (receiver_hailo 会自动处理旋转恢复)
            sem_img, orig_img, feat_q, trad_size = receiver._decode(data)

            # 5. 传统通信对比 (JPEG + 误码模拟)
            jpeg_data = data['image']
            # 模拟信道传输错误
            corr_jpeg = receiver._simulate_bit_errors(jpeg_data, ConfigReceiver.snr)
            # 解码受损 JPEG
            trad_img = cv2.imdecode(np.frombuffer(corr_jpeg, np.uint8), cv2.IMREAD_COLOR)

            # [关键] 旋转处理修正
            if trad_img is not None and rotated == 1:
                if trad_img.shape != orig_img.shape:
                    trad_img = np.transpose(trad_img, (1, 0, 2))

            if trad_img is None:
                print("JPEG decoding failed, using original as fallback.")
                trad_img = orig_img

                # 6. 保存图片
            sem_path = f"{receiver.config.save_dir_semantic}{cnt}.png"
            trad_path = f"{receiver.config.save_dir_traditional}{cnt}.png"
            cv2.imwrite(sem_path, sem_img)
            cv2.imwrite(trad_path, trad_img)

            # 7. 计算指标 (PSNR & SSIM)
            try:
                sem_psnr = calculate_psnr(orig_img, sem_img)
                trad_psnr = calculate_psnr(orig_img, trad_img)
                sem_ssim = calculate_ssim(orig_img, sem_img)
                trad_ssim = calculate_ssim(orig_img, trad_img)

                # 累加
                semantic_psnr_list.append(sem_psnr)
                traditional_psnr_list.append(trad_psnr)
                semantic_ssim_list.append(sem_ssim)
                traditional_ssim_list.append(trad_ssim)

            except Exception as e:
                print(f"Metric calc error: {e}")
                sem_psnr, trad_psnr, sem_ssim, trad_ssim = 0, 0, 0, 0

            # ============ [新增] 计算压缩率与吞吐量 ============
            original_image_size = data['original_image_size']  # KB
            # 语义特征大小 (KB)
            semantic_feature_size = feat_q.nbytes / 1024.0

            traditional_kb_list.append(float(trad_size))  # trad_size 来自 _decode 返回的 JPEG 大小
            semantic_kb_list.append(float(semantic_feature_size))

            # 压缩率 (Percentage Savings)
            # 公式: (原图 - 压缩后) / 原图 * 100%
            trad_cr = round((original_image_size - trad_size) * 100 / max(original_image_size, 1), 2)
            sem_cr = round((original_image_size - semantic_feature_size) * 100 / max(original_image_size, 1), 2)

            # 吞吐量 (Throughput)
            # 假设带宽 100 Mbps = 102400 Kbps = 12800 KB/s (这里沿用原代码逻辑，直接除以KB大小得到 imgs/s)
            # 注意：原代码逻辑 bandwidth_kbps = 102400 (bits) / size (KB*8)?
            # 原代码逻辑: 102400 / size_in_KB. 这其实是假设 100MBps? 不，100Mbps ≈ 12500 KB/s.
            # 为了完全对齐原代码 app_receiver_deepjscc.py:
            bandwidth_kbps = 102400
            trad_throughput = round(bandwidth_kbps / max(trad_size, 1), 2)
            sem_throughput = round(bandwidth_kbps / max(semantic_feature_size, 1), 2)

            print(f"Metrics -> Sem: PSNR={sem_psnr:.2f}, SSIM={sem_ssim:.4f}, Size={semantic_feature_size:.2f}KB")
            print(f"           Trad: PSNR={trad_psnr:.2f}, SSIM={trad_ssim:.4f}, Size={trad_size}KB")
            print(f"Stats   -> Sem CR={sem_cr}%, Throughput={sem_throughput} fps")

            # 8. 推送前端 (补全字段)
            socketio.emit('message', {
                'traditional_image_url': trad_path,
                'semantic_image_url': sem_path,
                'traditional_image_size': trad_size,
                'semantic_feature_size': int(semantic_feature_size),
                'semantic_psnr': round(sem_psnr, 2),
                'traditional_psnr': round(trad_psnr, 2),
                'semantic_ssim': round(sem_ssim, 4),
                'traditional_ssim': round(trad_ssim, 4),
                # 新增字段对接前端图表
                'traditional_compression_ratio': trad_cr,
                'semantic_compression_ratio': sem_cr,
                'traditional_throughput': trad_throughput,
                'semantic_throughput': sem_throughput,
            })

            # 9. 发送 ACK
            conn.sendall((1).to_bytes(4, 'big'))
            counter += 1

        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            break

    # === 10. 最终平均指标统计 ===
    print(f"\n{'=' * 25} FINAL STATISTICS {'=' * 25}")
    total_imgs = len(semantic_psnr_list)
    print(f"Total Images Received: {total_imgs}")

    if total_imgs > 0:
        # 质量指标
        avg_sem_psnr = sum(semantic_psnr_list) / total_imgs
        avg_trad_psnr = sum(traditional_psnr_list) / total_imgs
        avg_sem_ssim = sum(semantic_ssim_list) / total_imgs
        avg_trad_ssim = sum(traditional_ssim_list) / total_imgs

        # 流量指标
        avg_trad_kb = sum(traditional_kb_list) / total_imgs
        avg_sem_kb = sum(semantic_kb_list) / total_imgs
        avg_pkt_kb = sum(packet_kb_list) / total_imgs

        print(f"AVERAGE QUALITY METRICS:")
        print(f"  Semantic    - PSNR: {avg_sem_psnr:.2f} dB, SSIM: {avg_sem_ssim:.4f}")
        print(f"  Traditional - PSNR: {avg_trad_psnr:.2f} dB, SSIM: {avg_trad_ssim:.4f}")

        print(f"AVERAGE BANDWIDTH (KB/img):")
        print(f"  JPEG Payload:     {avg_trad_kb:.2f} KB")
        print(f"  Semantic Payload: {avg_sem_kb:.2f} KB")
        print(f"  Total Socket Pkt: {avg_pkt_kb:.2f} KB")

        if avg_trad_kb > 0:
            saving = (1.0 - avg_sem_kb / avg_trad_kb) * 100.0
            print(f"  Payload Saving:   {saving:.2f}% (vs JPEG)")
    else:
        print("No metrics collected.")
    print(f"{'=' * 68}\n")
    # ============================

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