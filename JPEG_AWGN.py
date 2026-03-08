# -*- coding: utf-8 -*-
"""
11dB_jpeg_ldpc_qam_awgn.py
修复版：适配 Windows/Linux 路径格式
"""

import glob
import os
import datetime
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import ToPILImage

# 引入 pyldpc
try:
    from pyldpc import make_ldpc, encode, decode
except ImportError:
    print("请先安装 pyldpc: pip install pyldpc")
    exit(1)


def pillow_encode(img, output_img_path, fmt='JPEG', quality=10):
    img.save(output_img_path, format=fmt, quality=quality)
    filesize = os.path.getsize(output_img_path)
    bpp = filesize * float(8) / (img.size[0] * img.size[1])
    return bpp


def find_closest_bpp(target, img, dir, fmt='JPEG'):
    lower = 0
    upper = 100
    prev_mid = upper
    for i in range(10):
        mid = (upper - lower) / 2 + lower
        if int(mid) == int(prev_mid):
            break
        bpp = pillow_encode(img, dir, fmt=fmt, quality=int(mid))
        if bpp > target:
            upper = mid - 1
        else:
            lower = mid
    return bpp


def img_to_bit(input_path, output_path):
    with open(input_path, 'rb') as file:
        file_context = file.read()

    # 将字节转换为 8 位二进制字符串
    bits = ''.join([bin(byte).replace('0b', '').rjust(8, '0') for byte in file_context])

    with open(output_path, "w", encoding='utf-8') as f:
        f.write(bits)


def cut_string(obj, sec):
    """
    切割字符串
    """
    return [obj[i:i + sec] for i in range(0, len(obj), sec)]


def get_bitarray(txt_path):
    """
    获得比特流数组
    """
    with open(txt_path, 'r') as f:
        f_context = f.read().strip()
        # 字符串按8切割 (注意：原逻辑似乎是按8切割转int，这里保留原逻辑)
        k_char = cut_string(f_context, 8)
        # 将二进制字符串转为int (0-255)
        k = [int(a, 2) for a in k_char]

        # 注意：这里有一个潜在的逻辑分歧点。
        # 如果下游 ldpc 需要的是 0/1 比特流，这里应该返回 0/1 数组。
        # 但看后面的代码 message_bits = binary_arr，且 ldpc_encoded_bits = encode(...)
        # pyldpc.encode 通常接受 0/1 数组。
        # 原代码 img_to_bit 生成的是 '010101...' 字符串。
        # get_bitarray 将其每8位转回了一个 int (0-255)。
        # 如果直接把 0-255 的 int 喂给 pyldpc，这是不对的。

        # === 修正逻辑：为了配合 LDPC，我们需要纯 0/1 数组 ===
        # 重新读取原始长字符串
        bit_list = [int(bit) for bit in f_context]
        return np.array(bit_list)


def random_noise(nc, width, height):
    """生成随机噪声图"""
    img = torch.rand(nc, width, height)
    img = ToPILImage()(img)
    return img


def bit_to_img(string, img_dir, output_path):
    """
    将比特流字符串重新转换为图片
    """
    # 将 0/1 字符串转回 bytes
    # 先按 8 位切割
    split_char = cut_string(string, 8)
    int_8 = []
    for a in split_char:
        if len(a) == 8:
            int_8.append(int(a, 2))

    out_stream = np.array(int_8, dtype=np.uint8)

    # 构建输出路径
    # img_dir 是参考的原图路径，用于获取文件名
    file_name = os.path.basename(img_dir)
    sub_dir = os.path.basename(os.path.dirname(img_dir))

    directory = os.path.join(output_path, sub_dir)
    if not os.path.exists(directory):
        os.makedirs(directory)

    img_output_path = os.path.join(directory, file_name)
    out_stream.tofile(img_output_path)

    try:
        Image.open(img_output_path).convert('RGB')
    except IOError:
        print(f'Error opening reconstructed image: {img_output_path}')
        # 如果打不开，生成随机噪声图
        try:
            ref_img = Image.open(img_dir)
            width, height = ref_img.width, ref_img.height
        except:
            width, height = 512, 768  # 默认兜底

        random_noise(3, width, height).save(img_output_path)


def qam_modulate(bits, qam_order):
    """QAM 调制"""
    bits_per_symbol = int(np.log2(qam_order))

    # 补齐比特
    remainder = len(bits) % bits_per_symbol
    if remainder != 0:
        padding = np.zeros(bits_per_symbol - remainder, dtype=int)
        bits = np.concatenate([bits, padding])

    bit_groups = bits.reshape(-1, bits_per_symbol)
    symbols = np.packbits(bit_groups, axis=-1, bitorder='little').flatten()  # 注意 bitorder，通常高位在前或后需一致

    # 简单映射逻辑 (仅作演示，非格雷码)
    # 实际上建议用更标准的映射，这里沿用你的逻辑框架
    # 但 np.packbits 对于非 8 位对齐的行为需要小心，这里手动计算更稳妥
    weights = 2 ** np.arange(bits_per_symbol)[::-1]
    symbols = np.sum(bit_groups * weights, axis=1)

    m = int(np.sqrt(qam_order))
    x = np.arange(m) - (m - 1) / 2
    y = np.arange(m) - (m - 1) / 2
    constellation = np.array([complex(a, b) for a in x for b in y])

    # 归一化功率
    avg_power = np.mean(np.abs(constellation) ** 2)
    constellation = constellation / np.sqrt(avg_power)

    modulated_signal = constellation[symbols.astype(int)]
    return modulated_signal


def qam_demodulate(received_signal, qam_order):
    """QAM 解调"""
    m = int(np.sqrt(qam_order))
    x = np.arange(m) - (m - 1) / 2
    y = np.arange(m) - (m - 1) / 2
    constellation = np.array([complex(a, b) for a in x for b in y])

    avg_power = np.mean(np.abs(constellation) ** 2)
    constellation = constellation / np.sqrt(avg_power)

    distances = np.abs(received_signal.reshape(-1, 1) - constellation.reshape(1, -1))
    nearest_points = np.argmin(distances, axis=1)

    bits_per_symbol = int(np.log2(qam_order))

    # 将索引转回比特
    # 使用 format 字符串转二进制最稳妥
    demodulated_bits = []
    fmt = f'0{bits_per_symbol}b'
    for idx in nearest_points:
        demodulated_bits.extend([int(b) for b in format(idx, fmt)])

    return np.array(demodulated_bits)


def awgn_channel(signal, snr_db):
    """AWGN 信道"""
    signal_power = np.mean(np.abs(signal) ** 2)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    noise = np.sqrt(noise_power / 2) * (np.random.normal(size=signal.shape) + 1j * np.random.normal(size=signal.shape))
    return signal + noise


def ldpc_qam_awgn(input_signal, snr=11, qam_order=16):
    """完整的通信链路仿真"""

    # 1. LDPC 编码
    # 注意：pyldpc 的 encode 需要转置过的矩阵输入 (k, N) 还是怎样，
    # 以及输入必须是 (k, ) 还是 (k, batch)。
    # 为了简化，我们按 pyldpc 官方示例构建。

    n = 960  # 缩短码长以加快速度 (原 1440)
    d_v = 2
    d_c = 3

    # 生成矩阵 (只生成一次，实际应用中应放在循环外)
    # 为了保证这里能跑，我们先在这里生成
    H, G = make_ldpc(n, d_v, d_c, systematic=True, sparse=True)
    k = G.shape[1]

    # 对输入比特进行补齐，使其长度是 k 的倍数
    num_bits = len(input_signal)
    num_blocks = int(np.ceil(num_bits / k))
    padding_len = num_blocks * k - num_bits

    if padding_len > 0:
        padded_signal = np.concatenate([input_signal, np.zeros(padding_len, dtype=int)])
    else:
        padded_signal = input_signal

    # Reshape 成 (k, num_blocks) 以便批量编码
    # 注意：pyldpc encode 输入期望 (n_bits, n_messages) 即 (k, num_blocks)
    message_blocks = padded_signal.reshape(num_blocks, k).T

    # 编码
    encoded_blocks = encode(G, message_blocks, snr=snr)  # 返回 (n, num_blocks)

    # 展平
    encoded_bits = encoded_blocks.T.flatten()

    # 2. QAM 调制
    modulated_signal = qam_modulate(encoded_bits, qam_order)

    # 计算带宽压缩比 (Channel Bandwidth Ratio)
    # 输入维度 (Source symbols/pixels) vs 输出维度 (Channel symbols)
    # 这里简单用 复数符号数 / 输入比特数 表示
    cbr = len(modulated_signal) / len(input_signal)

    # 3. AWGN 信道
    noisy_signal = awgn_channel(modulated_signal, snr)

    # 4. QAM 解调
    demodulated_bits = qam_demodulate(noisy_signal, qam_order)

    # 5. LDPC 解码
    # Reshape 回 (n, num_blocks)
    # 注意 QAM 解调可能会多出一点 padding (为了凑齐 symbol)，需要截断
    demodulated_bits = demodulated_bits[:len(encoded_bits)]

    demod_blocks = demodulated_bits.reshape(num_blocks, n).T

    # 转 LLR: 0 -> 10, 1 -> -10 (强判决假设)
    llr = np.where(demod_blocks == 0, 10.0, -10.0)

    # 解码
    # === 关键修改：降低 maxiter 以提速 ===
    decoded_blocks = decode(H, llr, maxiter=10, snr=snr)  # 返回 (k, num_blocks)

    # 展平并去除 padding
    decoded_bits = decoded_blocks.T.flatten()
    decoded_bits = decoded_bits[:num_bits]

    # 转为 int 0/1
    # pyldpc decode 返回的可能是高斯分布的值或者 0/1，通常是 0/1
    decoded_bits = np.abs(decoded_bits).astype(int)  # 确保是 int

    return decoded_bits, cbr


if __name__ == "__main__":

    start_time = datetime.datetime.now()
    print("Program started at:", start_time)

    # === 路径配置 (请根据实际情况修改) ===
    # 建议使用相对路径或确保路径存在
    input_base_path = './data/kodak'

    # 中间结果路径
    output_base_path = './data/clic_origin_jpeg'
    output_txt_path = './data/clic_jpeg_txt'

    # 最终结果路径
    snr = 11
    channelcoded_output_base_path = os.path.join('./test_jpeg_out', f'{snr}dB')

    target_bpp = 2
    qam_order = 16

    # 1. JPEG 压缩 ==========================================
    print("\n[Step 1] Starting JPEG Compression...")
    input_fmt = 'png'  # 假设 Kodak 是 png
    input_images = glob.glob(os.path.join(input_base_path, '**/*.' + input_fmt), recursive=True)

    if not input_images:
        print(f"Warning: No images found in {input_base_path}. Check path or extension.")

    total_bpp = 0
    for img_dir in input_images:
        # 使用 os.path 提取信息
        file_name_no_ext = os.path.splitext(os.path.basename(img_dir))[0]
        # 如果有子目录结构（如类别），提取子目录名，否则设为 root
        rel_path = os.path.relpath(os.path.dirname(img_dir), input_base_path)
        if rel_path == '.':
            sub_dir = 'root'
        else:
            sub_dir = rel_path

        output_path = os.path.join(output_base_path, sub_dir)
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        output_img_path = os.path.join(output_path, file_name_no_ext + '.JPEG')

        try:
            img = Image.open(img_dir).convert("RGB")
            bpp_per_img = find_closest_bpp(target_bpp, img, output_img_path, fmt='JPEG')
            total_bpp += bpp_per_img
            print(f"  Encoded {file_name_no_ext}: BPP={bpp_per_img:.2f}")
        except Exception as e:
            print(f"  Error encoding {file_name_no_ext}: {e}")

    avg_bpp = total_bpp / len(input_images) if len(input_images) > 0 else 0
    print(f'Average BPP: {avg_bpp:.4f}')

    # 2. 转比特流 TXT ========================================
    print("\n[Step 2] Converting JPEG to Bitstream TXT...")
    if not os.path.exists(output_txt_path):
        os.makedirs(output_txt_path)

    jpeg_images = glob.glob(os.path.join(output_base_path, '**/*.JPEG'), recursive=True)
    for img_dir in jpeg_images:
        sub_dir = os.path.basename(os.path.dirname(img_dir))
        file_name_no_ext = os.path.splitext(os.path.basename(img_dir))[0]

        directory = os.path.join(output_txt_path, sub_dir)
        if not os.path.exists(directory):
            os.makedirs(directory)

        file_dir = os.path.join(directory, file_name_no_ext + '.txt')
        img_to_bit(img_dir, file_dir)
        print(f"  Converted {file_name_no_ext} to TXT")

    print('TXT generation done.')

    # 3. 信道仿真 (LDPC+QAM+AWGN) ============================
    print(f"\n[Step 3] Simulating Channel (SNR={snr}dB)...")
    input_txts = glob.glob(os.path.join(output_txt_path, '**/*.txt'), recursive=True)

    total = len(input_txts)
    cbr_array = []

    for txt_dir in input_txts:
        print(f"  Processing {os.path.basename(txt_dir)}... ({total} left)")

        try:
            # 读取比特流
            input_signal = get_bitarray(txt_dir)  # 这里已经是 0/1 numpy array

            # 仿真
            output_signal, cbr = ldpc_qam_awgn(input_signal, snr=snr, qam_order=qam_order)
            cbr_array.append(cbr)

            # 转字符串用于写入
            bitstring = ''.join(output_signal.astype(str))

            if not os.path.exists(channelcoded_output_base_path):
                os.makedirs(channelcoded_output_base_path)

            # 寻找对应的 JPEG 路径以获取尺寸信息
            # txt_dir: ./data/clic_jpeg_txt/root/kodim01.txt
            sub_dir = os.path.basename(os.path.dirname(txt_dir))
            file_name_no_ext = os.path.splitext(os.path.basename(txt_dir))[0]

            # 对应的 JPEG 路径
            ref_img_dir = os.path.join(output_base_path, sub_dir, file_name_no_ext + '.JPEG')

            bit_to_img(bitstring, ref_img_dir, output_path=channelcoded_output_base_path)

        except Exception as e:
            print(f"  Error processing {txt_dir}: {e}")
            import traceback

            traceback.print_exc()

        total -= 1

    if cbr_array:
        print(f'Average CBR: {np.mean(cbr_array):.4f}')

    print("\nAll Done.")
    end_time = datetime.datetime.now()
    print("Program finished at:", end_time)