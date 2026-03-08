"""
jpeg_ldpc_qam_awgn.py
@author Echo
@description
对输入数据集图片模拟经过JPEG压缩并进行LDPC信道编码 -> QAM调制 -> AWGN信道 -> QAM解调 -> LDPC信道解码最后恢复出图片的整套流程
"""

from pyldpc import make_ldpc, encode, decode
import glob
import os
# import commpy
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import ToPILImage
# from commpy.channelcoding.ldpc import get_ldpc_code_params, ldpc_bp_decode, triang_ldpc_systematic_encode
import datetime

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


# def img_to_bit(input_path, output_path):
#     file = open(input_path, 'rb')  # 输入压缩后的文件
#     file_context = file.read()  # <class 'bytes'>字节流
#
#     tmp_a = []
#     bit_all = ''
#     for i in file_context:
#         tmp_a.append(i)  # int类型的数据
#     tmp_b = np.array(tmp_a, dtype=np.uint8)
#     for j in tmp_b:
#         k = bin(j).replace('0b', '').rjust(8, '0')
#         bit_all = bit_all + k
#     with open(output_path, "w", encoding='utf-8') as f:
#         f.write(bit_all)
#         f.close()
def img_to_bit(input_path, output_path):
    with open(input_path, 'rb') as file:
        file_context = file.read()

    bits = ''.join([bin(byte).replace('0b', '').rjust(8, '0') for byte in file_context])

    with open(output_path, "w", encoding='utf-8') as f:
        f.write(bits)


def cut_string(obj, sec):
    """
    切割字符串
    :param obj: 输入字符串
    :param sec: 切割的位数
    :return: 切割后的字符串
    """
    return [obj[i:i + sec] for i in range(0, len(obj), sec)]

def optimized_get_bitarray(txt_path):
    """
    获得比特流数组
    :param txt_path: 图片对应比特流存储的txt路径
    :return: 返回比特流数组，由0和1组成
    """
    with open(txt_path, 'r') as f:
        f_context = f.read().strip()  # 读取字符串

    # 将字符串中的每个字符转换为0或1
    bit_array = np.array([int(bit) for bit in f_context])
    return bit_array

def get_bitarray(txt_path):
    """
    获得比特流数组
    :param txt_path: 图片对应比特流存储的txt路径
    :return: 返回比特流数组
    """
    with open(txt_path, 'r') as f:
        f_context = f.read().strip()  # 读取字符串
        k_char = cut_string(f_context, 1)  # 字符串按8切割
        # int(a, 2)表示将二进制的字符串a表示为十进制的int
        k = [int(a, 2) for a in k_char]  # 字符串转换为int类型的数据
        bit_array = np.array(k)
        return bit_array


def get_bitstring(input_path):
    """
    获得比特流
    :param input_path: 图片对应比特流存储的txt路径
    :return: 返回比特流字符串
    """
    with open(input_path, 'r') as f:
        f_context = f.read().strip()  # 读取字符串
        k_char = cut_string(f_context, 1)  # 字符串按8切割
        # int(a, 2)表示将二进制的字符串a表示为十进制的int
        k = [int(a, 2) for a in k_char]  # 字符串转换为int类型的数据
        bit_array = np.array(k)

        bitstring = ''
        for i in bit_array:
            bitstring += str(i)
        return bitstring


def random_noise(nc, width, height):
    """Generator a random noise image from tensor.

    If nc is 1, the Grayscale image will be created.
    If nc is 3, the RGB image will be generated.

    Args:
        nc (int): (1 or 3) number of channels.
        width (int): width of output image.
        height (int): height of output image.
    Returns:
        PIL Image.
    """
    img = torch.rand(nc, width, height)
    img = ToPILImage()(img)
    return img


def bit_to_img(string, img_dir, output_path):
    """
    将比特流字符串重新转换为图片，若转换后的图片无法打开则将之变为一副对应尺寸的随机噪声图
    :param img_dir: 输入图片路径
    :param string: 图片比特流字符串
    :param output_path: 输出图片文件夹路径
    :return: None
    """
    split_char = cut_string(string, 8)  # 字符串按8切割
    # int(a, 2)表示将二进制的字符串a表示为十进制的int
    int_8 = [int(a, 2) for a in split_char]  # 字符串转换为int类型的数据
    out_stream = np.array(int_8, dtype=np.uint8)
    # print(out_stream)
    # print(out_stream.size)
    directory = output_path + '/' + os.path.basename(os.path.dirname(img_dir))
    if not os.path.exists(directory):
        os.makedirs(directory)
    img_output_path = directory + '/' + os.path.basename(img_dir)
    out_stream.tofile(img_output_path)

    try:
        Image.open(img_output_path).convert('RGB')
    except IOError:
        print('Error')
        width = Image.open(img_dir).width
        height = Image.open(img_dir).height
        random_noise(3, width, height).save(img_output_path)

def qam_modulate(bits, qam_order):
    """
    QAM 调制实现
    :param bits: 输入的比特流（1维数组）
    :param qam_order: QAM 调制阶数（如 16 表示 16-QAM）
    :return: QAM 调制后的复数信号
    """
    # 每个 QAM 符号需要 log2(QAM阶数) 个比特
    bits_per_symbol = int(np.log2(qam_order))
    assert len(bits) % bits_per_symbol == 0, "比特长度必须是每个符号需要的比特数的倍数"

    # 将比特流切分为每组 log2(QAM阶数) 个比特
    bit_groups = bits.reshape(-1, bits_per_symbol)

    # 将每组比特解释为整数（符号映射的索引）
    symbols = np.packbits(bit_groups, axis=-1, bitorder='little').flatten()

    # 构建 QAM 星座图
    m = int(np.sqrt(qam_order))  # QAM 星座图是 m x m 的正方形
    x = np.arange(m) - (m - 1) / 2  # 星座点的实部
    y = np.arange(m) - (m - 1) / 2  # 星座点的虚部
    constellation = np.array([complex(a, b) for a in x for b in y])

    # 使用星座图进行映射
    modulated_signal = constellation[symbols]
    return modulated_signal


def qam_demodulate(received_signal, qam_order):
    """
    QAM 解调实现
    :param received_signal: 接收到的 QAM 调制信号（复数数组）
    :param qam_order: QAM 调制阶数（如 16 表示 16-QAM）
    :return: 解调后的比特流（1维数组）
    """
    # 构建 QAM 星座图
    m = int(np.sqrt(qam_order))
    x = np.arange(m) - (m - 1) / 2
    y = np.arange(m) - (m - 1) / 2
    constellation = np.array([complex(a, b) for a in x for b in y])

    # 找到接收到的信号点到星座点的最近邻点
    distances = np.abs(received_signal.reshape(-1, 1) - constellation.reshape(1, -1))
    nearest_points = np.argmin(distances, axis=1)

    # 将星座点索引转为比特流
    bits_per_symbol = int(np.log2(qam_order))
    demodulated_bits = np.unpackbits(nearest_points.astype(np.uint8), bitorder='little')[-bits_per_symbol:]
    return demodulated_bits


def awgn_channel(signal, snr_db):
    """
    AWGN 信道实现
    :param signal: 输入信号（复数数组）
    :param snr_db: 信噪比（以 dB 为单位）
    :return: 加噪后的信号
    """
    # 计算信号功率
    signal_power = np.mean(np.abs(signal)**2)

    # 将 SNR 从 dB 转换为线性比例
    snr_linear = 10**(snr_db / 10)

    # 计算噪声功率
    noise_power = signal_power / snr_linear

    # 生成复高斯噪声
    noise = np.sqrt(noise_power / 2) * (np.random.normal(size=signal.shape) + 1j * np.random.normal(size=signal.shape))

    # 给信号添加噪声
    noisy_signal = signal + noise
    return noisy_signal

def ldpc_qam_awgn(input_signal, snr=11, qam_order=16):
    binary_arr = input_signal

    """
    LDPC信道编码
    """

    # 赋给message_bits作为信道编码的输入
    message_bits = binary_arr

    # LDPC 参数
    n = 1440  # 码字长度
    d_v = 2  # 每个变量节点的度
    d_c = 3  # 每个检查节点的度
    seed = 42  # 随机种子

    # 构建 LDPC 码（生成矩阵 G 和校验矩阵 H）
    H, G = make_ldpc(n, d_v, d_c, systematic=True, sparse=True)

    # 使用 pyldpc 对输入消息进行编码
    ldpc_encoded_bits = encode(G, message_bits, snr=100)

    """
    QAM调制
    """
    # 将 LDPC 编码后的比特流展开为一维
    bits = ldpc_encoded_bits.reshape(-1)

    # 使用自定义 QAM 调制
    modulated_signal = qam_modulate(bits, qam_order)

    cbr = len(modulated_signal) / len(input_signal)


     # 模拟 AWGN 信道
    noisy_signal = awgn_channel(modulated_signal, snr)

    """
    QAM 解调
    """

    # 使用自定义 QAM 解调
    demodulated_bits = qam_demodulate(noisy_signal, qam_order)

    """
    LDPC 解码
    """

    # 将解调后的比特流转换为 LDPC 的解码输入格式
    received_llr = 1 - 2 * demodulated_bits  # 转换为对数似然比（LLR）

    # 使用 pyldpc 解码
    ldpc_decoded_bits = decode(H, received_llr, maxiter=50)  # 使用最多 50 次迭代

    return ldpc_decoded_bits, cbr


if __name__ == "__main__":

    # 获取当前时间
    start_time = datetime.datetime.now()

    # 打印开始时间
    print("Program started at:", start_time)

    # 输入图片数据集根父路径
    input_base_path = './data/kodak'

    # 输入的图片格式(后缀名，区分大小写)
    input_fmt = 'png'

    # 压缩后的JPEG图片数据集根父路径（未进行LDPC+QAM+AWGN）
    output_base_path = './data/clic_origin_jpeg'

    # 图片对应字节比特流txt的目录
    output_txt_path = './data/clic_jpeg_txt'

    # 指定SNR
    snr = 11

    # 输出通过了信道传输的JPEG图片数据集根父路径（经过了LDPC+QAM+AWGN）
    channelcoded_output_base_path = './test_jpeg_out/'+str(snr)+'dB'

    # 目标BPP
    target_bpp = 2

    # 指定QAM调制阶数
    qam_order = 16

    """
    完成整套JPEG压缩
    """
    input_images = glob.glob(os.path.join(input_base_path, '**/*.' + input_fmt), recursive=True)
    total_bpp = 0
    for img_dir in input_images:
        # 加上类别目录
        output_path = output_base_path + '/' + img_dir.split('/')[-2]
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        # 输出图片路径(指定到文件名)
        output_img_path = output_path + '/' + img_dir.split('/')[-1].split('.')[0] + '.JPEG'
        img = Image.open(img_dir)
        img = img.convert("RGB")
        bpp_per_img = find_closest_bpp(target_bpp, img, output_img_path, fmt='JPEG')
        total_bpp += bpp_per_img
        print(bpp_per_img)

    avg_bpp = total_bpp / len(input_images)
    print('平均bpp: {}'.format(avg_bpp))

    """
    对JPEG压缩后的图片读取字节流，并转换为比特流，存入txt
    """
    if not os.path.exists(output_txt_path):
        os.makedirs(output_txt_path)

    input_images = glob.glob(os.path.join(output_base_path, '**/*.JPEG'), recursive=True)
    for img_dir in input_images:
        sub_dir = os.path.basename(os.path.dirname(img_dir))
        test_img_name = img_dir.split('/')[-1].split('.')[0]
        directory = output_txt_path + '/' + sub_dir
        if not os.path.exists(directory):
            os.makedirs(directory)
        file_dir = output_txt_path + '/' + sub_dir + '/' + test_img_name + '.txt'
        img_to_bit(img_dir, file_dir)
        print('{} make txt done'.format(test_img_name))

    print('txt done')

    """
    读取对应txt得到比特流数组，并进行LDPC+QAM+AWGN，最后恢复，得到经过JPEG压缩并且通过了LDPC+QAM+AWGN的图片
    """
    input_txts = glob.glob(os.path.join(output_txt_path, '**/*.txt'), recursive=True)

    total = len(input_txts)
    cbr_array = np.array([])

    for txt_dir in input_txts:
        img_to_bitarray = get_bitarray(txt_dir)
        input_signal = img_to_bitarray
        output_signal, cbr = ldpc_qam_awgn(input_signal, snr=snr, qam_order=qam_order)
        cbr_array = np.append(cbr_array, cbr)
        bitstring = ''
        for i in output_signal:
            bitstring += str(i)
        output_path = channelcoded_output_base_path
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        # 需要找到txt对应原图
        base_dir = output_base_path
        img_dir = base_dir + '/' + txt_dir.split('/')[-2] + '/' + txt_dir.split('/')[-1].split('.')[0] + '.JPEG'
        bit_to_img(bitstring, img_dir, output_path=output_path)
        total = total - 1
        print(total, 'images left')

    average_cbr = np.mean(cbr_array)
    print('average_cbr=', average_cbr)
