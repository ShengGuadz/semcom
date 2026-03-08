"""
quant_model.py
包含量化所需的类定义和构建函数。
"""
import torch
import torch.nn as nn
from torch.quantization import MinMaxObserver, QConfig
import copy
from model import DeepJSCC


# ================= 1. 基础组件定义 =================

class NormalizationLayer(nn.Module):
    def __init__(self, P=1):
        super().__init__()
        self.P = P

    def forward(self, z_hat):
        if z_hat.dim() == 4:
            batch_size = z_hat.size()[0]
            k = torch.prod(torch.tensor(z_hat.size()[1:]))
        elif z_hat.dim() == 3:
            batch_size = 1
            k = torch.prod(torch.tensor(z_hat.size()))
        else:
            raise ValueError('Unknown size of input')
        z_temp = z_hat.reshape(batch_size, 1, 1, -1)
        z_trans = z_hat.reshape(batch_size, 1, -1, 1)
        numer_factor = self.P * k
        denom_power = z_temp @ z_trans
        tensor = torch.sqrt(numer_factor) * z_hat / torch.sqrt(denom_power)
        if batch_size == 1 and z_hat.dim() == 3:
            return tensor.squeeze(0)
        return tensor


class QuantizableConvWithPReLU(nn.Module):
    def __init__(self, original_module=None):
        super().__init__()
        # 如果传入了原始模块，就复制参数；否则创建空层（用于占位）
        if original_module:
            self.conv = copy.deepcopy(original_module.conv)
            self.prelu = copy.deepcopy(original_module.prelu)
        else:
            # 这里的参数仅为占位，后续会被convert替换或load_state_dict覆盖
            self.conv = nn.Conv2d(1, 1, 1)
            self.prelu = nn.PReLU()

        self.dequant = torch.quantization.DeQuantStub()
        self.quant = torch.quantization.QuantStub()

    def forward(self, x):
        x = self.conv(x)
        x = self.dequant(x)
        x = self.prelu(x)
        x = self.quant(x)
        return x


class QuantizableTransConvWithPReLU(nn.Module):
    def __init__(self, original_module=None, is_output_layer=False):
        super().__init__()
        if original_module:
            self.transconv = copy.deepcopy(original_module.transconv)
            if isinstance(original_module.activate, nn.PReLU):
                self.activate = copy.deepcopy(original_module.activate)
            else:
                self.activate = type(original_module.activate)()
        else:
            self.transconv = nn.ConvTranspose2d(1, 1, 1)
            self.activate = nn.PReLU()

        self.is_output_layer = is_output_layer
        self.dequant = torch.quantization.DeQuantStub()
        self.quant = torch.quantization.QuantStub()

    def forward(self, x):
        x = self.transconv(x)
        x = self.dequant(x)
        x = self.activate(x)
        if not self.is_output_layer:
            x = self.quant(x)
        return x


# ================= 2. 包装器定义 =================

class QuantizableEncoder(nn.Module):
    def __init__(self, original_encoder):
        super().__init__()
        self.is_temp = original_encoder.is_temp
        self.norm = NormalizationLayer(P=1)
        self.quant_input = torch.quantization.QuantStub()
        self.conv1 = QuantizableConvWithPReLU(original_encoder.conv1)
        self.conv2 = QuantizableConvWithPReLU(original_encoder.conv2)
        self.conv3 = QuantizableConvWithPReLU(original_encoder.conv3)
        self.conv4 = QuantizableConvWithPReLU(original_encoder.conv4)
        self.conv5 = QuantizableConvWithPReLU(original_encoder.conv5)
        self.dequant_output = torch.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant_input(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        if not self.is_temp:
            x = self.conv5(x)
            x = self.dequant_output(x)
            x = self.norm(x)
        return x


class QuantizableDecoder(nn.Module):
    def __init__(self, original_decoder):
        super().__init__()
        self.quant_input = torch.quantization.QuantStub()
        self.tconv1 = QuantizableTransConvWithPReLU(original_decoder.tconv1)
        self.tconv2 = QuantizableTransConvWithPReLU(original_decoder.tconv2)
        self.tconv3 = QuantizableTransConvWithPReLU(original_decoder.tconv3)
        self.tconv4 = QuantizableTransConvWithPReLU(original_decoder.tconv4)
        self.tconv5 = QuantizableTransConvWithPReLU(original_decoder.tconv5, is_output_layer=True)

    def forward(self, x):
        x = self.quant_input(x)
        x = self.tconv1(x)
        x = self.tconv2(x)
        x = self.tconv3(x)
        x = self.tconv4(x)
        x = self.tconv5(x)
        return x


class QuantizableDeepJSCC(nn.Module):
    def __init__(self, original_model):
        super().__init__()
        self.encoder = QuantizableEncoder(original_model.encoder)
        self.channel = original_model.channel
        self.decoder = QuantizableDecoder(original_model.decoder)

    def forward(self, x):
        z = self.encoder(x)
        if hasattr(self, 'channel') and self.channel is not None:
            z = self.channel(z)
        x_hat = self.decoder(z)
        return x_hat


# ================= 3. 核心：构建 Int8 模型结构的函数 =================

def create_int8_model_structure(device='cpu'):
    """
    不依赖任何外部文件，凭空构建一个已经 Convert 过的 Int8 模型骨架。
    用于加载 state_dict。
    """
    # 1. 创建一个临时的 FP32 模型用于提供结构
    dummy_model = DeepJSCC(c=4, channel_type='AWGN', snr=10)

    # 2. 包装成量化结构
    quant_model = QuantizableDeepJSCC(dummy_model)
    quant_model.to(device)
    quant_model.eval()

    # 3. 设置量化配置 (必须与量化脚本中一致)
    backend = 'fbgemm'
    torch.backends.quantized.engine = backend
    quant_model.qconfig = torch.quantization.get_default_qconfig(backend)

    # 特殊处理 Decoder
    act_observer = quant_model.qconfig.activation
    weight_observer_per_tensor = MinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_symmetric)
    quant_model.decoder.qconfig = QConfig(activation=act_observer, weight=weight_observer_per_tensor)

    # 4. 执行 Prepare (插入观测者)
    torch.quantization.prepare(quant_model, inplace=True)

    # 5. 执行 Convert (转换为 Int8 结构)
    # 这一步会将 Conv2d 替换为 QuantizedConv2d，这样才能匹配 Int8 的权重文件
    torch.quantization.convert(quant_model, inplace=True)

    return quant_model