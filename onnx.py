import torch
import torch.nn as nn
from model import DeepJSCC

# ================= 配置 =================
# 模型权重路径
CHECKPOINT_PATH = "./out/checkpoints/CIFAR10_4_4.0_0.08_AWGN_22h13m53s_on_Jun_07_2024/epoch_998.pkl"
# 输出 ONNX 路径
ONNX_PATH = "deepjscc_encoder_512x768.onnx"
# 输入分辨率 (H, W) - 建议固定一个，另一方向靠代码旋转
INPUT_SHAPE = (1, 3, 512, 768)
# 通道数 (根据文件名 CIFAR10_4... 推断 c=4)
C_CHANNEL = 4


# =======================================

def export_encoder():
    print(f"Loading model from {CHECKPOINT_PATH}...")

    # 1. 初始化模型
    # 注意：SNR 参数在推理阶段不影响 Encoder 权重，只影响 Channel 模拟
    model = DeepJSCC(c=C_CHANNEL, channel_type='AWGN', snr=10)

    # 2. 加载权重
    # map_location 确保即使没有 GPU 也能加载
    state_dict = torch.load(CHECKPOINT_PATH, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()

    # 3. 提取 Encoder
    encoder = model.encoder

    # [关键步骤] 修改 Encoder 的 forward 方法
    # 原模型在 forward 里做了 self.norm(x)。
    # 我们需要它输出 conv5 的结果，交给 Python 代码做归一化。
    def forward_without_norm(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        # 对应 model.py 中的逻辑: if not self.is_temp: x = self.conv5(x)
        if not self.is_temp:
            x = self.conv5(x)
        # 这里直接返回 x，跳过 self.norm(x)
        return x

    # 动态替换实例的方法 (Monkey Patching)
    import types
    encoder.forward = types.MethodType(forward_without_norm, encoder)
    print("Patched encoder to skip internal normalization layer.")

    # 4. 导出 ONNX
    dummy_input = torch.randn(*INPUT_SHAPE)

    print(f"Exporting ONNX with input shape {INPUT_SHAPE}...")
    torch.onnx.export(
        encoder,
        dummy_input,
        ONNX_PATH,
        export_params=True,
        opset_version=11,  # Hailo 对 opset 11 支持较好
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output_feature'],
        dynamic_axes=None  # 这里的 HEF 需要固定维度，不要开 dynamic
    )
    print(f"Successfully exported to {ONNX_PATH}")


if __name__ == "__main__":
    export_encoder()