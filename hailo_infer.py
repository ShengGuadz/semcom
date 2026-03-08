import numpy as np
import hailo_platform as hpf
import signal
import sys
import atexit
import time

class HailoInference:
    def __init__(self, hef_path):
        self.hef_path = hef_path
        self.target = None
        self.network_group = None
        self.infer_pipeline = None
        self.input_vstream_info = None
        self.output_vstream_info = None
        self.input_shape = None
        self.output_shape = None
        self.network_group_context = None
        self._is_closed = False

    def load_model(self):
        """加载模型"""
        hef = hpf.HEF(self.hef_path)
        self.target = hpf.VDevice()

        configure_params = hpf.ConfigureParams.create_from_hef(hef, interface=hpf.HailoStreamInterface.PCIe)
        self.network_group = self.target.configure(hef, configure_params)[0]
        network_group_params = self.network_group.create_params()

        self.input_vstream_info = hef.get_input_vstream_infos()[0]
        self.output_vstream_info = hef.get_output_vstream_infos()[0]

        input_vstreams_params = hpf.InputVStreamParams.make_from_network_group(
            self.network_group, quantized=False, format_type=hpf.FormatType.FLOAT32)
        output_vstreams_params = hpf.OutputVStreamParams.make_from_network_group(
            self.network_group, quantized=False, format_type=hpf.FormatType.FLOAT32)

        self.input_shape = self.input_vstream_info.shape
        self.output_shape = self.output_vstream_info.shape

        print(f"Hailo HEF Input shape: {self.input_shape}, Output shape: {self.output_shape}")

        # 正确激活网络组并保存上下文管理器
        self.network_group_context = self.network_group.activate(network_group_params)
        self.network_group_context.__enter__()

        self.infer_pipeline = hpf.InferVStreams(
            self.network_group, input_vstreams_params, output_vstreams_params)
        self.infer_pipeline.__enter__()

        # 注册清理函数
        atexit.register(self._cleanup_handler)

        return self

    def infer(self, image_data):
        """执行推理"""
        if self._is_closed:
            raise RuntimeError("Inference engine has been closed")
        if not self.infer_pipeline:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # 确保输入数据维度正确
        if image_data.shape != tuple(self.input_shape):
            raise ValueError(f"Input shape mismatch. Expected {self.input_shape}, got {image_data.shape}")

        input_data = {self.input_vstream_info.name: np.expand_dims(image_data, axis=0)}
        results = self.infer_pipeline.infer(input_data)
        output_data = results[self.output_vstream_info.name]
        return output_data

    def _cleanup_handler(self):
        """由atexit调用的清理函数"""
        self.close()

    def close(self):
        """释放资源"""
        if self._is_closed:
            return

        print("Cleaning up Hailo resources...")

        try:
            if self.infer_pipeline:
                self.infer_pipeline.__exit__(None, None, None)
        except Exception as e:
            print(f"Error closing infer_pipeline: {e}")

        try:
            if self.network_group_context:
                self.network_group_context.__exit__(None, None, None)
        except Exception as e:
            print(f"Error deactivating network group: {e}")

        try:
            if self.target:
                self.target.__exit__(None, None, None)
        except Exception as e:
            print(f"Error closing target: {e}")

        self._is_closed = True
        self.infer_pipeline = None
        self.network_group_context = None
        self.target = None
        print("Hailo resources cleaned up successfully")

# 全局实例
inference_engine = None

def initialize_model(model_path = "deepjscc_encoder_optimized.hef"):
    """在某个地方初始化模型"""
    global inference_engine
    try:
        inference_engine = HailoInference(model_path)
        inference_engine.load_model()
        print("Model loaded successfully")
        return True
    except Exception as e:
        print(f"Failed to load model: {e}")
        return False

def run_inference(image_data):
    """在另一个地方执行推理"""
    global inference_engine
    if inference_engine is None:
        raise RuntimeError("Model not initialized. Call initialize_model() first.")

    try:
        result = inference_engine.infer(image_data)
        return result
    except Exception as e:
        print(f"Inference failed: {e}")
        raise

def cleanup_model():
    """在程序结束时清理资源"""
    global inference_engine
    if inference_engine:
        inference_engine.close()
        inference_engine = None

def static_power_normalization(x, k, P: float = 1.0):
    """
    静态功率归一化 - 避免动态形状计算
    使用预计算的维度参数

    参数:
    x : numpy.ndarray - 输入数据，形状为 (B, 2*c, H, W)
    k : float - 预计算的k值
    P : float - 预计算的P值

    返回:
    x_normalized : numpy.ndarray - 归一化后的数据
    """
    # 计算信号功率 - 在通道、高、宽维度上求和
    # x.shape = (B, 2*c, H, W)
    signal_power = np.sum(x * x, axis=(1, 2, 3), keepdims=True)  # (B, 1, 1, 1)

    # 使用预计算的k值和P值
    k_val = np.array(k, dtype=x.dtype)
    P_val = np.array(P, dtype=x.dtype)
    eps = np.array(1e-8, dtype=x.dtype)

    # 归一化因子
    norm_factor = np.sqrt(P_val * k_val / (signal_power + eps))

    # 应用归一化
    x_normalized = x * norm_factor

    return x_normalized

# 设置信号处理器确保强制退出时清理资源
def signal_handler(sig, frame):
    print('\nReceived interrupt signal. Cleaning up...')
    cleanup_model()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# 使用示例：
# 在一个模块或函数中加载模型
# success = initialize_model()
# if success:
#     # 在另一个模块或函数中执行推理
#     for i in range(2):
#         image = np.random.rand(*inference_engine.input_shape).astype(np.float32)
#         result = run_inference(image)
#         print(f"Inference {i+1} result type: {type(result)}")
#
#     # 在程序结束时清理
#     cleanup_model()
if __name__ == "__main__":
    success = initialize_model()
    if not success:
        print("Failed to initialize model")
        exit(1)

    print("Model loaded and ready for inference")


    for i in range(5):
        # 准备图像数据 - 确保形状与模型输入匹配
        time.sleep(1)
        image = np.random.rand(*inference_engine.input_shape).astype(np.float32)

        # 或者使用实际图像数据
        # image = load_your_image()  # 您的图像加载函数
        # image = preprocess_image(image)  # 您的预处理函数

        try:
            result = run_inference(image)
            print(f"Inference {i+1} completed, result shape: {result.shape}")
        except Exception as e:
            print(f"Error during inference: {e}")
            break
