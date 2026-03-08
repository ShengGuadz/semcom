import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import glob
from torch.quantization import MinMaxObserver, QConfig
from model import DeepJSCC
# 引入我们刚才新建的文件
from quant_model import QuantizableDeepJSCC


def get_dataloader(data_dir, batch_size=1):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])
    image_paths = glob.glob(os.path.join(data_dir, '*.*'))
    if not image_paths: return None

    class SimpleDataset(torch.utils.data.Dataset):
        def __init__(self, paths, transform):
            self.paths = paths
            self.transform = transform

        def __len__(self): return len(self.paths)

        def __getitem__(self, idx):
            img = Image.open(self.paths[idx]).convert('RGB')
            return self.transform(img)

    dataset = SimpleDataset(image_paths, transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def main():
    model_path = "./out/checkpoints/CIFAR10_4_4.0_0.08_AWGN_22h13m53s_on_Jun_07_2024/epoch_998.pkl"
    data_dir = "./data/kodak/"
    device = torch.device('cpu')

    print("1. Loading original Float32 model...")
    original_model = DeepJSCC(c=4, channel_type='AWGN', snr=10)
    if os.path.exists(model_path):
        original_model.load_state_dict(torch.load(model_path, map_location=device))
    original_model.to(device).eval()

    print("2. Preparing Quantization...")
    quantized_model = QuantizableDeepJSCC(original_model)
    quantized_model.to(device).eval()

    backend = 'fbgemm'
    torch.backends.quantized.engine = backend

    # 配置
    quantized_model.qconfig = torch.quantization.get_default_qconfig(backend)
    act_observer = quantized_model.qconfig.activation
    weight_observer_per_tensor = MinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_symmetric)
    quantized_model.decoder.qconfig = QConfig(activation=act_observer, weight=weight_observer_per_tensor)

    torch.quantization.prepare(quantized_model, inplace=True)

    print("3. Calibrating...")
    dataloader = get_dataloader(data_dir)
    with torch.no_grad():
        if dataloader:
            for i, data in enumerate(dataloader):
                if i >= 50: break
                quantized_model(data)

    print("4. Converting...")
    torch.quantization.convert(quantized_model, inplace=True)

    # === 关键修改：只保存参数字典 (state_dict) ===
    save_path = "deepjscc_int8.pth"
    if os.path.exists(save_path): os.remove(save_path)

    # 这里的保存内容变了，不再是 model 对象，而是 model.state_dict()
    print("5. Saving model weights (State Dict)...")
    torch.save(quantized_model.state_dict(), save_path)
    print(f"   Saved weights to {save_path}. Size: {os.path.getsize(save_path) / 1024 / 1024:.2f} MB")


if __name__ == '__main__':
    main()