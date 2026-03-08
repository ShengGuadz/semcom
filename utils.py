import torch
import torch.nn.functional as F
import os
import numpy as np
# 在utils.py中添加
from pytorch_msssim import ssim  # 需要安装: pip install pytorch-msssim


def get_ssim(image, gt):
    """
    计算SSIM指标
    Args:
        image: 重建图像 tensor [B, C, H, W] or [C, H, W]
        gt: 原始图像 tensor [B, C, H, W] or [C, H, W]
    Returns:
        ssim_value: SSIM值
    """
    # SSIM需要图像在[0,1]范围内
    if image.dim() == 3:
        image = image.unsqueeze(0)
    if gt.dim() == 3:
        gt = gt.unsqueeze(0)
    if image.max() > 1.0:
        image = image / 255.0
    if gt.max() > 1.0:
        gt = gt / 255.0

    ssim_value = ssim(image, gt, data_range=1.0, size_average=True)
    return ssim_value.item()

def image_normalization(norm_type):
    def _inner(tensor: torch.Tensor):
        if norm_type == 'normalization':
            return tensor / 255.0
        elif norm_type == 'denormalization':
            return tensor * 255.0
        else:
            raise Exception('Unknown type of normalization')
    return _inner


def get_psnr(image, gt, max_val=255, mse=None):
    if mse is None:
        mse = F.mse_loss(image, gt)
    mse = torch.tensor(mse) if not isinstance(mse, torch.Tensor) else mse

    psnr = 10 * torch.log10(max_val**2 / mse)
    return psnr


def save_model(model, dir, path):
    os.makedirs(dir, exist_ok=True)
    flag = 1
    while True:
        if os.path.exists(path):
            path = path + '_' + str(flag)
            flag += 1
        else:
            break
    torch.save(model.state_dict(), path)
    print("Model saved in {}".format(path))


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def view_model_param(model):
    total_param = 0

    for param in model.parameters():
        # print(param.data.size())
        total_param += np.prod(list(param.data.size()))
    return total_param
