


# -*- coding: utf-8 -*-
"""
Utility functions for DeepJSCC with CSI Feedback
"""

import torch
import torch.nn as nn
import numpy as np
import random
import os


def image_normalization(norm_type):
    """
    Image normalization/denormalization function
    
    Args:
        norm_type: 'normalization' or 'denormalization'
    
    Returns:
        Normalization function
    """
    def _inner(tensor: torch.Tensor):
        if norm_type == 'normalization':
            return tensor / 255.0
        elif norm_type == 'denormalization':
            return tensor * 255.0
        else:
            raise Exception('Unknown type of normalization')
    return _inner


def set_seed(seed):
    """
    Set random seed for reproducibility
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def save_model(model, save_dir, save_path):
    """
    Save model checkpoint
    
    Args:
        model: Model to save
        save_dir: Directory for saving
        save_path: Full path for the checkpoint file
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")


def load_model(model, load_path, device='cuda'):
    """
    Load model checkpoint
    
    Args:
        model: Model instance to load weights into
        load_path: Path to checkpoint file
        device: Device to load model onto
    
    Returns:
        Model with loaded weights
    """
    model.load_state_dict(torch.load(load_path, map_location=device))
    model.to(device)
    return model


def view_model_param(model):
    """
    Count total trainable parameters in model
    
    Args:
        model: PyTorch model
    
    Returns:
        Total number of trainable parameters
    """
    total_param = 0
    for param in model.parameters():
        if param.requires_grad:
            total_param += param.numel()
    return total_param


def compute_psnr(img1, img2, max_val=255.0):
    """
    Compute Peak Signal-to-Noise Ratio (PSNR)
    
    Args:
        img1: First image tensor
        img2: Second image tensor
        max_val: Maximum pixel value
    
    Returns:
        PSNR value in dB
    """
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    psnr = 20 * torch.log10(max_val / torch.sqrt(mse))
    return psnr.item()


def compute_ssim(img1, img2, window_size=11, size_average=True):
    """
    Compute Structural Similarity Index (SSIM)
    
    Args:
        img1: First image tensor (N, C, H, W)
        img2: Second image tensor (N, C, H, W)
        window_size: Size of the Gaussian window
        size_average: If True, return mean SSIM
    
    Returns:
        SSIM value
    """
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    
    # Create Gaussian window
    def gaussian_window(size, sigma=1.5):
        coords = torch.arange(size, dtype=torch.float32) - size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        return g.unsqueeze(0) * g.unsqueeze(1)
    
    window = gaussian_window(window_size)
    window = window.expand(img1.size(1), 1, window_size, window_size)
    window = window.to(img1.device).type_as(img1)
    
    # Compute means
    mu1 = torch.nn.functional.conv2d(img1, window, padding=window_size//2, groups=img1.size(1))
    mu2 = torch.nn.functional.conv2d(img2, window, padding=window_size//2, groups=img2.size(1))
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    # Compute variances
    sigma1_sq = torch.nn.functional.conv2d(img1 * img1, window, padding=window_size//2, groups=img1.size(1)) - mu1_sq
    sigma2_sq = torch.nn.functional.conv2d(img2 * img2, window, padding=window_size//2, groups=img2.size(1)) - mu2_sq
    sigma12 = torch.nn.functional.conv2d(img1 * img2, window, padding=window_size//2, groups=img1.size(1)) - mu1_mu2
    
    # SSIM formula
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    if size_average:
        return ssim_map.mean().item()
    else:
        return ssim_map.mean(dim=[1, 2, 3])


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class EarlyStopping:
    """Early stopping handler for training"""
    def __init__(self, patience=10, min_delta=0, mode='min'):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as an improvement
            mode: 'min' or 'max' depending on the metric
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'min':
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop
