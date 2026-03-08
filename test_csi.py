# -*- coding: utf-8 -*-
"""
Test Script for DeepJSCC with CSI Feedback on Kodak Dataset

This script evaluates trained models across different SNR values
and generates PSNR/SSIM performance curves.
"""

import os
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
import json

from model_csi import DeepJSCCWithCSIFeedback, DeepJSCC, ratio2filtersize
from utils_csi import compute_psnr, compute_ssim, set_seed


class KodakDataset(Dataset):
    """
    Kodak Dataset loader

    The Kodak dataset contains 24 high-quality images (768x512 or 512x768)
    Download from: http://r0k.us/graphics/kodak/
    """

    def __init__(self, root_dir, transform=None, crop_size=None):
        self.root_dir = root_dir
        self.transform = transform
        self.crop_size = crop_size

        self.image_files = []
        valid_extensions = ['.png', '.jpg', '.jpeg', '.bmp']

        if os.path.exists(root_dir):
            for f in sorted(os.listdir(root_dir)):
                if any(f.lower().endswith(ext) for ext in valid_extensions):
                    self.image_files.append(os.path.join(root_dir, f))

        if len(self.image_files) == 0:
            print(f"Warning: No images found in {root_dir}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')

        if self.crop_size is not None:
            w, h = image.size
            new_h, new_w = self.crop_size
            left = (w - new_w) // 2
            top = (h - new_h) // 2
            image = image.crop((left, top, left + new_w, top + new_h))

        if self.transform:
            image = self.transform(image)

        return image, os.path.basename(img_path)


def evaluate_model(model, dataloader, device, snr_list, channel_type='AWGN'):
    """Evaluate model across different SNR values"""
    model.eval()
    results = {snr: {'psnr': [], 'ssim': []} for snr in snr_list}

    with torch.no_grad():
        for snr in tqdm(snr_list, desc='Testing SNR values'):
            if hasattr(model, 'change_channel'):
                model.change_channel(channel_type=channel_type, snr=snr)
            elif hasattr(model, 'channel'):
                model.channel.snr = snr

            psnr_list = []
            ssim_list = []

            for images, _ in dataloader:
                images = images.to(device)

                if isinstance(model, DeepJSCCWithCSIFeedback):
                    outputs = model(images, return_intermediate=False)
                else:
                    outputs = model(images)

                images_denorm = images * 255.0
                outputs_denorm = torch.clamp(outputs * 255.0, 0, 255)

                for i in range(images.size(0)):
                    img_gt = images_denorm[i:i + 1]
                    img_pred = outputs_denorm[i:i + 1]

                    psnr = compute_psnr(img_pred, img_gt, max_val=255.0)
                    ssim = compute_ssim(img_pred, img_gt)

                    psnr_list.append(psnr)
                    ssim_list.append(ssim)

            results[snr]['psnr'] = np.mean(psnr_list)
            results[snr]['ssim'] = np.mean(ssim_list)
            results[snr]['psnr_std'] = np.std(psnr_list)
            results[snr]['ssim_std'] = np.std(ssim_list)

    return results


def plot_results(results_dict, save_path='./results', title_suffix=''):
    """Plot PSNR and SSIM curves"""
    os.makedirs(save_path, exist_ok=True)

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    markers = ['o', 's', '^', 'D', 'v', 'p']

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot PSNR
    ax1 = axes[0]
    for idx, (model_name, results) in enumerate(results_dict.items()):
        snrs = sorted(results.keys())
        psnrs = [results[snr]['psnr'] for snr in snrs]
        psnr_stds = [results[snr].get('psnr_std', 0) for snr in snrs]

        ax1.errorbar(snrs, psnrs, yerr=psnr_stds, label=model_name,
                     color=colors[idx % len(colors)],
                     marker=markers[idx % len(markers)],
                     markersize=8, linewidth=2, capsize=3)

    ax1.set_xlabel('SNR (dB)', fontsize=12)
    ax1.set_ylabel('PSNR (dB)', fontsize=12)
    ax1.set_title(f'PSNR vs SNR {title_suffix}', fontsize=14)
    ax1.legend(loc='lower right', fontsize=10)
    ax1.grid(True, linestyle='--', alpha=0.7)

    # Plot SSIM
    ax2 = axes[1]
    for idx, (model_name, results) in enumerate(results_dict.items()):
        snrs = sorted(results.keys())
        ssims = [results[snr]['ssim'] for snr in snrs]
        ssim_stds = [results[snr].get('ssim_std', 0) for snr in snrs]

        ax2.errorbar(snrs, ssims, yerr=ssim_stds, label=model_name,
                     color=colors[idx % len(colors)],
                     marker=markers[idx % len(markers)],
                     markersize=8, linewidth=2, capsize=3)

    ax2.set_xlabel('SNR (dB)', fontsize=12)
    ax2.set_ylabel('SSIM', fontsize=12)
    ax2.set_title(f'SSIM vs SNR {title_suffix}', fontsize=14)
    ax2.legend(loc='lower right', fontsize=10)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.set_ylim([0, 1])

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'performance_curves{title_suffix.replace(" ", "_")}.png'),
                dpi=300, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Test DeepJSCC with CSI Feedback')
    parser.add_argument('--kodak_path', type=str, default='./dataset/kodak')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--c', type=int, default=16)
    parser.add_argument('--feedback_bits', type=int, default=32)
    parser.add_argument('--channel', type=str, default='AWGN', choices=['AWGN', 'Rayleigh'])
    parser.add_argument('--snr_list', type=float, nargs='+', default=[1, 4, 7, 10, 13, 16, 19])
    parser.add_argument('--crop_size', type=int, nargs=2, default=[512, 512])
    parser.add_argument('--save_path', type=str, default='./test_results')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()
    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.save_path, exist_ok=True)

    # Load dataset
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = KodakDataset(args.kodak_path, transform, tuple(args.crop_size))
    dataloader = DataLoader(dataset, batch_size=1,num_workers=0, shuffle=False)

    all_results = {}

    # Test models with different configurations
    for fb_bits in [16, 32, 64]:
        model = DeepJSCCWithCSIFeedback(
            c=args.c, channel_type=args.channel, snr=args.snr_list[0],
            feedback_bits=fb_bits, use_csi_aware_decoder=True
        )
        if args.checkpoint:
            model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        model = model.to(device)

        results = evaluate_model(model, dataloader, device, args.snr_list, args.channel)
        all_results[f'DeepJSCC-CSI-{fb_bits}bits'] = results

    # Test baseline
    model_baseline = DeepJSCC(c=args.c, channel_type=args.channel, snr=args.snr_list[0])
    model_baseline = model_baseline.to(device)
    results_baseline = evaluate_model(model_baseline, dataloader, device, args.snr_list, args.channel)
    all_results['DeepJSCC-Baseline'] = results_baseline

    # Save and plot results
    with open(os.path.join(args.save_path, 'results.json'), 'w') as f:
        json.dump({k: {str(snr): v for snr, v in res.items()}
                   for k, res in all_results.items()}, f, indent=2)

    plot_results(all_results, args.save_path, f'({args.channel} Channel)')
    print(f"Results saved to {args.save_path}")


if __name__ == '__main__':
    main()