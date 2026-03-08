# -*- coding: utf-8 -*-
"""
Training Script for DeepJSCC with CSI Feedback

This script implements end-to-end training for:
1. Image encoding/decoding (main task)
2. CSI compression/decompression (auxiliary task)
3. Joint optimization with combined loss
"""

import os
import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import time
import glob
import argparse
from fractions import Fraction

from model_csi import DeepJSCCWithCSIFeedback, DeepJSCCWithAdaptiveCSIFeedback, ratio2filtersize
from utils_csi import (
    image_normalization, set_seed, save_model, view_model_param,
    compute_psnr, AverageMeter, EarlyStopping
)

try:
    from tensorboardX import SummaryWriter
except ImportError:
    from torch.utils.tensorboard import SummaryWriter


def train_epoch(model, optimizer, param, data_loader):
    """
    Train for one epoch with joint CSI optimization
    """
    model.train()
    
    img_loss_meter = AverageMeter()
    csi_loss_meter = AverageMeter()
    total_loss_meter = AverageMeter()
    
    for images, _ in data_loader:
        images = images.cuda() if param['parallel'] and torch.cuda.device_count() > 1 else images.to(param['device'])
        
        optimizer.zero_grad()
        
        # Forward pass with intermediate outputs
        outputs, csi_orig, csi_comp, csi_recon = model.forward(images, return_intermediate=True)
        
        # Denormalize for loss computation
        outputs_denorm = image_normalization('denormalization')(outputs)
        images_denorm = image_normalization('denormalization')(images)
        
        # Compute joint loss
        if param['joint_training']:
            total_loss, img_loss, csi_loss = model.loss(
                images_denorm, outputs_denorm, csi_orig, csi_recon
            ) if not param['parallel'] else model.module.loss(
                images_denorm, outputs_denorm, csi_orig, csi_recon
            )
            csi_loss_meter.update(csi_loss.item(), images.size(0))
        else:
            total_loss = model.loss(images_denorm, outputs_denorm) if not param['parallel'] else \
                         model.module.loss(images_denorm, outputs_denorm)
            img_loss = total_loss
        
        total_loss.backward()
        optimizer.step()
        
        img_loss_meter.update(img_loss.item() if hasattr(img_loss, 'item') else img_loss, images.size(0))
        total_loss_meter.update(total_loss.item(), images.size(0))
    
    return {
        'total_loss': total_loss_meter.avg,
        'img_loss': img_loss_meter.avg,
        'csi_loss': csi_loss_meter.avg if param['joint_training'] else 0
    }


def evaluate_epoch(model, param, data_loader):
    """
    Evaluate model on validation/test set
    """
    model.eval()
    
    img_loss_meter = AverageMeter()
    csi_loss_meter = AverageMeter()
    psnr_meter = AverageMeter()
    
    with torch.no_grad():
        for images, _ in data_loader:
            images = images.cuda() if param['parallel'] and torch.cuda.device_count() > 1 else images.to(param['device'])
            
            outputs, csi_orig, csi_comp, csi_recon = model.forward(images, return_intermediate=True)
            
            outputs_denorm = image_normalization('denormalization')(outputs)
            images_denorm = image_normalization('denormalization')(images)
            
            # Compute losses
            if param['joint_training']:
                _, img_loss, csi_loss = model.loss(
                    images_denorm, outputs_denorm, csi_orig, csi_recon
                ) if not param['parallel'] else model.module.loss(
                    images_denorm, outputs_denorm, csi_orig, csi_recon
                )
                csi_loss_meter.update(csi_loss.item(), images.size(0))
            else:
                img_loss = model.loss(images_denorm, outputs_denorm) if not param['parallel'] else \
                           model.module.loss(images_denorm, outputs_denorm)
            
            img_loss_meter.update(img_loss.item() if hasattr(img_loss, 'item') else img_loss, images.size(0))
            
            # Compute PSNR
            psnr = compute_psnr(outputs_denorm, images_denorm, max_val=255.0)
            psnr_meter.update(psnr, images.size(0))
    
    return {
        'img_loss': img_loss_meter.avg,
        'csi_loss': csi_loss_meter.avg if param['joint_training'] else 0,
        'psnr': psnr_meter.avg
    }


def train_pipeline(params):
    """
    Main training pipeline for DeepJSCC with CSI Feedback
    """
    dataset_name = params['dataset']
    
    # Load dataset
    if dataset_name == 'cifar10':
        transform = transforms.Compose([transforms.ToTensor()])
        train_dataset = datasets.CIFAR10(root='../dataset/', train=True, download=True, transform=transform)
        train_loader = DataLoader(train_dataset, shuffle=True, batch_size=params['batch_size'], 
                                  num_workers=params['num_workers'])
        test_dataset = datasets.CIFAR10(root='../dataset/', train=False, download=True, transform=transform)
        test_loader = DataLoader(test_dataset, shuffle=False, batch_size=params['batch_size'], 
                                 num_workers=params['num_workers'])
    elif dataset_name == 'imagenet':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((128, 128))])
        train_dataset = datasets.ImageFolder(root='../dataset/ImageNet/train', transform=transform)
        train_loader = DataLoader(train_dataset, shuffle=True, batch_size=params['batch_size'], 
                                  num_workers=params['num_workers'])
        test_dataset = datasets.ImageFolder(root='../dataset/ImageNet/val', transform=transform)
        test_loader = DataLoader(test_dataset, shuffle=False, batch_size=params['batch_size'], 
                                 num_workers=params['num_workers'])
    else:
        raise Exception('Unknown dataset')
    
    # Calculate filter size
    image_first = train_dataset[0][0]
    c = ratio2filtersize(image_first, params['ratio'])
    
    print(f"SNR: {params['snr']} dB, Inner channel: {c}, Ratio: {params['ratio']:.2f}")
    print(f"Feedback bits: {params['feedback_bits']}, CSI loss weight: {params['csi_loss_weight']}")
    
    # Create model
    if params['adaptive_feedback']:
        model = DeepJSCCWithAdaptiveCSIFeedback(
            c=c,
            channel_type=params['channel'],
            snr=params['snr'],
            feedback_bits_list=params['feedback_bits_list'],
            use_csi_aware_decoder=params['use_csi_aware_decoder'],
            csi_loss_weight=params['csi_loss_weight']
        )
    else:
        model = DeepJSCCWithCSIFeedback(
            c=c,
            channel_type=params['channel'],
            snr=params['snr'],
            feedback_bits=params['feedback_bits'],
            use_csi_aware_decoder=params['use_csi_aware_decoder'],
            csi_loss_weight=params['csi_loss_weight']
        )
    
    print(f"Total parameters: {view_model_param(model):,}")
    
    # Setup directories
    out_dir = params['out_dir']
    phaser = f"{dataset_name.upper()}_{c}_{params['snr']}_{params['ratio']:.2f}_{params['channel']}_fb{params['feedback_bits']}_{time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')}"
    root_log_dir = os.path.join(out_dir, 'logs', phaser)
    root_ckpt_dir = os.path.join(out_dir, 'checkpoints', phaser)
    root_config_dir = os.path.join(out_dir, 'configs', phaser)
    
    writer = SummaryWriter(log_dir=root_log_dir)
    
    # Device setup
    device = torch.device(params['device'] if torch.cuda.is_available() else 'cpu')
    if params['parallel'] and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        model = model.cuda()
    else:
        model = model.to(device)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=params['init_lr'], weight_decay=params['weight_decay'])
    
    # Scheduler
    if params['ReduceLROnPlateau']:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=params['lr_reduce_factor'],
            patience=params['lr_schedule_patience'], verbose=True
        )
    elif params['if_scheduler']:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=params['step_size'], gamma=params['gamma'])
    else:
        scheduler = None
    
    # Early stopping
    early_stopping = EarlyStopping(patience=params.get('early_stop_patience', 150), mode='min')
    
    writer.add_text('config', str(params))
    t0 = time.time()
    best_val_loss = float('inf')
    
    # Training loop
    try:
        with tqdm(range(params['epochs']), disable=params['disable_tqdm']) as t:
            for epoch in t:
                t.set_description(f'Epoch {epoch}')
                start = time.time()
                
                # Train
                train_metrics = train_epoch(model, optimizer, params, train_loader)
                
                # Validate
                val_metrics = evaluate_epoch(model, params, test_loader)
                
                # Logging
                writer.add_scalar('train/total_loss', train_metrics['total_loss'], epoch)
                writer.add_scalar('train/img_loss', train_metrics['img_loss'], epoch)
                writer.add_scalar('train/csi_loss', train_metrics['csi_loss'], epoch)
                writer.add_scalar('val/img_loss', val_metrics['img_loss'], epoch)
                writer.add_scalar('val/csi_loss', val_metrics['csi_loss'], epoch)
                writer.add_scalar('val/psnr', val_metrics['psnr'], epoch)
                writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)
                
                t.set_postfix(
                    time=f"{time.time() - start:.1f}s",
                    lr=f"{optimizer.param_groups[0]['lr']:.2e}",
                    train_loss=f"{train_metrics['total_loss']:.4f}",
                    val_loss=f"{val_metrics['img_loss']:.4f}",
                    psnr=f"{val_metrics['psnr']:.2f}"
                )
                
                # Save checkpoint
                if not os.path.exists(root_ckpt_dir):
                    os.makedirs(root_ckpt_dir)
                
                if val_metrics['img_loss'] < best_val_loss:
                    best_val_loss = val_metrics['img_loss']
                    torch.save(model.state_dict(), os.path.join(root_ckpt_dir, 'best_model.pkl'))
                
                torch.save(model.state_dict(), os.path.join(root_ckpt_dir, f'epoch_{epoch}.pkl'))
                
                # Clean old checkpoints
                files = glob.glob(os.path.join(root_ckpt_dir, 'epoch_*.pkl'))
                for file in files:
                    epoch_nb = int(file.split('_')[-1].split('.')[0])
                    if epoch_nb < epoch - 1:
                        os.remove(file)
                
                # Scheduler step
                if params['ReduceLROnPlateau'] and scheduler is not None:
                    scheduler.step(val_metrics['img_loss'])
                elif params['if_scheduler'] and scheduler is not None:
                    scheduler.step()
                
                # Early stopping
                if early_stopping(val_metrics['img_loss']):
                    print(f"\nEarly stopping at epoch {epoch}")
                    break
                
                if optimizer.param_groups[0]['lr'] < params['min_lr']:
                    print("\nLR reached minimum, stopping.")
                    break
                
                # if time.time() - t0 > params['max_time'] * 3600:
                #     print(f"\nMax training time ({params['max_time']}h) reached.")
                #     break
    
    except KeyboardInterrupt:
        print('\nTraining interrupted by user.')
    
    # Final evaluation
    test_metrics = evaluate_epoch(model, params, test_loader)
    train_metrics = evaluate_epoch(model, params, train_loader)
    
    print(f"\nFinal Results:")
    print(f"Test Loss: {test_metrics['img_loss']:.4f}, PSNR: {test_metrics['psnr']:.2f} dB")
    print(f"Train Loss: {train_metrics['img_loss']:.4f}, PSNR: {train_metrics['psnr']:.2f} dB")
    print(f"Total time: {(time.time() - t0) / 3600:.2f} hours")
    
    writer.close()
    
    # Save config
    if not os.path.exists(os.path.dirname(root_config_dir)):
        os.makedirs(os.path.dirname(root_config_dir))
    
    import yaml
    with open(root_config_dir + '.yaml', 'w') as f:
        yaml.dump({
            'dataset': dataset_name,
            'params': params,
            'inner_channel': c,
            'total_parameters': view_model_param(model),
            'test_loss': test_metrics['img_loss'],
            'test_psnr': test_metrics['psnr']
        }, f)


def config_parser():
    parser = argparse.ArgumentParser(description='DeepJSCC with CSI Feedback Training')
    
    # Dataset
    parser.add_argument('--dataset', default='cifar10', type=str, choices=['cifar10', 'imagenet'])
    parser.add_argument('--out', default='./out', type=str, help='Output directory')
    
    # Training
    parser.add_argument('--epochs', default=1000, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--init_lr', default=1e-3, type=float)
    parser.add_argument('--weight_decay', default=5e-4, type=float)
    parser.add_argument('--seed', default=42, type=int)
    
    # Channel
    # parser.add_argument('--channel', default='AWGN', type=str, choices=['AWGN', 'Rayleigh'])
    parser.add_argument('--channel', default=['AWGN', 'Rayleigh'], nargs='+', type=str, help='Channel types')
    # parser.add_argument('--snr_list', default=['19', '13', '7', '4', '1'], nargs='+')
    parser.add_argument('--snr_list', default=[ '7', '4', '1'], nargs='+')
    parser.add_argument('--ratio_list', default=['1/6', '1/12'], nargs='+')
    
    # CSI Feedback
    parser.add_argument('--feedback_bits', default=32, type=int, help='Number of feedback bits')
    parser.add_argument('--csi_loss_weight', default=0.1, type=float, help='Weight for CSI loss')
    parser.add_argument('--use_csi_aware_decoder', default=True, type=bool)
    parser.add_argument('--joint_training', default=True, type=bool, help='Enable joint CSI training')
    parser.add_argument('--adaptive_feedback', default=False, type=bool, help='Use adaptive feedback rate')
    parser.add_argument('--feedback_bits_list', default=[16, 32, 64], nargs='+', type=int)
    
    # Misc
    parser.add_argument('--device', default='cuda:0', type=str)
    parser.add_argument('--parallel', default=False, type=bool)
    parser.add_argument('--disable_tqdm', default=False, type=bool)
    parser.add_argument('--num_workers', default=4, type=int)
    
    return parser.parse_args()


def main():
    args = config_parser()
    
    # Convert string lists
    args.snr_list = list(map(float, args.snr_list))
    args.ratio_list = list(map(lambda x: float(Fraction(x)), args.ratio_list))
    
    set_seed(args.seed)
    
    # Build params dict
    params = {
        'dataset': args.dataset,
        'out_dir': args.out,
        'batch_size': args.batch_size,
        'num_workers': args.num_workers,
        'epochs': args.epochs,
        'init_lr': args.init_lr,
        'weight_decay': args.weight_decay,
        'device': args.device,
        'parallel': args.parallel,
        'disable_tqdm': args.disable_tqdm,
        'channel': args.channel,
        'feedback_bits': args.feedback_bits,
        'csi_loss_weight': args.csi_loss_weight,
        'use_csi_aware_decoder': args.use_csi_aware_decoder,
        'joint_training': args.joint_training,
        'adaptive_feedback': args.adaptive_feedback,
        'feedback_bits_list': args.feedback_bits_list,
        'if_scheduler': True,
        'step_size': 640,
        'gamma': 0.1,
        'ReduceLROnPlateau': False,
        'lr_reduce_factor': 0.5,
        'lr_schedule_patience': 15,
        'max_time': 12,
        'min_lr': 1e-5,
        'seed': args.seed
    }
    
    print("=" * 60)
    print("DeepJSCC with CSI Feedback Training")
    print("=" * 60)
    
    # for ratio in args.ratio_list:
    #     for snr in args.snr_list:
    #         params['ratio'] = ratio
    #         params['snr'] = snr
    #         print(f"\nTraining: SNR={snr}dB, Ratio={ratio:.4f}")
    #         train_pipeline(params)
    # 确保 args.channel 是列表 (防止用户只输入了一个字符串)
    if isinstance(args.channel, str):
        args.channel = [args.channel]

    # === 新增：最外层遍历 Channel ===
    for channel in args.channel:
        for ratio in args.ratio_list:
            for snr in args.snr_list:
                # 这一步很关键：把当前的单一信道字符串(如 'AWGN')传给 params
                params['channel'] = channel
                params['ratio'] = ratio
                params['snr'] = snr

                print(f"\nTraining: Channel={channel}, SNR={snr}dB, Ratio={ratio:.4f}")
                train_pipeline(params)

if __name__ == '__main__':
    main()
