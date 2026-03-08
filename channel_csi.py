# -*- coding: utf-8 -*-
"""
Extended Channel module with CSI generation and feedback support
"""

import torch
import torch.nn as nn
import numpy as np


class Channel(nn.Module):
    """
    Channel model supporting AWGN and Rayleigh fading with CSI generation
    """
    def __init__(self, channel_type='AWGN', snr=10):
        super(Channel, self).__init__()
        self.channel_type = channel_type
        self.snr = snr
        self._csi = None  # Store CSI for feedback
        
    def forward(self, x, return_csi=False):
        """
        Forward pass through the channel
        Args:
            x: Input signal tensor
            return_csi: If True, return (output, csi) tuple
        Returns:
            Channel output, optionally with CSI
        """
        if self.channel_type == 'AWGN':
            output, csi = self._awgn_channel(x)
        elif self.channel_type == 'Rayleigh':
            output, csi = self._rayleigh_channel(x)
        else:
            raise ValueError(f"Unknown channel type: {self.channel_type}")
        
        self._csi = csi
        
        if return_csi:
            return output, csi
        return output
    
    def _awgn_channel(self, x):
        """
        AWGN channel model
        For AWGN, CSI is essentially the noise variance (scalar per batch)
        """
        snr_linear = 10 ** (self.snr / 10.0)
        
        # Calculate signal power
        # signal_power = torch.mean(x ** 2, dim=(1, 2, 3), keepdim=True)
        if x.dim() >= 2:
            # 对除了批次维度外的所有维度求均值
            dims = list(range(1, x.dim()))
            signal_power = torch.mean(x ** 2, dim=dims, keepdim=True)
        else:
            # 处理极端情况
            signal_power = torch.mean(x ** 2, dim=-1, keepdim=True)
        # Calculate noise power
        noise_power = signal_power / snr_linear
        noise_std = torch.sqrt(noise_power)
        
        # Generate noise
        noise = torch.randn_like(x) * noise_std
        
        # Channel output
        output = x + noise
        
        # CSI for AWGN: noise variance per sample (batch_size,)
        # We expand it to a more informative representation
        batch_size = x.size(0)
        csi = torch.cat([
            signal_power.view(batch_size, 1),
            noise_power.view(batch_size, 1),
            torch.ones(batch_size, 1, device=x.device) * self.snr
        ], dim=1)  # (batch_size, 3)
        
        return output, csi
    
    def _rayleigh_channel(self, x):
        """
        Rayleigh fading channel model with complex channel coefficients
        CSI contains the channel coefficients
        """
        snr_linear = 10 ** (self.snr / 10.0)
        batch_size = x.size(0)
        
        # Generate Rayleigh fading coefficients (complex)
        # h = h_real + j*h_imag, where h_real, h_imag ~ N(0, 0.5)
        h_real = torch.randn(batch_size, 1, 1, 1, device=x.device) * np.sqrt(0.5)
        h_imag = torch.randn(batch_size, 1, 1, 1, device=x.device) * np.sqrt(0.5)
        
        # Channel magnitude |h|^2
        h_mag_sq = h_real ** 2 + h_imag ** 2
        
        # Apply fading (simplified: multiply by magnitude)
        # For real-valued signals, we use the magnitude
        h_mag = torch.sqrt(h_mag_sq)
        faded_signal = x * h_mag
        
        # Calculate noise power
        signal_power = torch.mean(faded_signal ** 2, dim=(1, 2, 3), keepdim=True)
        noise_power = signal_power / snr_linear
        noise_std = torch.sqrt(noise_power)
        
        # Add noise
        noise = torch.randn_like(x) * noise_std
        output = faded_signal + noise
        
        # CSI: channel coefficients and statistics
        # (batch_size, csi_dim) where csi_dim contains useful channel info
        csi = torch.cat([
            h_real.view(batch_size, 1),
            h_imag.view(batch_size, 1),
            h_mag_sq.view(batch_size, 1),
            signal_power.view(batch_size, 1),
            noise_power.view(batch_size, 1),
            torch.ones(batch_size, 1, device=x.device) * self.snr
        ], dim=1)  # (batch_size, 6)
        
        return output, csi
    
    def get_csi(self):
        """Get the stored CSI from the last forward pass"""
        return self._csi
    
    def get_csi_dim(self):
        """Get the dimension of CSI vector"""
        if self.channel_type == 'AWGN':
            return 3
        elif self.channel_type == 'Rayleigh':
            return 6
        else:
            return 0
    
    def get_channel(self):
        """Get channel type and SNR"""
        return {'type': self.channel_type, 'snr': self.snr}
    
    def set_snr(self, snr):
        """Update SNR value"""
        self.snr = snr


class ChannelWithPerfectCSI(Channel):
    """
    Channel model that simulates perfect CSI at receiver
    Used for comparison with CSI feedback scheme
    """
    def __init__(self, channel_type='AWGN', snr=10):
        super().__init__(channel_type, snr)
    
    def forward(self, x):
        output, csi = super().forward(x, return_csi=True)
        return output, csi
