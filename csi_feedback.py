# -*- coding: utf-8 -*-
"""
CSI Feedback Module for Lightweight CSI Compression and Reconstruction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CSICompressor(nn.Module):
    """
    CSI Compressor: Compresses high-dimensional CSI to low-dimensional feedback bits
    
    Architecture: 3-layer fully connected network with PReLU activation
    Input: Raw CSI vector
    Output: Compressed CSI (quantization indices or continuous representation)
    """
    def __init__(self, csi_dim, hidden_dim=64, feedback_bits=32, use_quantization=True):
        """
        Args:
            csi_dim: Dimension of input CSI vector
            hidden_dim: Hidden layer dimension
            feedback_bits: Number of feedback bits (output dimension)
            use_quantization: Whether to apply soft quantization during training
        """
        super(CSICompressor, self).__init__()
        
        self.csi_dim = csi_dim
        self.feedback_bits = feedback_bits
        self.use_quantization = use_quantization
        
        # 3-layer fully connected network
        self.fc1 = nn.Linear(csi_dim, hidden_dim)
        self.prelu1 = nn.PReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.prelu2 = nn.PReLU()
        self.fc3 = nn.Linear(hidden_dim, feedback_bits)
        
        # Batch normalization for stability
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, csi):
        """
        Forward pass for CSI compression
        
        Args:
            csi: Input CSI tensor of shape (batch_size, csi_dim)
        
        Returns:
            Compressed CSI of shape (batch_size, feedback_bits)
        """
        x = self.fc1(csi)
        x = self.bn1(x)
        x = self.prelu1(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.prelu2(x)
        
        x = self.fc3(x)
        
        # Apply sigmoid to bound output to [0, 1]
        x = torch.sigmoid(x)
        
        if self.use_quantization and self.training:
            # Soft quantization using Gumbel-Softmax-like approach
            # During training: add noise for regularization
            # During inference: hard quantization
            x = x + 0.1 * torch.randn_like(x) * (1 - x) * x
        
        return x
    
    def quantize(self, x):
        """Hard quantization for inference"""
        return (x > 0.5).float()


class CSIDecompressor(nn.Module):
    """
    CSI Decompressor: Reconstructs CSI from compressed feedback
    
    Architecture: 3-layer fully connected network (mirror of compressor)
    Input: Compressed CSI (feedback bits)
    Output: Reconstructed CSI
    """
    def __init__(self, feedback_bits=32, hidden_dim=64, csi_dim=6):
        """
        Args:
            feedback_bits: Number of input feedback bits
            hidden_dim: Hidden layer dimension
            csi_dim: Output CSI dimension
        """
        super(CSIDecompressor, self).__init__()
        
        self.feedback_bits = feedback_bits
        self.csi_dim = csi_dim
        
        # 3-layer fully connected network
        self.fc1 = nn.Linear(feedback_bits, hidden_dim)
        self.prelu1 = nn.PReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.prelu2 = nn.PReLU()
        self.fc3 = nn.Linear(hidden_dim, csi_dim)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, compressed_csi):
        """
        Forward pass for CSI reconstruction
        
        Args:
            compressed_csi: Compressed CSI of shape (batch_size, feedback_bits)
        
        Returns:
            Reconstructed CSI of shape (batch_size, csi_dim)
        """
        x = self.fc1(compressed_csi)
        x = self.bn1(x)
        x = self.prelu1(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.prelu2(x)
        
        x = self.fc3(x)
        
        return x


class CSIFeedbackModule(nn.Module):
    """
    Complete CSI Feedback Module combining Compressor and Decompressor
    
    This module enables end-to-end training of CSI compression and reconstruction
    jointly with the JSCC encoder/decoder.
    """
    def __init__(self, csi_dim, feedback_bits=32, hidden_dim=64, use_quantization=True):
        """
        Args:
            csi_dim: Dimension of CSI vector (depends on channel type)
            feedback_bits: Number of feedback bits
            hidden_dim: Hidden layer dimension for both compressor and decompressor
            use_quantization: Whether to apply soft quantization
        """
        super(CSIFeedbackModule, self).__init__()
        
        self.csi_dim = csi_dim
        self.feedback_bits = feedback_bits
        
        self.compressor = CSICompressor(
            csi_dim=csi_dim,
            hidden_dim=hidden_dim,
            feedback_bits=feedback_bits,
            use_quantization=use_quantization
        )
        
        self.decompressor = CSIDecompressor(
            feedback_bits=feedback_bits,
            hidden_dim=hidden_dim,
            csi_dim=csi_dim
        )
    
    def compress(self, csi):
        """Compress CSI at transmitter side"""
        return self.compressor(csi)
    
    def decompress(self, compressed_csi):
        """Decompress CSI at receiver side"""
        return self.decompressor(compressed_csi)
    
    def forward(self, csi):
        """
        Full forward pass: compress then decompress
        Used for training the feedback module
        
        Args:
            csi: Original CSI tensor
        
        Returns:
            Tuple of (compressed_csi, reconstructed_csi)
        """
        compressed = self.compressor(csi)
        reconstructed = self.decompressor(compressed)
        return compressed, reconstructed
    
    def feedback_loss(self, csi_original, csi_reconstructed):
        """
        Compute CSI reconstruction loss
        
        Args:
            csi_original: Ground truth CSI
            csi_reconstructed: Reconstructed CSI from feedback
        
        Returns:
            MSE loss between original and reconstructed CSI
        """
        return F.mse_loss(csi_reconstructed, csi_original)
    
    def get_compression_ratio(self):
        """Get the compression ratio"""
        return self.feedback_bits / self.csi_dim


class AdaptiveCSIFeedback(nn.Module):
    """
    Adaptive CSI Feedback Module with variable-rate compression
    
    Supports multiple feedback bit rates and can adaptively select
    the appropriate rate based on channel conditions.
    """
    def __init__(self, csi_dim, feedback_bits_list=[16, 32, 64], hidden_dim=64):
        """
        Args:
            csi_dim: Dimension of CSI vector
            feedback_bits_list: List of supported feedback bit rates
            hidden_dim: Hidden layer dimension
        """
        super(AdaptiveCSIFeedback, self).__init__()
        
        self.csi_dim = csi_dim
        self.feedback_bits_list = feedback_bits_list
        
        # Create multiple feedback modules for different rates
        self.feedback_modules = nn.ModuleDict({
            str(bits): CSIFeedbackModule(csi_dim, bits, hidden_dim)
            for bits in feedback_bits_list
        })
        
        # Rate selection network
        self.rate_selector = nn.Sequential(
            nn.Linear(csi_dim, hidden_dim),
            nn.PReLU(),
            nn.Linear(hidden_dim, len(feedback_bits_list)),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, csi, feedback_bits=None):
        """
        Forward pass with optional rate selection
        
        Args:
            csi: Input CSI tensor
            feedback_bits: Specific feedback bits to use (None for adaptive)
        
        Returns:
            Tuple of (compressed_csi, reconstructed_csi, selected_rate)
        """
        if feedback_bits is not None:
            # Use specified rate
            module = self.feedback_modules[str(feedback_bits)]
            compressed, reconstructed = module(csi)
            return compressed, reconstructed, feedback_bits
        else:
            # Adaptive rate selection
            rate_probs = self.rate_selector(csi)
            selected_idx = torch.argmax(rate_probs, dim=-1)
            
            # For simplicity, use the most common selection in batch
            most_common_idx = torch.mode(selected_idx).values.item()
            selected_bits = self.feedback_bits_list[most_common_idx]
            
            module = self.feedback_modules[str(selected_bits)]
            compressed, reconstructed = module(csi)
            
            return compressed, reconstructed, selected_bits
