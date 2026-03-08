# -*- coding: utf-8 -*-
"""
DeepJSCC with Lightweight CSI Compression Feedback and End-to-End Joint Optimization

This module extends the original DeepJSCC model with:
1. CSI compression module at transmitter
2. CSI decompression module at receiver
3. CSI-aware decoder that utilizes reconstructed CSI
4. Joint loss function for end-to-end optimization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from channel_csi import Channel
from csi_feedback import CSIFeedbackModule, AdaptiveCSIFeedback


def ratio2filtersize(x: torch.Tensor, ratio):
    """Calculate filter size based on compression ratio"""
    if x.dim() == 4:
        before_size = torch.prod(torch.tensor(x.size()[1:]))
    elif x.dim() == 3:
        before_size = torch.prod(torch.tensor(x.size()))
    else:
        raise Exception('Unknown size of input')
    encoder_temp = _Encoder(is_temp=True)
    z_temp = encoder_temp(x)
    c = before_size * ratio / torch.prod(torch.tensor(z_temp.size()[-2:]))
    return int(c)


class _ConvWithPReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(_ConvWithPReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.prelu = nn.PReLU()
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.conv(x)
        x = self.prelu(x)
        return x


class _TransConvWithPReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, 
                 activate=None, padding=0, output_padding=0):
        super(_TransConvWithPReLU, self).__init__()
        self.transconv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride, padding, output_padding)
        self.activate = activate if activate is not None else nn.PReLU()
        
        if isinstance(self.activate, nn.PReLU):
            nn.init.kaiming_normal_(self.transconv.weight, mode='fan_out', nonlinearity='leaky_relu')
        else:
            nn.init.xavier_normal_(self.transconv.weight)

    def forward(self, x):
        x = self.transconv(x)
        x = self.activate(x)
        return x


class _Encoder(nn.Module):
    """Image Encoder for JSCC"""
    def __init__(self, c=1, is_temp=False, P=1):
        super(_Encoder, self).__init__()
        self.is_temp = is_temp
        self.conv1 = _ConvWithPReLU(in_channels=3, out_channels=16, kernel_size=5, stride=2, padding=2)
        self.conv2 = _ConvWithPReLU(in_channels=16, out_channels=32, kernel_size=5, stride=2, padding=2)
        self.conv3 = _ConvWithPReLU(in_channels=32, out_channels=32, kernel_size=5, padding=2)
        self.conv4 = _ConvWithPReLU(in_channels=32, out_channels=32, kernel_size=5, padding=2)
        self.conv5 = _ConvWithPReLU(in_channels=32, out_channels=2*c, kernel_size=5, padding=2)
        self.norm = self._normlizationLayer(P=P)

    @staticmethod
    def _normlizationLayer(P=1):
        def _inner(z_hat: torch.Tensor):
            if z_hat.dim() == 4:
                batch_size = z_hat.size()[0]
                k = torch.prod(torch.tensor(z_hat.size()[1:]))
            elif z_hat.dim() == 3:
                batch_size = 1
                k = torch.prod(torch.tensor(z_hat.size()))
            else:
                raise Exception('Unknown size of input')
            z_temp = z_hat.reshape(batch_size, 1, 1, -1)
            z_trans = z_hat.reshape(batch_size, 1, -1, 1)
            tensor = torch.sqrt(P * k) * z_hat / torch.sqrt((z_temp @ z_trans))
            if batch_size == 1:
                return tensor.squeeze(0)
            return tensor
        return _inner

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        if not self.is_temp:
            x = self.conv5(x)
            x = self.norm(x)
        return x


class _Decoder(nn.Module):
    """Image Decoder for JSCC (without CSI awareness)"""
    def __init__(self, c=1):
        super(_Decoder, self).__init__()
        self.tconv1 = _TransConvWithPReLU(
            in_channels=2*c, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.tconv2 = _TransConvWithPReLU(
            in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.tconv3 = _TransConvWithPReLU(
            in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.tconv4 = _TransConvWithPReLU(
            in_channels=32, out_channels=16, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.tconv5 = _TransConvWithPReLU(
            in_channels=16, out_channels=3, kernel_size=5, stride=2, 
            padding=2, output_padding=1, activate=nn.Sigmoid())

    def forward(self, x):
        x = self.tconv1(x)
        x = self.tconv2(x)
        x = self.tconv3(x)
        x = self.tconv4(x)
        x = self.tconv5(x)
        return x


class _CSIAwareDecoder(nn.Module):
    """
    CSI-Aware Decoder: Utilizes reconstructed CSI for improved decoding
    
    The CSI information is injected into the decoder through:
    1. Feature modulation (FiLM-like mechanism)
    2. Concatenation with intermediate features
    """
    def __init__(self, c=1, csi_dim=6):
        super(_CSIAwareDecoder, self).__init__()
        
        self.csi_dim = csi_dim
        
        # CSI embedding network
        self.csi_embed = nn.Sequential(
            nn.Linear(csi_dim, 64),
            nn.PReLU(),
            nn.Linear(64, 64)
        )
        
        # FiLM generators for each decoder layer (scale and shift)
        self.film1 = nn.Linear(64, 32 * 2)  # 32 channels, scale + shift
        self.film2 = nn.Linear(64, 32 * 2)
        self.film3 = nn.Linear(64, 32 * 2)
        self.film4 = nn.Linear(64, 16 * 2)
        
        # Decoder layers
        self.tconv1 = _TransConvWithPReLU(
            in_channels=2*c, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.tconv2 = _TransConvWithPReLU(
            in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.tconv3 = _TransConvWithPReLU(
            in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.tconv4 = _TransConvWithPReLU(
            in_channels=32, out_channels=16, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.tconv5 = _TransConvWithPReLU(
            in_channels=16, out_channels=3, kernel_size=5, stride=2, 
            padding=2, output_padding=1, activate=nn.Sigmoid())
    
    def _apply_film(self, x, film_params):
        """Apply FiLM modulation: y = gamma * x + beta"""
        batch_size = x.size(0)
        channels = x.size(1)

        # Split into scale (gamma) and shift (beta)
        gamma = film_params[:, :channels].view(batch_size, channels, 1, 1)
        beta = film_params[:, channels:].view(batch_size, channels, 1, 1)

        # Apply modulation with sigmoid for stable scaling
        gamma = torch.sigmoid(gamma) * 2  # Scale range [0, 2]

        return gamma * x + beta




    def forward(self, z, csi):
        """
        Forward pass with CSI-aware decoding
        
        Args:
            z: Channel output tensor
            csi: Reconstructed CSI tensor (batch_size, csi_dim)
        
        Returns:
            Reconstructed image
        """
        # Embed CSI
        csi_feat = self.csi_embed(csi)
        
        # Generate FiLM parameters
        film1_params = self.film1(csi_feat)
        film2_params = self.film2(csi_feat)
        film3_params = self.film3(csi_feat)
        film4_params = self.film4(csi_feat)
        
        # Decode with FiLM modulation
        x = self.tconv1(z)
        x = self._apply_film(x, film1_params)
        
        x = self.tconv2(x)
        x = self._apply_film(x, film2_params)
        
        x = self.tconv3(x)
        x = self._apply_film(x, film3_params)
        
        x = self.tconv4(x)
        x = self._apply_film(x, film4_params)
        
        x = self.tconv5(x)
        
        return x


class DeepJSCC(nn.Module):
    """Original DeepJSCC model (for backward compatibility)"""
    def __init__(self, c, channel_type='AWGN', snr=None):
        super(DeepJSCC, self).__init__()
        self.encoder = _Encoder(c=c)
        if snr is not None:
            self.channel = Channel(channel_type, snr)
        self.decoder = _Decoder(c=c)

    def forward(self, x):
        z = self.encoder(x)
        if hasattr(self, 'channel') and self.channel is not None:
            z = self.channel(z)
        x_hat = self.decoder(z)
        return x_hat

    def change_channel(self, channel_type='AWGN', snr=None):
        if snr is None:
            self.channel = None
        else:
            self.channel = Channel(channel_type, snr)

    def get_channel(self):
        if hasattr(self, 'channel') and self.channel is not None:
            return self.channel.get_channel()
        return None

    def loss(self, prd, gt):
        criterion = nn.MSELoss(reduction='mean')
        loss = criterion(prd, gt)
        return loss


class DeepJSCCWithCSIFeedback(nn.Module):
    """
    DeepJSCC with Lightweight CSI Compression Feedback
    
    Key Features:
    1. CSI compression at transmitter side
    2. CSI decompression at receiver side  
    3. CSI-aware decoder utilizing reconstructed CSI
    4. End-to-end joint optimization with combined loss
    
    Architecture:
        Transmitter: Encoder -> [CSI Compressor]
        Channel: AWGN/Rayleigh (generates CSI)
        Receiver: [CSI Decompressor] -> CSI-Aware Decoder
    """
    def __init__(self, c, channel_type='AWGN', snr=10, feedback_bits=32, 
                 use_csi_aware_decoder=True, csi_loss_weight=0.1):
        """
        Args:
            c: Number of output channels for encoder
            channel_type: Type of channel ('AWGN' or 'Rayleigh')
            snr: Signal-to-noise ratio in dB
            feedback_bits: Number of bits for CSI feedback
            use_csi_aware_decoder: Whether to use CSI-aware decoder
            csi_loss_weight: Weight for CSI reconstruction loss in joint training
        """
        super(DeepJSCCWithCSIFeedback, self).__init__()
        
        self.c = c
        self.channel_type = channel_type
        self.snr = snr
        self.feedback_bits = feedback_bits
        self.use_csi_aware_decoder = use_csi_aware_decoder
        self.csi_loss_weight = csi_loss_weight
        
        # Image encoder
        self.encoder = _Encoder(c=c)
        
        # Channel
        self.channel = Channel(channel_type, snr)
        csi_dim = self.channel.get_csi_dim()
        
        # CSI feedback module
        self.csi_feedback = CSIFeedbackModule(
            csi_dim=csi_dim,
            feedback_bits=feedback_bits,
            hidden_dim=64,
            use_quantization=True
        )
        
        # Decoder (CSI-aware or standard)
        if use_csi_aware_decoder:
            self.decoder = _CSIAwareDecoder(c=c, csi_dim=csi_dim)
        else:
            self.decoder = _Decoder(c=c)
        
        # Loss functions
        self.mse_loss = nn.MSELoss(reduction='mean')
    
    def forward(self, x, return_intermediate=False):
        """
        Forward pass through the complete system
        
        Args:
            x: Input image tensor (batch_size, 3, H, W)
            return_intermediate: If True, return intermediate results
        
        Returns:
            If return_intermediate:
                (x_hat, csi_original, csi_compressed, csi_reconstructed)
            Else:
                x_hat (reconstructed image)
        """
        # Encode image
        z = self.encoder(x)
        
        # Pass through channel and get CSI
        z_channel, csi_original = self.channel(z, return_csi=True)
        
        # Compress and decompress CSI through feedback link
        csi_compressed, csi_reconstructed = self.csi_feedback(csi_original)
        
        # Decode with CSI information
        if self.use_csi_aware_decoder:
            x_hat = self.decoder(z_channel, csi_reconstructed)
        else:
            x_hat = self.decoder(z_channel)
        
        if return_intermediate:
            return x_hat, csi_original, csi_compressed, csi_reconstructed
        
        return x_hat
    
    def loss(self, x, x_hat, csi_original=None, csi_reconstructed=None):
        """
        Compute joint loss for end-to-end training
        
        Args:
            x: Original image
            x_hat: Reconstructed image
            csi_original: Original CSI (optional, for joint training)
            csi_reconstructed: Reconstructed CSI (optional, for joint training)
        
        Returns:
            Total loss (image reconstruction + CSI reconstruction)
        """
        # Image reconstruction loss (main task)
        img_loss = self.mse_loss(x_hat, x)
        
        # CSI reconstruction loss (auxiliary task)
        if csi_original is not None and csi_reconstructed is not None:
            csi_loss = self.mse_loss(csi_reconstructed, csi_original)
            total_loss = img_loss + self.csi_loss_weight * csi_loss
            return total_loss, img_loss, csi_loss
        
        return img_loss
    
    def change_channel(self, channel_type='AWGN', snr=None):
        """Update channel configuration"""
        if snr is not None:
            self.channel = Channel(channel_type, snr)
            self.snr = snr
            self.channel_type = channel_type
            
            # Update CSI feedback module if CSI dimension changes
            new_csi_dim = self.channel.get_csi_dim()
            if new_csi_dim != self.csi_feedback.csi_dim:
                self.csi_feedback = CSIFeedbackModule(
                    csi_dim=new_csi_dim,
                    feedback_bits=self.feedback_bits,
                    hidden_dim=64
                )
                if self.use_csi_aware_decoder:
                    self.decoder = _CSIAwareDecoder(c=self.c, csi_dim=new_csi_dim)
    
    def get_channel(self):
        """Get current channel configuration"""
        return self.channel.get_channel()
    
    def get_feedback_bits(self):
        """Get number of feedback bits"""
        return self.feedback_bits
    
    def get_model_info(self):
        """Get model configuration information"""
        return {
            'c': self.c,
            'channel_type': self.channel_type,
            'snr': self.snr,
            'feedback_bits': self.feedback_bits,
            'use_csi_aware_decoder': self.use_csi_aware_decoder,
            'csi_loss_weight': self.csi_loss_weight,
            'csi_dim': self.channel.get_csi_dim()
        }


class DeepJSCCWithAdaptiveCSIFeedback(DeepJSCCWithCSIFeedback):
    """
    DeepJSCC with Adaptive-Rate CSI Feedback
    
    Extends DeepJSCCWithCSIFeedback with variable-rate CSI compression
    that can adapt to channel conditions.
    """
    def __init__(self, c, channel_type='AWGN', snr=10, 
                 feedback_bits_list=[16, 32, 64],
                 use_csi_aware_decoder=True, csi_loss_weight=0.1):
        # Initialize parent without CSI feedback (we'll replace it)
        nn.Module.__init__(self)
        
        self.c = c
        self.channel_type = channel_type
        self.snr = snr
        self.feedback_bits_list = feedback_bits_list
        self.use_csi_aware_decoder = use_csi_aware_decoder
        self.csi_loss_weight = csi_loss_weight
        
        # Image encoder
        self.encoder = _Encoder(c=c)
        
        # Channel
        self.channel = Channel(channel_type, snr)
        csi_dim = self.channel.get_csi_dim()
        
        # Adaptive CSI feedback module
        self.csi_feedback = AdaptiveCSIFeedback(
            csi_dim=csi_dim,
            feedback_bits_list=feedback_bits_list,
            hidden_dim=64
        )
        
        # Decoder
        if use_csi_aware_decoder:
            self.decoder = _CSIAwareDecoder(c=c, csi_dim=csi_dim)
        else:
            self.decoder = _Decoder(c=c)
        
        self.mse_loss = nn.MSELoss(reduction='mean')
    
    def forward(self, x, feedback_bits=None, return_intermediate=False):
        """
        Forward pass with optional rate selection
        
        Args:
            x: Input image
            feedback_bits: Specific feedback rate (None for adaptive)
            return_intermediate: Whether to return intermediate results
        """
        z = self.encoder(x)
        z_channel, csi_original = self.channel(z, return_csi=True)
        
        csi_compressed, csi_reconstructed, selected_rate = self.csi_feedback(
            csi_original, feedback_bits=feedback_bits
        )
        
        if self.use_csi_aware_decoder:
            x_hat = self.decoder(z_channel, csi_reconstructed)
        else:
            x_hat = self.decoder(z_channel)
        
        if return_intermediate:
            return x_hat, csi_original, csi_compressed, csi_reconstructed, selected_rate
        
        return x_hat


if __name__ == '__main__':
    # Test the models
    print("Testing DeepJSCC with CSI Feedback...")
    
    # Test standard model
    model = DeepJSCCWithCSIFeedback(c=20, channel_type='Rayleigh', snr=10, feedback_bits=32)
    print(f"Model info: {model.get_model_info()}")
    
    x = torch.rand(4, 3, 32, 32)
    x_hat, csi_orig, csi_comp, csi_recon = model(x, return_intermediate=True)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {x_hat.shape}")
    print(f"CSI original shape: {csi_orig.shape}")
    print(f"CSI compressed shape: {csi_comp.shape}")
    print(f"CSI reconstructed shape: {csi_recon.shape}")
    
    # Test loss computation
    total_loss, img_loss, csi_loss = model.loss(x, x_hat, csi_orig, csi_recon)
    print(f"Total loss: {total_loss.item():.4f}")
    print(f"Image loss: {img_loss.item():.4f}")
    print(f"CSI loss: {csi_loss.item():.4f}")
    
    # Test adaptive model
    print("\nTesting Adaptive CSI Feedback model...")
    adaptive_model = DeepJSCCWithAdaptiveCSIFeedback(
        c=20, channel_type='Rayleigh', snr=10, 
        feedback_bits_list=[16, 32, 64]
    )
    x_hat_adaptive = adaptive_model(x)
    print(f"Adaptive output shape: {x_hat_adaptive.shape}")
    
    # Count parameters
    def count_params(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    baseline = DeepJSCC(c=20, channel_type='Rayleigh', snr=10)
    print(f"\nParameter count:")
    print(f"Baseline DeepJSCC: {count_params(baseline):,}")
    print(f"DeepJSCC with CSI Feedback: {count_params(model):,}")
    print(f"Parameter increase: {(count_params(model) - count_params(baseline)) / count_params(baseline) * 100:.1f}%")
