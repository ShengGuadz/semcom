# DeepJSCC with Lightweight CSI Compression Feedback

基于DeepJSCC的轻量化CSI压缩反馈与端到端联合优化实现。

## 创新点概述

本项目在原有DeepJSCC基础上增加了轻量化CSI压缩反馈机制:

1. **发射端CSI压缩模块**: 3层全连接网络，将高维CSI压缩为低维反馈比特
2. **接收端CSI解压缩模块**: 镜像结构的3层全连接网络，恢复CSI信息
3. **CSI感知解码器**: 利用FiLM机制将CSI信息注入解码过程
4. **端到端联合训练**: 图像重建损失 + CSI重建损失的加权组合

## 目录结构

```
csi_jscc/
├── channel.py          # 扩展的信道模块(支持CSI生成)
├── csi_feedback.py     # CSI压缩/解压缩模块
├── model.py            # 集成CSI反馈的DeepJSCC模型
├── train.py            # 训练脚本
├── utils.py            # 工具函数
└── README.md           # 说明文档
```

## 核心模块说明

### 1. CSI反馈模块 (`csi_feedback.py`)

```python
# CSI压缩器
class CSICompressor(nn.Module):
    # 输入: CSI向量 (batch_size, csi_dim)
    # 输出: 压缩后的表示 (batch_size, feedback_bits)
    
# CSI解压缩器  
class CSIDecompressor(nn.Module):
    # 输入: 压缩表示 (batch_size, feedback_bits)
    # 输出: 重建的CSI (batch_size, csi_dim)

# 完整反馈模块
class CSIFeedbackModule(nn.Module):
    # 组合压缩器和解压缩器
```

### 2. 扩展的DeepJSCC模型 (`model.py`)

```python
class DeepJSCCWithCSIFeedback(nn.Module):
    """
    系统架构:
    发射端: 图像 -> Encoder -> 归一化信号
    信道:   信号 -> Channel(产生CSI) -> 带噪信号
    反馈:   CSI -> Compressor -> 压缩CSI -> Decompressor -> 重建CSI  
    接收端: 带噪信号 + 重建CSI -> CSI-Aware Decoder -> 重建图像
    """
```

### 3. CSI感知解码器

使用FiLM (Feature-wise Linear Modulation) 机制将CSI信息注入解码器:

```python
# FiLM调制: y = gamma * x + beta
# gamma和beta由CSI特征生成
```

## 使用方法

### 训练

```bash
# 基础训练 (AWGN信道, 32bit反馈)
python train.py --dataset cifar10 --channel AWGN --feedback_bits 32 --snr_list 10 --ratio_list 1/6

# Rayleigh信道训练
python train.py --dataset cifar10 --channel Rayleigh --feedback_bits 64 --csi_loss_weight 0.2

# 多SNR/多压缩率训练
python train.py --dataset cifar10 --channel AWGN --snr_list 19 13 7 4 1 --ratio_list 1/6 1/12
```

### 主要参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--feedback_bits` | 32 | CSI反馈比特数 |
| `--csi_loss_weight` | 0.1 | CSI损失权重 |
| `--use_csi_aware_decoder` | True | 是否使用CSI感知解码器 |
| `--joint_training` | True | 是否联合训练 |
| `--channel` | AWGN | 信道类型(AWGN/Rayleigh) |

### 代码示例

```python
from model import DeepJSCCWithCSIFeedback
import torch

# 创建模型
model = DeepJSCCWithCSIFeedback(
    c=20,                      # 编码器输出通道数
    channel_type='Rayleigh',   # 信道类型
    snr=10,                    # 信噪比(dB)
    feedback_bits=32,          # 反馈比特数
    use_csi_aware_decoder=True,# CSI感知解码
    csi_loss_weight=0.1        # CSI损失权重
)

# 前向传播
x = torch.rand(4, 3, 32, 32)
x_hat, csi_orig, csi_comp, csi_recon = model(x, return_intermediate=True)

# 计算联合损失
total_loss, img_loss, csi_loss = model.loss(x, x_hat, csi_orig, csi_recon)
```

## 信道模型

### AWGN信道
- CSI维度: 3 (信号功率, 噪声功率, SNR)
- 输出: `y = x + n`, 其中 `n ~ N(0, σ²)`

### Rayleigh衰落信道  
- CSI维度: 6 (h_real, h_imag, |h|², 信号功率, 噪声功率, SNR)
- 输出: `y = h*x + n`, 其中 `h ~ CN(0, 1)`

## 损失函数

```python
L_total = L_img + λ * L_csi

# L_img: 图像重建MSE损失
# L_csi: CSI重建MSE损失  
# λ: csi_loss_weight (默认0.1)
```

## 参数量分析

以c=20为例:
- 基线DeepJSCC: ~120K 参数
- 增加CSI反馈后: ~140K 参数
- 参数增加比例: ~15-20%

## 实验建议

1. **反馈比特数选择**: 
   - AWGN信道: 16-32 bits即可
   - Rayleigh信道: 32-64 bits效果更好

2. **CSI损失权重**:
   - 建议范围: 0.05 - 0.2
   - 过大会影响主任务性能

3. **训练策略**:
   - 可先预训练基础JSCC，再微调CSI模块
   - 或直接端到端联合训练

## 依赖

```
torch >= 1.8.0
torchvision
numpy
tqdm
tensorboardX (或 torch.utils.tensorboard)
pyyaml
```

## 参考

- 原始DeepJSCC论文: "Deep Joint Source-Channel Coding for Wireless Image Transmission"
- FiLM机制: "FiLM: Visual Reasoning with a General Conditioning Layer"
