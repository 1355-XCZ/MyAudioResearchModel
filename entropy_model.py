"""
无条件自回归熵模型 q(z) - 发布版
仅移除条件标签相关代码，所有其他科学逻辑完全保留

核心功能完全保留：
1. 右移输入（避免自我泄漏）
2. 帧内AR（序列长度 = G×M，不是 T*G*M）
3. SKIP非负化（id = K）
4. 添加 group/layer 嵌入
5. predict_next_token_prob（ECVQ判决）
6. 所有计算和训练方法
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math
import logging
from copy import deepcopy

logger = logging.getLogger(__name__)


class PositionalEncoding(nn.Module):
    """正弦位置编码（帧内位置）"""
    
    def __init__(self, d_model: int, max_len: int = 100):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, D) 其中 L = 帧内序列长度
        Returns:
            (B, L, D)
        """
        return x + self.pe[:x.size(1), :]


class AutoregressiveEntropyModel(nn.Module):
    """
    无条件自回归熵模型（帧内AR）- 仅 q(z)
    
    修复点：
    1. 序列 = 每帧的 (G×M) 个token（不是跨帧）
    2. 输入右移一位，避免自我泄漏
    3. SKIP id = K（非负）
    4. 添加 group/layer 位置嵌入
    """
    
    def __init__(self, config, rvq_config):
        """
        Args:
            config: EntropyModelConfig实例
            rvq_config: GroupedRVQConfig实例
        """
        super().__init__()
        self.config = config
        self.rvq_config = rvq_config
        
        # 词表大小
        self.K = rvq_config.fine_codebook_size  # 码本大小
        self.SKIP_ID = self.K                   # SKIP 使用 K（非负）
        self.V = self.K + 1                     # 词表 = K个码 + 1个SKIP
        self.BOS_ID = self.V                    # BOS（输入专用）
        
        # 帧内序列长度
        self.num_groups = rvq_config.num_groups
        self.num_layers_per_group = rvq_config.num_fine_layers
        self.L = self.num_groups * self.num_layers_per_group  # 帧内长度
        
        # Token Embedding（包含BOS）
        self.token_embedding = nn.Embedding(
            num_embeddings=self.V + 1,  # V个输出类 + 1个BOS
            embedding_dim=config.d_model
        )
        
        # 位置编码（帧内）
        self.pos_encoding = PositionalEncoding(config.d_model, max_len=self.L)
        
        # Group/Layer 嵌入（让模型知道"这是哪个组/层"）
        self.group_embedding = nn.Embedding(self.num_groups, config.d_model)
        self.layer_embedding = nn.Embedding(self.num_layers_per_group, config.d_model)
        
        # 预计算 group_ids 和 layer_ids（固定顺序）
        self.register_buffer('group_ids', self._create_group_ids())
        self.register_buffer('layer_ids', self._create_layer_ids())
        
        # Transformer编码器（因果掩码）
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.num_heads,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_layers
        )
        
        # 输出层：预测下一个token（输出类数 = V，不含BOS）
        self.output_proj = nn.Linear(config.d_model, self.V)
        
        # 无条件模型标记
        self.condition_on_label = False
        
        # 缓存因果掩码（恒定帧内长度 L，避免重复构造和传输）
        self.register_buffer(
            "causal_mask_buf",
            self.generate_causal_mask(self.L, device=torch.device("cpu"))
        )
        
        # Mask缓存（ChatGPT建议 - 避免每次重建）
        self._mask_cache = {}

        logger.info(f"✅ AutoregressiveEntropyModel 初始化（无条件版本 - 仅 q(z)）:")
        logger.info(f"   - 词表: V={self.V} (K={self.K} + SKIP={self.SKIP_ID})")
        logger.info(f"   - 帧内序列长度: L={self.L} ({self.num_groups}组 × {self.num_layers_per_group}层)")
        logger.info(f"   - BOS id: {self.BOS_ID}")
        logger.info(f"   - 因果掩码缓存: 已注册为 buffer")
    
    def _create_group_ids(self) -> torch.Tensor:
        """
        创建固定的 group_ids 序列
        顺序：g 外层，m 内层
        例如 G=2, M=3: [0,0,0, 1,1,1]
        """
        ids = []
        for g in range(self.num_groups):
            for m in range(self.num_layers_per_group):
                ids.append(g)
        return torch.tensor(ids, dtype=torch.long)
    
    def _create_layer_ids(self) -> torch.Tensor:
        """
        创建固定的 layer_ids 序列
        顺序：g 外层，m 内层
        例如 G=2, M=3: [0,1,2, 0,1,2]
        """
        ids = []
        for g in range(self.num_groups):
            for m in range(self.num_layers_per_group):
                ids.append(m)
        return torch.tensor(ids, dtype=torch.long)
    
    def forward(
        self,
        indices: torch.Tensor,
        labels: Optional[torch.Tensor] = None  # 保留参数接口兼容性，但不使用
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播（训练用，教师强制）
        
        Args:
            indices: (B, T, L) 码本索引序列
                     其中 L = num_groups * num_layers_per_group
            labels: (B,) 保留参数兼容性，无条件模型不使用
        
        Returns:
            logits: (B*T, L, V) 预测分布
            targets: (B*T, L) 目标索引
        """
        B, T, L = indices.shape
        assert L == self.L, f"期望 L={self.L}，实际 L={L}"
        
        # 展平为帧内序列: (B*T, L)
        seq = indices.view(B * T, L)
        
        # 统一 SKIP 编码：-1 → K
        seq = torch.where(seq < 0, torch.full_like(seq, self.SKIP_ID), seq)
        
        # 输入 = 右移一位（第0位用BOS）
        inp = torch.roll(seq, shifts=1, dims=1)
        inp[:, 0] = self.BOS_ID
        
        # Token Embedding
        x = self.token_embedding(inp)  # (B*T, L, d_model)
        
        # 添加位置编码（帧内）
        x = self.pos_encoding(x)
        
        # 添加 group/layer 嵌入（让模型知道位置）
        x = x + self.group_embedding(self.group_ids).unsqueeze(0)  # broadcast (1, L, d) → (B*T, L, d)
        x = x + self.layer_embedding(self.layer_ids).unsqueeze(0)
        
        # 因果掩码（帧内，使用缓存）
        causal_mask = self.causal_mask_buf.to(x.device, non_blocking=True)
        
        # Transformer
        x = self.transformer(x, mask=causal_mask)  # (B*T, L, d_model)
        
        # 预测下一个token
        logits = self.output_proj(x)  # (B*T, L, V)
        
        # 目标：原序列（非右移）
        targets = seq  # (B*T, L)
        
        return logits, targets
    
    def compute_nll(
        self,
        indices: torch.Tensor,
        labels: Optional[torch.Tensor] = None,  # 保留参数兼容性
        valid_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        计算负对数似然 -log q(z)（支持 padding 屏蔽）
        
        Args:
            indices: (B, T, L) 码本索引
            labels: (B,) 保留参数兼容性，无条件模型不使用
            valid_mask: (B, T) bool tensor，标记有效帧（屏蔽padding）
        
        Returns:
            nll_per_sample: (B,) 每个样本的NLL（bits，只统计有效帧）
        """
        B, T, L = indices.shape
        
        # 前向传播
        logits, targets = self.forward(indices, labels)  # logits: (B*T, L, V), targets: (B*T, L)
        
        # 计算交叉熵（每个位置）
        logits_flat = logits.reshape(-1, self.V)  # (B*T*L, V)
        targets_flat = targets.reshape(-1)  # (B*T*L,)
        
        ce = F.cross_entropy(logits_flat, targets_flat, reduction='none')  # (B*T*L,)
        ce = ce.view(B, T, L)  # (B, T, L)
        
        # 转换为bits（log₂）
        nll_bits = ce / math.log(2)  # (B, T, L)
        
        # 应用 valid_mask（只统计有效帧）
        if valid_mask is not None:
            mask = valid_mask.unsqueeze(-1).float()  # (B, T, 1)
            nll_bits = nll_bits * mask  # (B, T, L)
        
        # 求和（每个样本，只统计有效帧）
        nll_per_sample = nll_bits.sum(dim=(1, 2))  # (B,)
        
        return nll_per_sample
    
    def compute_bits(
        self,
        indices: torch.Tensor,
        labels: Optional[torch.Tensor] = None,  # 保留参数兼容性
        valid_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        计算编码所需总bits（整个batch，只统计有效帧）
        
        Args:
            indices: (B, T, L) 码本索引
            labels: (B,) 保留参数兼容性，无条件模型不使用
            valid_mask: (B, T) bool tensor，标记有效帧
        
        Returns:
            total_bits: scalar tensor
        """
        nll_per_sample = self.compute_nll(indices, labels, valid_mask)
        return nll_per_sample.sum()
    
    def compute_rate_bps(
        self,
        indices: torch.Tensor,
        frame_rate_hz: float = 50.0,
        labels: Optional[torch.Tensor] = None,  # 保留参数兼容性
        valid_mask: Optional[torch.Tensor] = None
    ) -> float:
        """
        计算码率（bits per second，只统计有效帧）
        
        Args:
            indices: (B, T, L) 码本索引
            frame_rate_hz: 帧率（Hz）
            labels: (B,) 保留参数兼容性，无条件模型不使用
            valid_mask: (B, T) bool tensor，标记有效帧
        
        Returns:
            rate_bps: 码率（bps）
        """
        B, T, L = indices.shape
        
        total_bits = self.compute_bits(indices, labels, valid_mask).item()
        
        # 计算有效时长（只统计有效帧）
        if valid_mask is not None:
            total_valid_frames = valid_mask.sum().item()
        else:
            total_valid_frames = B * T
        
        total_time_sec = total_valid_frames / frame_rate_hz
        
        rate_bps = total_bits / total_time_sec if total_time_sec > 0 else 0.0
        return rate_bps
    
    def predict_next_token_prob(
        self,
        indices_history: torch.Tensor,
        labels: Optional[torch.Tensor] = None  # 保留参数兼容性
    ) -> torch.Tensor:
        """
        预测下一个token的概率分布（用于ECVQ判决）
        
        Args:
            indices_history: (B, L') 已有的索引序列（L' < L）
            labels: (B,) 保留参数兼容性，无条件模型不使用
        
        Returns:
            probs: (B, V) 下一个token的概率分布
        """
        # ChatGPT建议：推理模式包装整个函数
        with torch.inference_mode():
            B, L_prime = indices_history.shape
            
            # 统一SKIP
            seq = torch.where(indices_history < 0,
                             torch.full_like(indices_history, self.SKIP_ID),
                             indices_history)
            
            # 添加BOS
            inp = torch.cat([torch.full((B, 1), self.BOS_ID, device=seq.device, dtype=seq.dtype),
                            seq], dim=1)  # (B, L'+1)
            
            # Embedding
            x = self.token_embedding(inp)
            x = self.pos_encoding(x)
            
            # Group/layer嵌入（只到 L'+1）
            x = x + self.group_embedding(self.group_ids[:L_prime+1]).unsqueeze(0)
            x = x + self.layer_embedding(self.layer_ids[:L_prime+1]).unsqueeze(0)
            
            # Transformer（因果掩码）
            # ChatGPT建议：使用缓存mask（兼容旧checkpoint）
            if hasattr(self, '_get_causal_mask'):
                mask = self._get_causal_mask(L_prime + 1, x.device)
            else:
                mask = self.generate_causal_mask(L_prime + 1, x.device)
            x = self.transformer(x, mask=mask)
            
            # 只取最后一个位置
            logits = self.output_proj(x[:, -1, :])  # (B, V)
            probs = F.softmax(logits, dim=-1)
            
            return probs
    
    @staticmethod
    def generate_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
        """
        生成因果掩码（上三角，float additive mask）
        
        Args:
            seq_len: 序列长度
            device: 设备
        
        Returns:
            mask: (seq_len, seq_len) float tensor，-inf 表示屏蔽
        """
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        mask = mask.masked_fill(mask == 0, 0.0)
        return mask  # (L, L) float，更兼容不同 PyTorch 版本


def create_entropy_model(config, rvq_config):
    """
    创建无条件熵模型 q(z)
    
    Args:
        config: EntropyModelConfig实例
        rvq_config: GroupedRVQConfig实例
    
    Returns:
        模型实例
    """
    logger.info("创建无条件熵模型 q(z)")
    return AutoregressiveEntropyModel(config, rvq_config)


# ============================================================================
# 训练工具
# ============================================================================

def train_entropy_model_step(
    model: AutoregressiveEntropyModel,
    indices: torch.Tensor,
    labels: Optional[torch.Tensor],
    optimizer: torch.optim.Optimizer
) -> dict:
    """
    单步训练熵模型
    
    Args:
        model: 熵模型
        indices: (B, T, L) 码本索引
        labels: (B,) 保留参数兼容性，无条件模型不使用
        optimizer: 优化器
    
    Returns:
        metrics: dict 训练指标
    """
    model.train()
    optimizer.zero_grad()
    
    # 前向传播
    logits, targets = model(indices, labels)
    
    # 计算损失
    logits_flat = logits.reshape(-1, model.V)
    targets_flat = targets.reshape(-1)
    
    loss = F.cross_entropy(logits_flat, targets_flat)
    
    # 反向传播
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), model.config.max_grad_norm)
    optimizer.step()
    
    # 计算准确率
    with torch.inference_mode():
        preds = logits_flat.argmax(dim=-1)
        acc = (preds == targets_flat).float().mean()
    
    metrics = {
        'loss': loss.item(),
        'acc': acc.item(),
        'bpf': loss.item() / math.log(2),  # bits per frame（近似）
    }

    return metrics
