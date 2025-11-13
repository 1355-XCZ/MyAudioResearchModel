"""
分组RVQ + SKIP机制实现
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List
import numpy as np
import math
import logging

from vector_quantize_pytorch import VectorQuantize

logger = logging.getLogger(__name__)


class GroupedResidualVQ(nn.Module):
    """
    分组残差向量量化 + SKIP机制
    
    架构：
        768维 → 12组 × 64维
        每组：
            (可选) Coarse层 (K=256)
            Fine层 1-3 (K=128 + SKIP)
    """
    
    def __init__(self, config):
        """
        Args:
            config: GroupedRVQConfig实例
        """
        super().__init__()
        self.config = config
        
        assert config.feature_dim % config.num_groups == 0, \
            f"feature_dim ({config.feature_dim}) 必须被 num_groups ({config.num_groups}) 整除"
        
        # 分组参数
        self.feature_dim = config.feature_dim
        self.num_groups = config.num_groups
        self.group_dim = config.feature_dim // config.num_groups
        
        # 组内细层（每组独立，使用EMA模式）
        self.fine_vqs = nn.ModuleList()
        for g in range(config.num_groups):
            group_vqs = nn.ModuleList()
            for m in range(config.num_fine_layers):
                vq = VectorQuantize(
                    dim=self.group_dim,
                    codebook_size=config.fine_codebook_size,
                    decay=config.decay,  # EMA模式：码本自动更新，不需要优化器
                    commitment_weight=config.commitment_weight,
                    kmeans_init=config.kmeans_init,
                    kmeans_iters=config.kmeans_iters,
                    threshold_ema_dead_code=config.threshold_ema_dead_code,
                )
                group_vqs.append(vq)
            self.fine_vqs.append(group_vqs)
        
        # SKIP机制（修复：使用非负id）
        self.enable_skip = config.enable_skip
        self.skip_token_id = config.fine_codebook_size  # K 作为SKIP id（非负）
        
        logger.info(f"✅ GroupedResidualVQ 初始化:")
        logger.info(f"   - 特征维度: {self.feature_dim}")
        logger.info(f"   - 分组: {self.num_groups} 组 × {self.group_dim} 维")
        logger.info(f"   - 细层: 每组 {config.num_fine_layers} 层 × {config.fine_codebook_size} 码")
        logger.info(f"   - SKIP: {'启用' if self.enable_skip else '禁用'}")
    
    def forward(
        self,
        x: torch.Tensor,
        lambda_rate: Optional[torch.Tensor] = None,
        entropy_model: Optional[nn.Module] = None,
        valid_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        target_bpf: Optional[float] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """
        前向传播（支持ECVQ + padding屏蔽 + 真实熵模型判决）
        
        Args:
            x: (B, T, feature_dim) 输入特征
            lambda_rate: (scalar) 拉格朗日乘子 λ（用于ECVQ）
            entropy_model: 熵模型 q(k|context)（用于计算真实码长）
            valid_mask: (B, T) bool/byte tensor，标记有效帧（屏蔽padding）
            labels: (B,) 情感标签（用于条件熵模型）
        
        Returns:
            quantized: (B, T, feature_dim) 量化后特征
            indices: (B, T, num_total_layers) 码本索引（包含SKIP）
            commit_loss: (scalar) commitment损失
            stats: dict 统计信息
        """
        B, T, D = x.shape
        assert D == self.feature_dim
        
        # 处理valid_mask（屏蔽padding帧）
        if valid_mask is None:
            valid_bt = torch.ones(B, T, dtype=torch.bool, device=x.device)
        else:
            valid_bt = valid_mask.to(torch.bool)
        
        # 重塑为分组形式: (B, T, num_groups, group_dim)
        x_grouped = x.view(B, T, self.num_groups, self.group_dim)
        
        # 存储结果
        all_indices = []        # 所有层的索引
        all_commit_losses = []  # 所有层的commitment loss
        reconstructed_grouped = torch.zeros_like(x_grouped)
        
        total_skips = 0
        total_codes = 0
        
        # 健康监控：残差能量（每层）
        residual_energies = []  # 用于诊断"层是否学到有效方向"
        
        # ===== 组内细层（残差量化 + SKIP）=====
        # 预分配帧内历史缓存（用于熵模型判决）
        total_layers = self.num_groups * self.config.num_fine_layers
        all_indices_buffer = torch.full(
            (B, T, total_layers), self.skip_token_id,
            device=x.device, dtype=torch.long
        )
        
        for g in range(self.num_groups):
            # 当前组的输入
            x_g = x_grouped[:, :, g, :]  # (B, T, group_dim)
            
            # 残差初始化
            residual = x_g.clone()
            group_reconstructed = torch.zeros_like(x_g)
            
            # 逐层量化
            for m, vq in enumerate(self.fine_vqs[g]):
                # 带mask的量化（只对有效帧更新码本）
                quantized, indices, commit_loss = self._quantize_with_mask(vq, residual, valid_bt)
                
                # 当前位置（帧内顺序：group外层，layer内层）
                current_pos = g * self.config.num_fine_layers + m
                
                # ECVQ判决（使用真实熵模型）
                if self.enable_skip and lambda_rate is not None and entropy_model is not None:
                    # 获取帧内历史（当前位置之前的所有token）
                    indices_history = all_indices_buffer[:, :, :current_pos]  # (B, T, L')
                    
                    # 使用真实熵模型做ECVQ判决
                    indices_with_skip, quantized_with_skip = self._ecvq_decision_with_entropy(
                        residual=residual,
                        quantized=quantized,
                        indices=indices,
                        entropy_model=entropy_model,
                        indices_history=indices_history,
                        labels=labels,
                        lambda_rate=lambda_rate,
                        valid_bt=valid_bt
                    )
                    
                    skip_mask = (indices_with_skip == self.skip_token_id) & valid_bt
                    total_skips += skip_mask.sum().item()
                    total_codes += valid_bt.sum().item()
                    
                elif self.enable_skip and (target_bpf is not None):
                    # 分位数门控：按目标bpf控制SKIP率（朋友方案）
                    max_bpf = self.num_groups * self.config.num_fine_layers * math.log2(self.config.fine_codebook_size)
                    target_skip = float(1.0 - max(0.0, min(target_bpf, max_bpf)) / max_bpf)
                    
                    # ΔD归一化
                    distortion_send = torch.sum((residual - quantized) ** 2, dim=-1)  # (B, T)
                    distortion_skip = torch.sum(residual ** 2, dim=-1)  # (B, T)
                    denom = (distortion_skip + 1e-8)
                    delta = (distortion_skip - distortion_send) / denom  # (B, T), in [0,1]
                    
                    # 只在有效帧上计算分位数阈值τ
                    valid_delta = delta[valid_bt]
                    if bool(valid_delta.numel() > 0):
                        tau = torch.quantile(valid_delta, q=target_skip)  # 标量
                        skip_mask = (delta <= tau) & valid_bt
                    else:
                        skip_mask = torch.zeros_like(valid_bt)
                    
                    indices_with_skip = indices.clone()
                    indices_with_skip[skip_mask] = self.skip_token_id
                    quantized_with_skip = quantized.clone()
                    quantized_with_skip[skip_mask] = 0.0
                    
                    total_skips += skip_mask.sum().item()
                    total_codes += valid_bt.sum().item()
                    
                elif self.enable_skip and lambda_rate is not None:
                    # 无熵模型：简化ECVQ（仅用失真，不考虑bits）
                    distortion_skip = torch.sum(residual ** 2, dim=-1)  # (B, T)
                    distortion_send = torch.sum((residual - quantized) ** 2, dim=-1)  # (B, T)
                    
                    # 无尺度化比率判决（更稳健）
                    skip_mask = ((distortion_skip - distortion_send) / (distortion_skip + 1e-8) < lambda_rate * 1e-2) & valid_bt
                    
                    indices_with_skip = indices.clone()
                    indices_with_skip[skip_mask] = self.skip_token_id
                    quantized_with_skip = quantized.clone()
                    quantized_with_skip[skip_mask] = 0.0
                    
                    total_skips += skip_mask.sum().item()
                    total_codes += valid_bt.sum().item()
                else:
                    # 标准RVQ（无SKIP）
                    indices_with_skip = indices
                    quantized_with_skip = quantized
                    total_codes += valid_bt.sum().item()
                
                # 更新历史缓存
                all_indices_buffer[:, :, current_pos] = indices_with_skip
                
                # 更新重建和残差
                all_indices.append(indices_with_skip)
                group_reconstructed += quantized_with_skip
                residual = residual - quantized_with_skip
                
                all_commit_losses.append(commit_loss)
                
                # 记录残差能量（健康监控，只统计有效帧）
                if bool(valid_bt.any().item()):
                    residual_energy = (residual[valid_bt] ** 2).mean().item()
                else:
                    residual_energy = 0.0
                residual_energies.append(residual_energy)
            
            # 存储该组的重建
            reconstructed_grouped[:, :, g, :] = group_reconstructed
        
        # ===== 3. 合并结果 =====
        # 重塑回原始形状
        quantized = reconstructed_grouped.view(B, T, self.feature_dim)
        
        # 合并索引: (B, T, num_groups * num_fine_layers)
        indices = torch.stack(all_indices, dim=-1)  # (B, T, num_total_layers)
        
        # 合并损失
        commit_loss = torch.stack(all_commit_losses).mean()
        
        # 统计信息
        skip_rate = total_skips / max(total_codes, 1) if total_codes > 0 else 0.0
        stats = {
            'skip_rate': skip_rate,
            'total_skips': total_skips,
            'total_codes': total_codes,
            'num_layers': len(all_indices),
            'residual_energies': residual_energies,  # 每层残差能量（健康监控）
        }
        
        return quantized, indices, commit_loss, stats
    
    def encode(
        self,
        x: torch.Tensor,
        lambda_rate: float = 1.0,
        entropy_model: Optional[nn.Module] = None,
        valid_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, dict]:
        """
        编码（获取索引和码率信息）
        
        Args:
            x: (B, T, feature_dim) 输入特征
            lambda_rate: 拉格朗日乘子
            entropy_model: 熵模型
            valid_mask: (B, T) bool tensor，标记有效帧
            labels: (B,) 情感标签（用于条件熵模型）
        
        Returns:
            indices: (B, T, num_total_layers) 码本索引
            info: dict 包含码率等信息
        """
        lambda_tensor = torch.tensor(lambda_rate, device=x.device, dtype=x.dtype)
        _, indices, _, stats = self.forward(x, lambda_tensor, entropy_model, valid_mask, labels)
        
        # 计算码率（如果有熵模型）
        if entropy_model is not None:
            bits = entropy_model.compute_bits(indices, labels=None, valid_mask=valid_mask)
            if valid_mask is not None:
                valid_frames = valid_mask.sum().item()
            else:
                B, T = indices.shape[:2]
                valid_frames = B * T
            bits_per_frame = bits.item() / max(valid_frames, 1)
        else:
            # 理论上界（均匀分布）
            import math
            bits_per_layer = math.log2(self.config.fine_codebook_size)
            bits_per_frame = bits_per_layer * stats['num_layers'] * (1 - stats['skip_rate'])
        
        info = {
            'indices': indices,
            'bits_per_frame': float(bits_per_frame),
            'skip_rate': stats['skip_rate'],
        }
        
        return indices, info
    
    def _quantize_with_mask(
        self, 
        vq: nn.Module, 
        residual: torch.Tensor, 
        valid_bt: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        带mask的量化（只对有效帧更新码本，避免padding污染）
        
        Args:
            vq: VectorQuantize模块
            residual: (B, T, group_dim) 残差
            valid_bt: (B, T) bool mask，标记有效帧
        
        Returns:
            quantized: (B, T, group_dim) 量化后的向量
            indices: (B, T) 码本索引
            commit_loss: commitment损失
        """
        B, T, Dg = residual.shape
        
        # 如果全部有效，直接调用VQ
        if bool(valid_bt.all().item()):
            quantized, indices, commit_loss = vq(residual)
            return quantized, indices, commit_loss
        
        # 如果全部无效，返回零
        if not bool(valid_bt.any().item()):
            quantized = torch.zeros_like(residual)
            indices = torch.full((B, T), self.skip_token_id, 
                                device=residual.device, dtype=torch.long)
            commit_loss = torch.tensor(0.0, device=residual.device)
            return quantized, indices, commit_loss
        
        # 部分有效：只对有效帧做量化
        flat_in = residual[valid_bt]  # (N_valid, Dg)
        q_flat, idx_flat, commit = vq(flat_in.unsqueeze(0))  # 添加batch维度
        q_flat = q_flat.squeeze(0)
        idx_flat = idx_flat.squeeze(0)
        
        # Scatter回(B, T, Dg) / (B, T)
        quantized = torch.zeros_like(residual)
        quantized[valid_bt] = q_flat
        
        indices = torch.full((B, T), self.skip_token_id, 
                            device=residual.device, dtype=torch.long)
        indices[valid_bt] = idx_flat
        
        return quantized, indices, commit
    
    @torch.inference_mode()  # ChatGPT建议：推理模式
    def _ecvq_decision_with_entropy(
        self,
        residual: torch.Tensor,
        quantized: torch.Tensor,
        indices: torch.Tensor,
        entropy_model: nn.Module,
        indices_history: torch.Tensor,
        labels: Optional[torch.Tensor],
        lambda_rate: torch.Tensor,
        valid_bt: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        ECVQ 判决（使用真实熵模型）
        
        Args:
            residual: (B, T, Dg) 当前残差
            quantized: (B, T, Dg) VQ量化后的向量
            indices: (B, T) VQ返回的码本索引
            entropy_model: 训练好的熵模型
            indices_history: (B, T, L') 帧内已决策的索引历史
            labels: (B,) 情感标签（条件模型需要）
            lambda_rate: 拉格朗日乘子 λ
            valid_bt: (B, T) bool mask
        
        Returns:
            final_indices: (B, T) 最终决策（包含SKIP）
            final_quantized: (B, T, Dg) 最终量化向量
        """
        B, T, Dg = residual.shape
        
        # 计算失真
        distortion_send = ((residual - quantized) ** 2).sum(dim=-1)  # (B, T)
        distortion_skip = (residual ** 2).sum(dim=-1)  # (B, T)
        delta_distortion = distortion_skip - distortion_send  # (B, T) 跳过节省的失真
        
        # 尺度归一化：按残差能量归一化ΔD，保证跨组比特分配公平性
        energy = residual.pow(2).mean(dim=-1).clamp_min(1e-6)  # (B, T)
        delta_distortion = delta_distortion / energy
        
        # 使用熵模型预测下一个token的概率分布
        BT = B * T
        L_prime = indices_history.shape[-1]
        
        # 构造扁平历史（允许 L'=0，即BOS-only先验）
        history_flat = indices_history.reshape(BT, L_prime)  # (B*T, L')
        
        # 如果是条件模型，扩展labels
        if entropy_model.condition_on_label and labels is not None:
            labels_expanded = labels.unsqueeze(1).repeat(1, T).reshape(BT)  # (B*T,)
        else:
            labels_expanded = None
        
        # 使用熵模型预测（L'==0 时即BOS-only先验，模型自己学习的分布）
        probs = entropy_model.predict_next_token_prob(history_flat, labels_expanded)
        probs = probs.view(B, T, -1)  # (B, T, V)
        
        # 计算 bits_send（发送码字k）
        indices_clamped = indices.clamp(0, probs.shape[-1] - 1)
        probs_send = probs.gather(-1, indices_clamped.unsqueeze(-1)).squeeze(-1)  # (B, T)
        probs_send = probs_send.clamp_min(1e-12)
        bits_send = -torch.log2(probs_send)  # (B, T)
        
        # 计算 bits_skip（跳过）
        probs_skip = probs[:, :, self.skip_token_id].clamp_min(1e-12)  # (B, T)
        bits_skip = -torch.log2(probs_skip)  # (B, T)
        
        # Δbits = bits_send - bits_skip
        delta_bits = bits_send - bits_skip  # (B, T)
        
        # ECVQ判决：当 ΔD < λ·Δbits 时选择SKIP（收益不足）
        should_skip = (delta_distortion < lambda_rate * delta_bits)
        should_skip = should_skip & valid_bt  # 只在有效帧上判决
        
        # 应用决策
        final_indices = indices.clone()
        final_indices[should_skip] = self.skip_token_id
        
        final_quantized = quantized.clone()
        final_quantized[should_skip] = 0.0
        
        return final_indices, final_quantized
    
    def decode(self, indices: torch.Tensor) -> torch.Tensor:
        """
        解码（从索引重建特征）
        
        Args:
            indices: (B, T, num_total_layers) 码本索引
        
        Returns:
            reconstructed: (B, T, feature_dim) 重建特征
        
        Raises:
            NotImplementedError: 此方法尚未实现
        """
        raise NotImplementedError(
            "GroupedResidualVQ.decode 尚未实现（需要逐层码本查表与累加残差）。\n"
            "训练阶段不需要 decode，为避免误用这里直接抛错。\n"
            "实现时需要：从 VQ 的 codebook 中查找嵌入向量，逐层累加残差。"
        )
    
    def get_codebook_usage(self, indices: torch.Tensor) -> dict:
        """
        计算码本利用率
        
        Args:
            indices: (B, T, num_layers) 码本索引
        
        Returns:
            usage_stats: dict 每层的利用率统计
        """
        B, T, num_layers = indices.shape
        usage_stats = {}
        
        for layer in range(num_layers):
            layer_indices = indices[:, :, layer].reshape(-1).cpu()  # 移到CPU处理
            
            # 排除SKIP（统一为 K）
            non_skip = layer_indices[layer_indices != self.skip_token_id]
            
            if non_skip.numel() > 0:
                unique_codes = torch.unique(non_skip)
                utilization = unique_codes.numel() / self.config.fine_codebook_size
            else:
                utilization = 0.0
            
            usage_stats[f'layer_{layer}'] = {
                'utilization': float(utilization),
                'unique_codes': int(unique_codes.numel()) if non_skip.numel() > 0 else 0,
                'total_codes': self.config.fine_codebook_size,
                'skip_rate': float((layer_indices == self.skip_token_id).sum().item()) / max(layer_indices.numel(), 1),
            }
        
        return usage_stats


def create_grouped_rvq(config) -> GroupedResidualVQ:
    """
    创建分组RVQ模型
    
    Args:
        config: GroupedRVQConfig实例
    
    Returns:
        模型实例
    """
    return GroupedResidualVQ(config)

