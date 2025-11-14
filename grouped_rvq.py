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
                    
                    if self.config.use_full_ranking_ecvq:
                        # 使用全量/Top-N ECVQ重排序（真·ECVQ）
                        BT = B * T
                        L_prime = indices_history.shape[-1]
                        
                        # 预测下一token概率
                        history_flat = indices_history.reshape(BT, L_prime)
                        labels_expanded = labels.unsqueeze(1).repeat(1, T).reshape(BT) if labels is not None else None
                        probs = entropy_model.predict_next_token_prob(history_flat, labels_expanded)
                        probs = probs.view(B, T, -1)  # (B, T, K+1)
                        
                        # 获取码本矩阵
                        E = self._get_codebook_weight(vq).to(residual.device, residual.dtype)
                        
                        # Top-N配置（可选）
                        topn = getattr(self.config, 'ecvq_topk', None)
                        
                        # 全量重排序
                        indices_with_skip, quantized_with_skip = self._ecvq_rerank(
                            residual=residual,
                            probs=probs,
                            E=E,
                            lambda_rate=lambda_rate,
                            valid_bt=valid_bt,
                            topk=topn
                        )
                    else:
                        # 使用原始Top-1+SKIP方法（向后兼容）
                        vq_codebook = self._get_codebook_weight(vq)
                        indices_with_skip, quantized_with_skip = self._ecvq_decision_with_entropy(
                            residual=residual,
                            quantized=quantized,
                            indices=indices,
                            entropy_model=entropy_model,
                            indices_history=indices_history,
                            labels=labels,
                            lambda_rate=lambda_rate,
                            valid_bt=valid_bt,
                            codebook=vq_codebook,
                            use_full_ranking=False
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
    
    def _get_codebook_weight(self, vq: nn.Module) -> torch.Tensor:
        """
        返回该VQ层的码本矩阵 E: (K, Dg)
        兼容不同vector_quantize_pytorch版本的字段命名
        """
        # 尝试常见的码本字段名
        candidates = []
        
        # 1. 直接字段
        for name in ['codebook', 'embedding', 'embeddings', 'embed']:
            if hasattr(vq, name):
                obj = getattr(vq, name)
                if hasattr(obj, 'weight'):
                    candidates.append(obj.weight)
                elif torch.is_tensor(obj):
                    candidates.append(obj)
        
        # 2. _codebook 子对象（当前版本）
        if hasattr(vq, '_codebook'):
            cb = getattr(vq, '_codebook')
            for name in ['embed', 'weight', 'embedding']:
                if hasattr(cb, name):
                    w = getattr(cb, name)
                    if torch.is_tensor(w):
                        candidates.append(w)
        
        # 3. EMA版本可能在ema_codebook下
        if hasattr(vq, 'ema_codebook'):
            ema = getattr(vq, 'ema_codebook')
            for name in ['embedding', 'embed', 'weight']:
                if hasattr(ema, name):
                    w = getattr(ema, name)
                    if hasattr(w, 'weight'):
                        candidates.append(w.weight)
                    elif torch.is_tensor(w):
                        candidates.append(w)
        
        # 返回第一个找到的tensor
        for w in candidates:
            if torch.is_tensor(w):
                # 处理多码本情况：(num_codebooks, K, D) → 取第一个码本 → (K, D)
                if w.ndim == 3 and w.size(0) == 1:
                    return w[0]  # (K, D)
                elif w.ndim == 2:
                    return w  # (K, D)
        
        raise RuntimeError(f"无法定位VQ码本权重，尝试的字段: {list(vars(vq).keys())}")
    
    @torch.inference_mode()
    def _ecvq_rerank(
        self,
        residual: torch.Tensor,          # (B, T, Dg)
        probs: torch.Tensor,             # (B, T, K+1) 来自q(k|history)，最后一列是SKIP
        E: torch.Tensor,                 # (K, Dg) 码本矩阵
        lambda_rate: torch.Tensor,       # 标量
        valid_bt: torch.Tensor,          # (B, T) 有效帧mask
        topk: Optional[int] = None       # 若给定，则先取欧氏距离Top-N
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        全量/Top-N ECVQ重排序：在所有码字+SKIP中选择率失真最优
        
        真正的ECVQ判决：J(k) = ΔD(k) + λ·(-log₂ q(k|ctx))
        """
        B, T, Dg = residual.shape
        K = E.size(0)
        
        # 计算所有码字的ΔD(k) = ||e_k||² - 2·r·e_k（忽略||r||²，不影响argmin）
        r = residual.reshape(B * T, Dg)                      # (BT, Dg)
        E_norm2 = (E ** 2).sum(dim=-1)                       # (K,)
        
        # 距离矩阵（向量化计算）
        D_full = E_norm2.unsqueeze(0) - 2.0 * (r @ E.t())   # (BT, K)
        D_full = D_full.view(B, T, K)                        # (B, T, K)
        
        if topk is not None and topk < K:
            # Top-N折中：先按欧氏距离挑Top-N，再做率失真重排
            _, idx_topk = torch.topk(-D_full, k=topk, dim=-1)  # 负号：距离小优先
            
            # 从probs中取对应候选
            p_codes = probs[..., :K].gather(-1, idx_topk).clamp_min(1e-12)  # (B, T, N)
            bits_codes = -torch.log2(p_codes)
            
            # 候选的ΔD
            D_codes = D_full.gather(-1, idx_topk)
            
            # 组合J(k) = D(k) + λ·bits(k)
            J_codes = D_codes + lambda_rate * bits_codes                     # (B, T, N)
            
            # SKIP的代价
            bits_skip = -torch.log2(probs[..., K].clamp_min(1e-12))          # (B, T)
            J_skip = bits_skip * lambda_rate                                  # (B, T)
            
            # 合并SKIP
            J_all = torch.cat([J_codes, J_skip.unsqueeze(-1)], dim=-1)       # (B, T, N+1)
            k_star_local = J_all.argmin(dim=-1)                              # (B, T)
            
            # 将局部Top-N索引还原到全局0..K映射
            chosen_is_skip = (k_star_local == topk)
            idx_global = torch.zeros_like(k_star_local)
            
            # 使用高级索引批量还原
            mask_send = ~chosen_is_skip
            if mask_send.any():
                b_idx, t_idx = torch.where(mask_send)
                idx_global[b_idx, t_idx] = idx_topk[b_idx, t_idx, k_star_local[b_idx, t_idx]]
            idx_global[chosen_is_skip] = K  # 用K表示SKIP
            
        else:
            # 全量重排序
            bits_codes = -torch.log2(probs[..., :K].clamp_min(1e-12))        # (B, T, K)
            J_codes = D_full + lambda_rate * bits_codes                       # (B, T, K)
            
            # SKIP的代价（失真=0）
            bits_skip = -torch.log2(probs[..., K].clamp_min(1e-12))           # (B, T)
            J_skip = bits_skip * lambda_rate                                   # (B, T)
            
            # 拼接并选择最优
            J_all = torch.cat([J_codes, J_skip.unsqueeze(-1)], dim=-1)        # (B, T, K+1)
            idx_global = J_all.argmin(dim=-1)                                  # (B, T); 0..K，K=SKIP
        
        # 写回索引与量化向量
        chosen_is_skip = (idx_global == K)
        
        indices_final = idx_global.clone()                                     # (B, T)
        indices_final[chosen_is_skip] = self.skip_token_id                     # 标记为SKIP
        
        quantized_final = torch.zeros_like(residual)                           # (B, T, Dg)
        
        # 向量化构造量化向量
        if (~chosen_is_skip).any():
            mask_send = ~chosen_is_skip
            quantized_final[mask_send] = E[idx_global[mask_send]]
        
        # 屏蔽无效帧
        if not valid_bt.all():
            quantized_final[~valid_bt] = 0.0
            indices_final[~valid_bt] = self.skip_token_id
        
        return indices_final, quantized_final
    
    @torch.inference_mode()
    def _ecvq_decision_with_entropy(
        self,
        residual: torch.Tensor,
        quantized: torch.Tensor,
        indices: torch.Tensor,
        entropy_model: nn.Module,
        indices_history: torch.Tensor,
        labels: Optional[torch.Tensor],
        lambda_rate: torch.Tensor,
        valid_bt: torch.Tensor,
        codebook: torch.Tensor,  # 新增：码本 (K, Dg)
        use_full_ranking: bool = True  # 新增：是否使用全量重排序
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        ECVQ 判决（使用真实熵模型）
        
        支持两种模式：
        1. use_full_ranking=True: 全量重排序（真·ECVQ）
        2. use_full_ranking=False: Top-1+SKIP近似（原始方法）
        
        Args:
            residual: (B, T, Dg) 当前残差
            quantized: (B, T, Dg) VQ量化后的向量
            indices: (B, T) VQ返回的码本索引
            entropy_model: 训练好的熵模型
            indices_history: (B, T, L') 帧内已决策的索引历史
            labels: (B,) 情感标签（条件模型需要）
            lambda_rate: 拉格朗日乘子 λ
            valid_bt: (B, T) bool mask
            codebook: (K, Dg) 当前层的码本
            use_full_ranking: 是否使用全量重排序
        
        Returns:
            final_indices: (B, T) 最终决策（包含SKIP）
            final_quantized: (B, T, Dg) 最终量化向量
        """
        B, T, Dg = residual.shape
        K = codebook.shape[0]  # 码本大小
        
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
        
        if use_full_ranking:
            # ===== 全量重排序（真·ECVQ）=====
            # 1) 计算所有码字的失真
            r_flat = residual.view(BT, Dg)  # (BT, Dg)
            
            # ||e_k||^2: (K,)
            codebook_norm2 = (codebook ** 2).sum(dim=-1)  # (K,)
            
            # ||r - e_k||^2 = ||r||^2 + ||e_k||^2 - 2*r·e_k
            r_norm2 = (r_flat ** 2).sum(dim=-1, keepdim=True)  # (BT, 1)
            inner_prod = r_flat @ codebook.T  # (BT, K)
            D_all = r_norm2 + codebook_norm2.unsqueeze(0) - 2 * inner_prod  # (BT, K)
            D_all = D_all.view(B, T, K)  # (B, T, K)
            
            # SKIP的失真（不发送任何码字）
            D_skip = (residual ** 2).sum(dim=-1)  # (B, T)
            
            # 2) 计算所有候选的bits
            bits_codes = -torch.log2(probs[:, :, :K].clamp_min(1e-12))  # (B, T, K)
            bits_skip = -torch.log2(probs[:, :, self.skip_token_id].clamp_min(1e-12))  # (B, T)
            
            # 3) 计算率失真代价 J(k) = D(k) + λ·bits(k)
            J_codes = D_all + lambda_rate.unsqueeze(-1) * bits_codes  # (B, T, K)
            J_skip = D_skip + lambda_rate * bits_skip  # (B, T)
            
            # 4) 拼接SKIP并选择最优
            J_all = torch.cat([J_codes, J_skip.unsqueeze(-1)], dim=-1)  # (B, T, K+1)
            k_star = J_all.argmin(dim=-1)  # (B, T)
            
            # 5) 应用决策
            chosen_is_skip = (k_star == K)
            
            final_indices = k_star.clone()
            final_indices[chosen_is_skip] = self.skip_token_id
            
            # 构造量化向量
            final_quantized = torch.zeros_like(residual)
            for b in range(B):
                for t in range(T):
                    if not chosen_is_skip[b, t]:
                        final_quantized[b, t] = codebook[k_star[b, t]]
            
        else:
            # ===== Top-1+SKIP近似（原始方法）=====
            # 计算失真
            distortion_send = ((residual - quantized) ** 2).sum(dim=-1)  # (B, T)
            distortion_skip = (residual ** 2).sum(dim=-1)  # (B, T)
            delta_distortion = distortion_skip - distortion_send  # (B, T)
            
            # 尺度归一化
            energy = residual.pow(2).mean(dim=-1).clamp_min(1e-6)  # (B, T)
            delta_distortion = delta_distortion / energy
        
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

