"""
训练无条件熵模型 q(z)（发布版）
仅训练 q(z)，不包含条件模型 q(z|y)
"""

import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
import logging
import json
import math
import random
import numpy as np

from config import get_default_config
from grouped_rvq import GroupedResidualVQ
from entropy_model import create_entropy_model
from data_loader import create_dataloaders

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def set_seed(seed: int):
    """设置随机种子保证可复现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    logger.info(f"✅ 已设置随机种子: {seed}")


def masked_token_kl(student_logits, teacher_logits, valid_mask, temperature=1.0):
    """
    计算 token 级 KL(student || teacher)，仅统计有效帧。
    logits: (B*T*L, V)
    valid_mask: (B, T) → 展平后重复 L 次
    """
    BTL, V = student_logits.shape
    # 温度平滑
    t = temperature
    s_logprob = F.log_softmax(student_logits / t, dim=-1)
    t_prob     = F.softmax(teacher_logits / t, dim=-1).detach()

    kl = F.kl_div(s_logprob, t_prob, reduction='none', log_target=False)  # (BTL, V)
    kl = kl.sum(dim=-1)  # (BTL,)

    # 展平 mask
    # valid_mask: (B, T) → (B, T, L) → (BTL,)
    B = valid_mask.size(0); T = valid_mask.size(1)
    L = BTL // (B * T)
    mask_tokens = valid_mask.unsqueeze(-1).expand(-1, -1, L).reshape(-1).float()
    if mask_tokens.sum() == 0:
        return student_logits.new_tensor(0.0)
    return (kl * mask_tokens).sum() / mask_tokens.sum()


def extract_indices_from_rvq(rvq_model, dataloader, device, target_bpf=None, ecvq_lambda=None):
    """
    用训练好的RVQ提取所有样本的离散索引（支持带SKIP的ECVQ）
    
    Args:
        rvq_model: 训练好的GroupedResidualVQ
        dataloader: 数据加载器
        device: 设备
        target_bpf: 目标码率（bits per frame），使用分位数门控（复杂方式）
        ecvq_lambda: 直接指定λ值（简单方式，优先使用）
    
    Returns:
        all_indices: List of (T, L) tensors (uint8格式)
        all_labels: List of scalar labels
        all_valid_masks: List of (T,) bool tensors
    """
    rvq_model.eval()
    all_indices = []
    all_labels = []
    all_valid_masks = []
    
    logger.info("提取离散索引...")
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="编码"):
            features = batch['features'].to(device, non_blocking=True)
            labels = batch['labels'].to(device, non_blocking=True)
            lengths = batch['lengths'].to(device, non_blocking=True)
            
            B, T, D = features.shape
            t_idx = torch.arange(T, device=device).unsqueeze(0)
            valid_bt = (t_idx < lengths.unsqueeze(1))  # (B, T)
            
            # 用RVQ编码
            if ecvq_lambda is not None:
                # 简单方式：直接用λ值（旧代码方式）
                _, indices, _, _ = rvq_model(
                    features,
                    lambda_rate=torch.tensor(ecvq_lambda, device=device),
                    entropy_model=None,  # 关键：走简化ECVQ分支
                    valid_mask=valid_bt
                )
            else:
                # 分位数门控方式（新方式）
            _, indices, _, _ = rvq_model(
                features,
                lambda_rate=None,
                entropy_model=None,
                valid_mask=valid_bt,
                target_bpf=target_bpf  # 使用目标bpf分位数门控
            )
            
            # 逐样本存储（保留变长，用uint8节省内存）
            for i in range(B):
                T_i = int(lengths[i].item())
                # 关键修复：索引范围0-128，用uint8存储（8倍瘦身）
                idx_i = indices[i, :T_i, :].to(torch.uint8).cpu()  # (T_i, L)
                all_indices.append(idx_i)
                all_labels.append(labels[i].cpu())  # scalar
                all_valid_masks.append(torch.ones(T_i, dtype=torch.bool))  # (T_i,)
    
    logger.info(f"✅ 提取完成: {len(all_indices)} 个样本")
    return all_indices, all_labels, all_valid_masks


def train_entropy_model(
    entropy_model,
    all_indices,
    all_labels,
    all_valid_masks,
    training_config,
    entropy_config,
    model_name="q_z",
    teacher_model=None  # 新增：传入 q(z) 作为 teacher（仅 q(z|y) 时使用）
):
    """
    训练熵模型（支持带KL牵引的条件模型）

    Args:
        entropy_model: AutoregressiveEntropyModel 或 ConditionalEntropyModel
        all_indices: List of (T, L) 离散索引
        all_labels: List of labels
        all_valid_masks: List of (T,) bool masks
        training_config: 训练配置
        entropy_config: 熵模型配置
        model_name: 模型名称 ("q_z" 或 "q_z_y")
        teacher_model: 可选的teacher模型（q(z)），用于KL牵引q(z|y)
    """
    device = torch.device(training_config.device if torch.cuda.is_available() else 'cpu')
    entropy_model = entropy_model.to(device)
    if teacher_model is not None:
        teacher_model = teacher_model.to(device)
        teacher_model.eval()  # teacher 固定

    # 只优化需要训练的参数（支持"从 q(z) 微调"）
    optim_params = [p for p in entropy_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        optim_params,
        lr=entropy_config.learning_rate,
        weight_decay=entropy_config.weight_decay
    )
    
    # 创建目录
    checkpoint_dir = Path(training_config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 80)
    logger.info(f"开始训练熵模型: {model_name}")
    logger.info(f"  - 样本数: {len(all_indices)}")
    logger.info(f"  - 词表大小: {entropy_model.V}")
    logger.info(f"  - 帧内序列长度: {entropy_model.L}")
    logger.info("=" * 80)
    
    best_loss = float('inf')
    start_epoch = 1
    num_samples = len(all_indices)
    
    # 早停机制
    patience_counter = 0
    best_epoch = 0
    
    # 根据配置决定是否从检查点恢复训练（只加载同一实验的检查点）
    if training_config.resume_from_checkpoint:
        checkpoint_path = checkpoint_dir / f"{model_name}_best.pt"
        if checkpoint_path.exists():
            logger.info(f"从当前实验目录恢复训练: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            entropy_model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            best_loss = checkpoint.get('loss', float('inf'))
            start_epoch = checkpoint.get('epoch', 1) + 1
            logger.info(f"✅ 已恢复训练: epoch={start_epoch}, best_loss={best_loss:.4f}")
        else:
            logger.warning(
                f"⚠️  配置要求从检查点恢复，但当前实验目录中不存在: {checkpoint_path}\n"
                f"   将从头训练。如需从头训练，设置 TrainingConfig.resume_from_checkpoint=False"
            )
    else:
        logger.info("🚀 从头训练熵模型（不加载检查点）")
    
    # 应用数据比例（用于快速实验）
    if entropy_config.train_data_fraction < 1.0:
        subset_size = int(num_samples * entropy_config.train_data_fraction)
        all_indices = all_indices[:subset_size]
        all_labels = all_labels[:subset_size]
        all_valid_masks = all_valid_masks[:subset_size]
        num_samples = len(all_indices)
        logger.info(f"⚙️  使用训练数据的 {entropy_config.train_data_fraction:.1%}，共 {num_samples} 个样本")
    
    batch_size = entropy_config.batch_size
    num_epochs = entropy_config.num_epochs
    
    # 获取实际使用的优化配置（考虑总开关）
    opt_cfg = entropy_config.get_effective_config()
    
    # 日志打印（使用边框展示层级）
    logger.info("")
    logger.info("┌" + "─" * 78 + "┐")
    logger.info("│ ⚙️  熵模型训练配置" + " " * 59 + "│")
    logger.info("├" + "─" * 78 + "┤")
    logger.info(f"│  基础参数:                                                              │")
    logger.info(f"│    • batch_size: {batch_size:<58} │")
    logger.info(f"│    • num_epochs: {num_epochs:<58} │")
    logger.info(f"│    • 训练样本数: {num_samples} ({entropy_config.train_data_fraction:.0%})" + " " * (70 - len(f"{num_samples} ({entropy_config.train_data_fraction:.0%})") - 18) + "│")
    logger.info(f"│                                                                              │")
    
    opt_status = '✅ 启用' if entropy_config.enable_optimizations else '❌ 禁用'
    logger.info(f"│  性能优化:                                                              │")
    logger.info(f"│    • 总开关: {opt_status:<62} │")
    
    if entropy_config.enable_optimizations:
        max_frames = opt_cfg['max_frames_per_sample']
        desc_max_frames = '不限制' if max_frames == 0 else f'限制到{max_frames}帧'
        logger.info(f"│    • 帧数裁剪: {max_frames} ({desc_max_frames})" + " " * (70 - len(f"{max_frames} ({desc_max_frames})") - 16) + "│")
        
        stride = opt_cfg['frame_stride']
        desc_stride = '全保留' if stride == 1 else f'每{stride}帧取1帧'
        logger.info(f"│    • 帧下采样: {stride} ({desc_stride})" + " " * (70 - len(f"{stride} ({desc_stride})") - 16) + "│")
        
        amp_status = '✅ BF16' if opt_cfg['use_amp'] else '❌ FP32'
        logger.info(f"│    • 混合精度: {amp_status:<58} │")
        
        tf32_status = '✅ 启用' if opt_cfg['use_tf32'] else '❌ 禁用'
        logger.info(f"│    • TF32加速: {tf32_status:<58} │")
    
    logger.info("└" + "─" * 78 + "┘")
    logger.info("")
    
    # 启用TF32加速（A100/H100）
    if opt_cfg['use_tf32'] and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    
    # 用于记录是否已经警告过空batch
    _empty_batch_warned = False
    
    for epoch in range(start_epoch, num_epochs + 1):
        entropy_model.train()
        total_loss = 0
        total_ce_loss = 0
        total_kl_loss = 0
        total_acc = 0
        num_batches = 0
        
        # 洗牌
        import random
        indices_list = list(range(num_samples))
        random.shuffle(indices_list)
        
        pbar = tqdm(range(0, num_samples, batch_size), desc=f"Epoch {epoch}/{num_epochs}")
        for start_idx in pbar:
            end_idx = min(start_idx + batch_size, num_samples)
            batch_indices_list = indices_list[start_idx:end_idx]
            
            # 收集batch（需要padding）
            batch_data = [all_indices[i] for i in batch_indices_list]
            batch_labels = torch.stack([all_labels[i] for i in batch_indices_list])
            batch_masks = [all_valid_masks[i] for i in batch_indices_list]
            
            # 性能优化：帧裁剪和下采样（根据总开关决定是否应用）
            if opt_cfg['max_frames_per_sample'] > 0 or opt_cfg['frame_stride'] > 1:
                cropped_data, cropped_masks, cropped_labels = [], [], []
                for idx, mask, label in zip(batch_data, batch_masks, batch_labels):
                    original_T = idx.shape[0]
                    
                    # 帧下采样（改进：如果原始序列长度小于stride，保留原序列）
                    if opt_cfg['frame_stride'] > 1:
                        if original_T >= opt_cfg['frame_stride']:
                            idx = idx[::opt_cfg['frame_stride']]
                            mask = mask[::opt_cfg['frame_stride']]
                        # 如果原始序列长度小于stride，保留原序列（避免变成空序列）
                        # 这样即使短序列也能被处理
                    
                    # 帧数裁剪
                    T_i = idx.shape[0]
                    if opt_cfg['max_frames_per_sample'] > 0 and T_i > opt_cfg['max_frames_per_sample']:
                        import random
                        start = random.randint(0, T_i - opt_cfg['max_frames_per_sample'])
                        idx = idx[start:start + opt_cfg['max_frames_per_sample']]
                        mask = mask[start:start + opt_cfg['max_frames_per_sample']]
                    
                    # 兜底：跳过空序列（理论上不应该发生，但保留以防万一）
                    if idx.shape[0] > 0:
                        cropped_data.append(idx)
                        cropped_masks.append(mask)
                        cropped_labels.append(label)
                
                batch_data = cropped_data
                batch_masks = cropped_masks
                batch_labels = torch.stack(cropped_labels) if cropped_labels else batch_labels[:0]
            
            # 跳过空batch（极端情况下所有样本都被过滤）
            if len(batch_data) == 0:
                # 记录警告（只在第一次遇到时）
                if not _empty_batch_warned:
                    logger.warning(
                        f"⚠️  Epoch {epoch}: 遇到空batch（所有样本都被过滤），"
                        f"可能是frame_stride={opt_cfg['frame_stride']}或数据过短。"
                        f"建议检查数据长度或调整frame_stride。"
                    )
                    _empty_batch_warned = True
                continue
            
            # Padding
            max_T = max(item.shape[0] for item in batch_data)
            L = batch_data[0].shape[1]
            B_batch = len(batch_data)
            
            indices_batch = torch.full((B_batch, max_T, L), 
                                      entropy_model.SKIP_ID, 
                                      dtype=torch.long)
            valid_mask_batch = torch.zeros(B_batch, max_T, dtype=torch.bool)
            
            for i, (idx, mask) in enumerate(zip(batch_data, batch_masks)):
                T_i = idx.shape[0]
                # 将uint8索引转回long用于embedding
                indices_batch[i, :T_i, :] = idx.to(torch.long)
                valid_mask_batch[i, :T_i] = mask
            
            # 搬运到GPU（优化：pin_memory + non_blocking）
            indices_batch = indices_batch.pin_memory().to(device, non_blocking=True)
            batch_labels = batch_labels.pin_memory().to(device, non_blocking=True)
            valid_mask_batch = valid_mask_batch.pin_memory().to(device, non_blocking=True)
            
            # 前向传播（根据总开关决定是否使用混合精度）
            optimizer.zero_grad()

            with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=opt_cfg['use_amp'] and torch.cuda.is_available()):
                if entropy_model.condition_on_label:
                    logits, targets = entropy_model(indices_batch, batch_labels)
                else:
                    logits, targets = entropy_model(indices_batch, None)

                # 计算损失（带mask）
                logits_flat = logits.reshape(-1, entropy_model.V)
                targets_flat = targets.reshape(-1)

                ce = F.cross_entropy(logits_flat, targets_flat, reduction='none')
                ce = ce.view(B_batch, max_T, L)

                # 可选增强：对SKIP位置加权
                # skip_weight < 1.0: 降低SKIP权重，鼓励正常编码（避免过度SKIP）
                # skip_weight > 1.0: 增加SKIP权重，提高p_skip（用于极低码率）
                # skip_weight = 1.0: 不加权（默认）
                skip_weight = getattr(entropy_config, "skip_weight", 1.0)
                if abs(skip_weight - 1.0) > 1e-6:  # 只要 != 1 就生效
                    targets_3d = targets.view(B_batch, max_T, L)
                    token_weights = torch.ones_like(ce)
                    token_weights[targets_3d == entropy_model.SKIP_ID] = skip_weight
                    ce = ce * token_weights

                # 应用mask（只统计有效帧）
                mask_3d = valid_mask_batch.unsqueeze(-1).float()
                ce_masked = ce * mask_3d
                ce_loss = ce_masked.sum() / max(mask_3d.sum(), 1.0)

                # KL 牵引（仅当提供 teacher_model 且权重大于 0）
                kl_loss = logits_flat.new_tensor(0.0)
                if teacher_model is not None and getattr(entropy_config, "kl_tether_weight", 0.0) > 0:
                    with torch.no_grad():
                        # teacher 前向（不带标签）
                        t_logits, _ = teacher_model(indices_batch, None)
                    t_logits_flat = t_logits.reshape(-1, teacher_model.V)
                    kl_loss = masked_token_kl(
                        student_logits=logits_flat.float(),
                        teacher_logits=t_logits_flat.float(),
                        valid_mask=valid_mask_batch,
                        temperature=getattr(entropy_config, "kl_tether_temperature", 1.0),
                    ) * entropy_config.kl_tether_weight

            # 在AMP块外计算loss，确保总是被定义
            loss = ce_loss + kl_loss

            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                optim_params,
                entropy_config.max_grad_norm
            )
            optimizer.step()
            
            # 计算准确率（修正：将帧级mask扩展到token级）
            with torch.inference_mode():
                # 在AMP之外重新计算preds（避免精度问题）
                preds = logits_flat.float().argmax(dim=-1)  # (B*T*L,)
                # 将 (B, T) 的帧级mask扩展到 (B, T, L) 的token级，再展平成 (B*T*L,)
                mask_tokens = valid_mask_batch.unsqueeze(-1).expand(-1, -1, L).reshape(-1).float()
                if mask_tokens.sum() > 0:
                    acc = ((preds == targets_flat).float() * mask_tokens).sum() / mask_tokens.sum()
                else:
                    acc = torch.tensor(0.0)
            
            # 累加统计（在continue之后，确保所有变量都已定义）
            total_loss += loss.item()
            total_ce_loss += ce_loss.item()
            total_kl_loss += kl_loss.item()
            total_acc += acc.item()
            num_batches += 1

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{acc.item():.3f}',
                'bpf': f'{ce_loss.item() / math.log(2):.2f}'   # 用CE计算bpf，不含KL
            })
        
        # 检查是否有有效batch（防止所有batch都被跳过导致除零错误）
        if num_batches == 0:
            logger.warning(f"⚠️  Epoch {epoch}: 所有batch都被跳过，可能是数据过滤太严格！")
            continue
        
        avg_loss = total_loss / num_batches
        avg_ce_loss = total_ce_loss / num_batches
        avg_kl_loss = total_kl_loss / num_batches
        avg_acc = total_acc / num_batches
        avg_bpf = avg_ce_loss / math.log(2)  # 用CE计算bpf，不含KL

        # 进度展示
        progress_pct = epoch / num_epochs * 100
        logger.info("")
        logger.info(f"{'='*80}")
        logger.info(f"Epoch {epoch}/{num_epochs} ({progress_pct:.1f}%) - {model_name}")
        logger.info(f"  Total Loss: {avg_loss:.4f} | CE: {avg_ce_loss:.4f} | KL: {avg_kl_loss:.4f}")
        logger.info(f"  Acc: {avg_acc:.3f} | BPF: {avg_bpf:.2f}")
        logger.info(f"{'='*80}")
        
        # 早停机制：基于Loss（越低越好）
        if entropy_config.enable_early_stopping:
            # 检查是否有显著改进
            improvement = best_loss - avg_loss  # Loss下降为正值
            
            # 第一个epoch或有显著改进
            if best_loss == float('inf') or improvement > entropy_config.early_stopping_min_delta * abs(avg_loss):
                # 有显著改进（相对改进>1%），保存最佳模型并重置patience
                best_loss = avg_loss
                best_epoch = epoch
                patience_counter = 0
                
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': entropy_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_loss,
                    'acc': avg_acc,
                    'hparams': {
                        'd_model': entropy_config.d_model,
                        'num_layers': entropy_config.num_layers,
                        'num_heads': entropy_config.num_heads,
                        'd_ff': entropy_config.d_ff,
                        'dropout': entropy_config.dropout,
                        'max_seq_len': entropy_config.max_seq_len,
                        'causal': entropy_config.causal,
                        'skip_weight': entropy_config.skip_weight,
                        'learning_rate': entropy_config.learning_rate,
                        'weight_decay': entropy_config.weight_decay,
                        'warmup_steps': entropy_config.warmup_steps,
                        'max_grad_norm': entropy_config.max_grad_norm,
                    }
                }
                save_path = checkpoint_dir / f"{model_name}_best.pt"
                save_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(checkpoint, save_path)
                logger.info(f"✅ 保存最佳模型: {save_path} (Loss改进: {improvement:.4f}, {improvement/best_loss*100:.2f}%)")
            else:
                # 无显著改进，增加patience计数
                patience_counter += 1
                relative_improvement = improvement / best_loss * 100 if best_loss > 0 else 0
                logger.info(f"⏸️  无显著改进 (改进: {relative_improvement:.3f}%, 阈值: {entropy_config.early_stopping_min_delta*100:.1f}%)")
                logger.info(f"   Early stopping: {patience_counter}/{entropy_config.early_stopping_patience}")
                
                # 检查是否应该早停
                if patience_counter >= entropy_config.early_stopping_patience:
                    logger.info("")
                    logger.info("=" * 80)
                    logger.info(f"🛑 Early Stopping触发！")
                    logger.info(f"   - 最佳epoch: {best_epoch}")
                    logger.info(f"   - 最佳Loss: {best_loss:.4f}")
                    logger.info(f"   - 连续{patience_counter}个epoch无显著改进")
                    logger.info("=" * 80)
                    break  # 提前退出训练循环
        else:
            # 不使用早停，保持原逻辑
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_epoch = epoch
                
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': entropy_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_loss,
                    'acc': avg_acc,
                    'hparams': {
                        'd_model': entropy_config.d_model,
                        'num_layers': entropy_config.num_layers,
                        'num_heads': entropy_config.num_heads,
                        'd_ff': entropy_config.d_ff,
                        'dropout': entropy_config.dropout,
                        'max_seq_len': entropy_config.max_seq_len,
                        'causal': entropy_config.causal,
                        'skip_weight': entropy_config.skip_weight,
                        'learning_rate': entropy_config.learning_rate,
                        'weight_decay': entropy_config.weight_decay,
                        'warmup_steps': entropy_config.warmup_steps,
                        'max_grad_norm': entropy_config.max_grad_norm,
                    }
                }
                save_path = checkpoint_dir / f"{model_name}_best.pt"
                save_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(checkpoint, save_path)
                logger.info(f"✅ 保存最佳模型: {save_path}")
    
    logger.info("=" * 80)
    logger.info(f"✅ {model_name} 训练完成! 最佳Loss: {best_loss:.4f}")
    logger.info("=" * 80)


def main():
    """主函数：训练无条件熵模型 q(z)（发布版）"""
    # 加载配置
    config = get_default_config()
    grouped_rvq_config = config['grouped_rvq']
    entropy_model_config = config['entropy_model']
    data_config = config['data']
    training_config = config['training']
    
    # 设置随机种子（确保可重现）
    set_seed(data_config.seed)
    
    device = torch.device(training_config.device if torch.cuda.is_available() else 'cpu')
    
    # 1. 加载训练好的RVQ
    logger.info("初始化RVQ模型...")
    rvq_model = GroupedResidualVQ(grouped_rvq_config).to(device)
    
    # 加载RVQ checkpoint（使用本地训练的checkpoint）
    rvq_checkpoint_path = Path('checkpoints/grouped_rvq_best.pt')
    
    if rvq_checkpoint_path.exists():
        logger.info(f"加载RVQ检查点: {rvq_checkpoint_path}")
        checkpoint = torch.load(rvq_checkpoint_path, map_location=device)
        rvq_model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"✅ 已加载RVQ checkpoint")
    else:
        raise FileNotFoundError(
            f"RVQ checkpoint不存在: {rvq_checkpoint_path}\n"
            f"请先训练RVQ: python train_rvq.py 或 sbatch run_train_rvq.slurm"
        )
    
    # 2. 加载数据 (q(z)使用emilia数据集)
    # 关键修复：索引提取阶段使用EntropyModelConfig的参数（不是TrainingConfig）
    logger.info("加载数据 (q(z)使用emilia数据集)...")
    
    # 创建临时配置，使用EntropyModelConfig的batch_size和max_frames
    from dataclasses import replace
    
    extract_training_config = replace(
        training_config,
        batch_size=entropy_model_config.batch_size,  # 使用熵模型的batch_size
        num_workers=0  # 降低workers防止额外内存占用
    )
    
    extract_data_config = replace(
        data_config,
        max_frames=entropy_model_config.max_frames_per_sample,  # 使用熵模型的max_frames
        max_samples=int(37722 * entropy_model_config.train_data_fraction) if entropy_model_config.train_data_fraction < 1.0 else None  # 关键：索引提取也应用数据比例
    )
    
    logger.info(f"索引提取配置: batch_size={entropy_model_config.batch_size}, "
                f"max_frames={entropy_model_config.max_frames_per_sample}帧 "
                f"({entropy_model_config.max_frames_per_sample/50:.1f}秒)")
    
    train_loader, val_loader = create_dataloaders(
        extract_data_config, extract_training_config, grouped_rvq_config,
        use_bucketing=True,   # 开启分桶加速训练
        num_buckets=10
    )
    
    # 3. 用RVQ提取训练集的离散索引（包含SKIP，让熵模型学习SKIP的概率）
    logger.info("提取训练集索引（使用简化ECVQ生成带SKIP的样本）...")
    
    # 使用多个λ值生成多样化的SKIP模式（恢复旧代码的简单方式）
    # λ越大，SKIP越多，码率越低
    lambda_grid = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0]
    logger.info(f"使用 {len(lambda_grid)} 个λ值: {lambda_grid}")
    
    # 创建索引缓存目录
    indices_cache_dir = Path("indices_cache")
    indices_cache_dir.mkdir(exist_ok=True)
    
    train_indices, train_labels, train_masks = [], [], []
    
    for idx, lam in enumerate(lambda_grid):
        cache_file = indices_cache_dir / f"lambda_{lam:.2f}.pt"
        
        # 检查缓存
        if cache_file.exists():
            logger.info(f"  λ={lam}: 从缓存加载...")
            cached = torch.load(cache_file)
            idxs = cached['indices']
            labs = cached['labels']
            masks = cached['masks']
            logger.info(f"    ✅ 从缓存加载: {len(idxs)}个样本")
        else:
            logger.info(f"  λ={lam} 抽取索引...")
            idxs, labs, masks = extract_indices_from_rvq(
                rvq_model, train_loader, device, ecvq_lambda=lam
            )
            
            # 立即保存到磁盘（防止崩溃丢失）
            torch.save({
                'indices': idxs,
                'labels': labs,
                'masks': masks
            }, cache_file)
            logger.info(f"    💾 已保存到缓存: {cache_file.name}")
        
        train_indices += idxs
        train_labels += labs
        train_masks += masks
        logger.info(f"    → 累计 {len(train_indices)} 个样本")
    
    logger.info(f"✅ 总共提取 {len(train_indices)} 个样本（包含多种SKIP模式，覆盖宽码率范围）")
    
    # 4. 训练无条件熵模型 q(z)（包含SKIP概率）
    logger.info("\n" + "=" * 80)
    logger.info("训练无条件熵模型 q(z)（发布版）")
    logger.info("  注：训练数据包含SKIP，模型能学习真实的SKIP概率")
    logger.info("=" * 80)
    
    entropy_model_q_z = create_entropy_model(
        entropy_model_config, 
        grouped_rvq_config
    )
    
    train_entropy_model(
        entropy_model_q_z,
        train_indices,
        train_labels,
        train_masks,
        training_config,
        entropy_model_config,
        model_name="entropy_model"  # 保存为entropy_model_best.pt（与评估脚本匹配）
    )

    logger.info("\n" + "=" * 80)
    logger.info("✅ 无条件熵模型 q(z) 训练完成!")
    logger.info("✅ Checkpoint已保存: checkpoints/entropy_model_best.pt")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
