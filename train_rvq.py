"""
训练分组RVQ（阶段1）
目标：训练稳定的码本
"""

import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
import logging
import json
import sys
import random
import numpy as np

try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except:
    HAS_TENSORBOARD = False

from config import get_default_config
from grouped_rvq import GroupedResidualVQ
from data_loader import create_dataloaders

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def set_seed(seed: int, deterministic: bool = False):
    """设置随机种子保证可复现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        logger.info(f"✅ 已设置随机种子: {seed} (deterministic mode)")
    else:
        torch.backends.cudnn.benchmark = True
        logger.info(f"✅ 已设置随机种子: {seed}")


class WarmupCosineScheduler:
    """Warmup + Cosine Annealing学习率调度器"""
    
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.base_lr = optimizer.param_groups[0]['lr']
        self.current_step = 0
    
    def step(self):
        """更新学习率（带除零防护）"""
        self.current_step += 1
        
        if self.total_steps <= self.warmup_steps:
            # 小数据/少步数：简单warmup
            lr = self.base_lr * min(1.0, self.current_step / max(1, self.warmup_steps))
        else:
            if self.current_step < self.warmup_steps:
                # Linear warmup
                lr = self.base_lr * self.current_step / self.warmup_steps
            else:
                # Cosine annealing（带防护）
                denom = max(1, self.total_steps - self.warmup_steps)
                progress = (self.current_step - self.warmup_steps) / denom
                progress = float(np.clip(progress, 0.0, 1.0))
                lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + np.cos(np.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr
    
    def get_last_lr(self):
        """获取当前学习率"""
        return [param_group['lr'] for param_group in self.optimizer.param_groups]


def _accumulate_codebook_usage(usage_acc, indices, valid_bt, skip_id, K):
    """
    累积码本使用统计（按批，避免维度不齐）
    
    Args:
        usage_acc: 累加器（List[dict] 或 None）
        indices: (B, T, L) 码本索引
        valid_bt: (B, T) bool mask
        skip_id: SKIP token的id
        K: 码本大小
    
    Returns:
        updated usage_acc
    """
    B, T, L = indices.shape
    
    # 初始化累加器（包含histogram用于困惑度）
    if usage_acc is None:
        usage_acc = [{'used': set(), 'count': 0, 'skips': 0, 'histogram': {}} for _ in range(L)]
    
    # 移到CPU处理（避免GPU内存占用）
    idx_cpu = indices.cpu()
    mask_cpu = valid_bt.cpu()
    
    # 逐层统计
    for l in range(L):
        layer_idx = idx_cpu[:, :, l]  # (B, T)
        layer_idx = layer_idx[mask_cpu]  # (N_valid,) 只取有效帧
        
        if layer_idx.numel() == 0:
            continue
        
        # 统计SKIP
        skips = (layer_idx == skip_id).sum().item()
        
        # 统计使用的码字（排除SKIP）
        non_skip = layer_idx[layer_idx != skip_id]
        if non_skip.numel() > 0:
            usage_acc[l]['used'].update(non_skip.tolist())
            
            # 累积直方图（用于计算困惑度）
            for idx in non_skip.tolist():
                usage_acc[l]['histogram'][idx] = usage_acc[l]['histogram'].get(idx, 0) + 1
        
        # 累加计数
        usage_acc[l]['count'] += layer_idx.numel()
        usage_acc[l]['skips'] += skips
    
    return usage_acc


def train_model(rvq_config, data_config, training_config):
    """训练分组RVQ"""
    # 设置随机种子保证可复现性
    set_seed(data_config.seed, deterministic=False)
    
    device = torch.device(training_config.device if torch.cuda.is_available() else 'cpu')
    
    # 创建目录
    for d in [training_config.checkpoint_dir, training_config.log_dir, training_config.results_dir]:
        Path(d).mkdir(parents=True, exist_ok=True)
    
    # 加载数据
    logger.info("加载数据...")
    train_loader, val_loader = create_dataloaders(
        data_config, training_config, rvq_config,
        use_bucketing=training_config.use_bucketing,
        num_buckets=training_config.num_buckets
    )
    
    # 创建模型
    logger.info("创建模型...")
    model = GroupedResidualVQ(rvq_config).to(device)
    
    # 初始化码本（kmeans_init需要先前向传播一次）
    logger.info("初始化码本...")
    with torch.no_grad():
        # 创建dummy batch用于初始化
        dummy_batch = next(iter(train_loader))
        dummy_features = dummy_batch['features'][:min(8, len(dummy_batch['features']))].to(device)
        dummy_lengths = dummy_batch['lengths'][:min(8, len(dummy_batch['lengths']))].to(device)
        
        # 创建mask
        B, T, D = dummy_features.shape
        t_idx = torch.arange(T, device=device).unsqueeze(0)
        valid_bt = (t_idx < dummy_lengths.unsqueeze(1))
        
        # 前向传播初始化码本
        _ = model(dummy_features, valid_mask=valid_bt)
    
    # 检查可训练参数
    learnable_params = [p for p in model.parameters() if p.requires_grad]
    n_params_total = sum(p.numel() for p in model.parameters())
    n_params_learnable = sum(p.numel() for p in learnable_params)
    
    logger.info(f"✅ 码本初始化完成")
    logger.info(f"   - 总参数: {n_params_total}")
    logger.info(f"   - 可训练参数: {n_params_learnable}")
    
    # 根据是否有可训练参数决定是否创建优化器
    if n_params_learnable == 0:
        # EMA模式：码本在forward中自动更新，不需要优化器
        optimizer = None
        scheduler = None
        use_step_scheduler = False
        logger.info("⚙️  EMA模式：码本通过EMA自动更新，不使用优化器")
    else:
        # 可学习模式：需要优化器
        optimizer = torch.optim.AdamW(
            learnable_params,
            lr=training_config.learning_rate,
            weight_decay=training_config.weight_decay,
            betas=training_config.betas
        )
        logger.info(f"⚙️  可学习模式：使用优化器更新 {n_params_learnable} 个参数")
    
        # 学习率调度器（带warmup）
        total_steps = len(train_loader) * training_config.num_epochs
        if training_config.scheduler == "cosine":
            scheduler = WarmupCosineScheduler(
                optimizer, 
                warmup_steps=training_config.warmup_steps,
                total_steps=total_steps,
                min_lr=1e-6
            )
            use_step_scheduler = True  # 每步更新
        elif training_config.scheduler == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=20, gamma=0.5
            )
            use_step_scheduler = False  # 每epoch更新
        else:
            scheduler = None
            use_step_scheduler = False
    
    # TensorBoard
    writer = None
    if training_config.use_tensorboard and HAS_TENSORBOARD:
        writer = SummaryWriter(log_dir=training_config.log_dir)
    
    # 训练
    logger.info("=" * 80)
    logger.info(f"开始训练分组RVQ:")
    logger.info(f"  - 特征维度: {rvq_config.feature_dim}")
    logger.info(f"  - 分组: {rvq_config.num_groups}组 × {model.group_dim}维")
    logger.info(f"  - 每组层数: {rvq_config.num_fine_layers}")
    logger.info(f"  - 码本大小: {rvq_config.fine_codebook_size}")
    logger.info(f"  - SKIP: {'启用' if rvq_config.enable_skip else '禁用'}")
    logger.info(f"  - 训练集: {len(train_loader.dataset)} | 验证集: {len(val_loader.dataset)}")
    logger.info("=" * 80)
    
    best_val_loss = float('inf')
    best_val_cosine = 0.0  # 早停基于还原率（越高越好）
    global_step = 0
    
    # 早停机制
    patience_counter = 0
    best_epoch = 0
    
    for epoch in range(1, training_config.num_epochs + 1):
        # 若使用分桶采样器，更新epoch保证每轮洗牌不同
        if hasattr(train_loader, 'batch_sampler') and hasattr(train_loader.batch_sampler, 'set_epoch'):
            train_loader.batch_sampler.set_epoch(epoch)
        
        # ===== 训练 =====
        model.train()
        total_recon = 0
        total_commit = 0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{training_config.num_epochs}")
        for batch in pbar:
            features = batch['features'].to(device, non_blocking=True)  # (B, T, 768)
            labels = batch['labels'].to(device, non_blocking=True)       # (B,)
            lengths = batch['lengths'].to(device, non_blocking=True)     # (B,)
            
            # 创建有效帧掩码（避免padding污染）
            B, T, D = features.shape
            t_idx = torch.arange(T, device=device).unsqueeze(0)  # (1, T)
            valid_bt = (t_idx < lengths.unsqueeze(1))  # (B, T) bool，用于VQ mask
            mask = valid_bt.float().unsqueeze(-1)  # (B, T, 1) float，用于损失计算
            
            # 前向传播（阶段1不用ECVQ，但传入valid_mask屏蔽padding）
            reconstructed, indices, commit_loss, stats = model(
                features,
                lambda_rate=None,  # 阶段1：不用ECVQ
                entropy_model=None,
                valid_mask=valid_bt  # 屏蔽padding帧，避免污染码本
            )
            
            # 第一个batch打印形状验证（修复：移到前向之后）
            if global_step == 0:
                logger.info(f"✅ 输入形状验证: {features.shape}")
                logger.info(f"✅ 序列长度范围: {lengths.min().item()}-{lengths.max().item()}")
                logger.info(f"✅ 有效帧比例: {valid_bt.sum().item()}/{B*T} = {valid_bt.float().mean():.2%}")
                logger.info(f"✅ Commit loss类型检查: {type(commit_loss)}, 值: {float(commit_loss.mean()):.4f}")
                logger.info(f"   （确认这是要加到总损失的'码本/承诺'项）")
            
            # Masked MSE（避免padding污染）
            mse_num = ((reconstructed - features) ** 2 * mask).sum()
            mse_den = (mask.sum() * D).clamp_min(1.0)
            recon_loss = mse_num / mse_den
            
            # Commit loss（按有效帧比例缩放）- 修复：不要除以D
            valid_frames = mask.squeeze(-1).sum()  # 有效帧个数
            valid_ratio = (valid_frames / (B * T)).detach()  # ∈ (0,1]
            commit_loss_scalar = commit_loss.mean() * valid_ratio
            total_loss = recon_loss + commit_loss_scalar
            
            # 反向传播（EMA模式下跳过，码本自动更新）
            if optimizer is not None:
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), training_config.grad_clip)
                optimizer.step()
                
                # 学习率调度（每步）
                if scheduler is not None and use_step_scheduler:
                    current_lr = scheduler.step()
                else:
                    current_lr = optimizer.param_groups[0]['lr']
            else:
                # EMA模式：码本在forward中已自动更新，loss仅用于监控
                current_lr = training_config.learning_rate
            
            total_recon += recon_loss.item()
            total_commit += commit_loss_scalar.item()
            num_batches += 1
            
            pbar.set_postfix({
                'recon': f'{recon_loss.item():.4f}',
                'commit': f'{commit_loss_scalar.item():.4f}',
                'lr': f'{current_lr:.2e}'
            })
            
            # TensorBoard记录
            if writer and global_step % training_config.log_interval == 0:
                writer.add_scalar('train/recon', recon_loss.item(), global_step)
                writer.add_scalar('train/commit', commit_loss_scalar.item(), global_step)
                writer.add_scalar('train/learning_rate', current_lr, global_step)
                
                # 健康监控：残差能量曲线（应随层递减）
                if 'residual_energies' in stats:
                    for layer_idx, energy in enumerate(stats['residual_energies']):
                        writer.add_scalar(f'health/residual_energy_layer_{layer_idx}', energy, global_step)
            
            global_step += 1
        
        avg_recon = total_recon / num_batches
        avg_commit = total_commit / num_batches
        
        logger.info(f"Epoch {epoch} 训练 - Recon: {avg_recon:.4f}, Commit: {avg_commit:.4f}")
        
        # ===== 验证 =====
        if epoch % training_config.eval_interval == 0:
            model.eval()
            total_val_recon = 0
            total_val_cosine_sum = 0  # 改为加权和
            total_valid_frames = 0    # 有效帧总数
            num_val_batches = 0
            
            # 码本利用率累加器（按批累计，避免维度不齐）
            usage_acc = None
            
            with torch.no_grad():
                for batch in val_loader:
                    features = batch['features'].to(device, non_blocking=True)
                    lengths = batch['lengths'].to(device, non_blocking=True)
                    
                    # 创建有效帧掩码
                    B, T, D = features.shape
                    t_idx = torch.arange(T, device=device).unsqueeze(0)
                    valid_bt = (t_idx < lengths.unsqueeze(1))  # (B, T) bool
                    mask = valid_bt.float().unsqueeze(-1)  # (B, T, 1)
                    
                    reconstructed, indices, _, stats = model(features, valid_mask=valid_bt)
                    
                    # 累积码本使用统计（只统计有效帧）
                    usage_acc = _accumulate_codebook_usage(
                        usage_acc, indices, valid_bt, 
                        model.skip_token_id, rvq_config.fine_codebook_size
                    )
                    
                    # Masked MSE
                    mse_num = ((reconstructed - features) ** 2 * mask).sum()
                    mse_den = (mask.sum() * D).clamp_min(1.0)
                    val_recon = (mse_num / mse_den).item()
                    total_val_recon += val_recon
                    
                    # Masked Cosine Similarity（按有效帧加权）
                    mask_flat = mask.reshape(-1)  # (B*T, 1)
                    features_flat = features.reshape(-1, D)
                    reconstructed_flat = reconstructed.reshape(-1, D)
                    
                    # 过滤掉padding帧
                    valid_idx = mask_flat.squeeze() > 0
                    if valid_idx.any():
                        features_valid = features_flat[valid_idx]
                        reconstructed_valid = reconstructed_flat[valid_idx]
                        # 累积：按有效帧数加权
                        cos_sim_sum = F.cosine_similarity(features_valid, reconstructed_valid, dim=1).sum().item()
                        total_val_cosine_sum += cos_sim_sum
                        total_valid_frames += valid_idx.sum().item()
                    
                    num_val_batches += 1
            
            val_recon = total_val_recon / num_val_batches
            val_cosine = total_val_cosine_sum / max(total_valid_frames, 1)  # 按有效帧数加权
            val_rmse = (val_recon ** 0.5)
            
            # 还原率估计
            restoration_rate = val_cosine * 100
            
            logger.info(f"Epoch {epoch} 验证:")
            logger.info(f"  - Recon: {val_recon:.4f}, RMSE: {val_rmse:.4f}")
            logger.info(f"  - CosSim: {val_cosine:.4f} → 还原率: {restoration_rate:.2f}%")
            
            if writer:
                writer.add_scalar('val/recon', val_recon, epoch)
                writer.add_scalar('val/rmse', val_rmse, epoch)
                writer.add_scalar('val/cosine_similarity', val_cosine, epoch)
                writer.add_scalar('val/restoration_rate', restoration_rate, epoch)
            
            # 健康监控：码本利用率（整个验证集，按批累计）
            logger.info("分析码本利用率（整个验证集）...")
            
            # 从累加器生成统计数据（包含困惑度）
            import math
            usage_stats = {}
            perplexities = []
            
            for l, acc in enumerate(usage_acc):
                total = acc['count']
                unique = len(acc['used'])
                util = (unique / rvq_config.fine_codebook_size) if total > 0 else 0.0
                skip_rate = (acc['skips'] / total) if total > 0 else 0.0
                
                # 计算困惑度（perplexity = 2^entropy）
                if len(acc['histogram']) > 0 and total > 0:
                    # 计算概率分布
                    probs = [count / total for count in acc['histogram'].values()]
                    # 计算熵（bits）
                    entropy = -sum(p * math.log2(p) for p in probs if p > 0)
                    # 困惑度
                    perplexity = 2 ** entropy
                else:
                    perplexity = 0.0
                
                perplexities.append(perplexity)
                
                usage_stats[f'layer_{l}'] = {
                    'utilization': util,
                    'unique_codes': unique,
                    'total_codes': rvq_config.fine_codebook_size,
                    'skip_rate': skip_rate,
                    'perplexity': perplexity,
                }
            
            avg_utilization = sum(s['utilization'] for s in usage_stats.values()) / max(1, len(usage_stats))
            avg_perplexity = sum(perplexities) / max(1, len(perplexities))
            logger.info(f"  - 平均利用率: {avg_utilization:.2%}")
            logger.info(f"  - 平均困惑度: {avg_perplexity:.1f} / {rvq_config.fine_codebook_size} (理想值≈码本大小)")
            
            # 打印每层详情（所有层，包含困惑度）
            num_layers = len(usage_stats)
            logger.info(f"  - 每层详情（全部{num_layers}层）:")
            for layer_idx in range(num_layers):
                s = usage_stats[f'layer_{layer_idx}']
                logger.info(f"    层{layer_idx:3d}: 利用率={s['utilization']:5.1%} ({s['unique_codes']:3d}/{s['total_codes']:3d}码), "
                          f"困惑度={s['perplexity']:6.1f}, SKIP={s['skip_rate']:5.1%}")
            
            if writer:
                writer.add_scalar('health/avg_codebook_utilization', avg_utilization, epoch)
                writer.add_scalar('health/avg_perplexity', avg_perplexity, epoch)
                for layer_key, layer_stats in usage_stats.items():
                    layer_idx = int(layer_key.split('_')[1])
                    writer.add_scalar(f'health/codebook_util_layer_{layer_idx}', layer_stats['utilization'], epoch)
                    writer.add_scalar(f'health/perplexity_layer_{layer_idx}', layer_stats['perplexity'], epoch)
                    writer.add_scalar(f'health/skip_rate_layer_{layer_idx}', layer_stats['skip_rate'], epoch)
            
            # 早停机制：基于还原率（余弦相似度）
            if rvq_config.enable_early_stopping:
                # 检查是否有显著提升
                improvement = val_cosine - best_val_cosine
                
                if improvement > rvq_config.early_stopping_min_delta:
                    # 有提升，保存最佳模型并重置patience
                    best_val_cosine = val_cosine
                    best_val_loss = val_recon
                    best_epoch = epoch
                    patience_counter = 0
                    
                    checkpoint = {
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict() if optimizer is not None else None,
                        'val_loss': val_recon,
                        'val_cosine': val_cosine,
                        'rvq_config': {k: v for k, v in rvq_config.__dict__.items() if not k.startswith('_')},
                    }
                    save_path = Path(training_config.checkpoint_dir) / "grouped_rvq_best.pt"
                    save_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(checkpoint, save_path)
                    logger.info(f"✅ 保存最佳模型: {save_path} (还原率提升: +{improvement*100:.3f}%)")
                else:
                    # 无提升，增加patience计数
                    patience_counter += 1
                    logger.info(f"⏸️  无显著提升 (提升: {improvement*100:.3f}%, 阈值: {rvq_config.early_stopping_min_delta*100:.3f}%)")
                    logger.info(f"   Early stopping: {patience_counter}/{rvq_config.early_stopping_patience}")
                    
                    # 检查是否应该早停
                    if patience_counter >= rvq_config.early_stopping_patience:
                        logger.info("")
                        logger.info("=" * 80)
                        logger.info(f"🛑 Early Stopping触发！")
                        logger.info(f"   - 最佳epoch: {best_epoch}")
                        logger.info(f"   - 最佳还原率: {best_val_cosine*100:.2f}%")
                        logger.info(f"   - 连续{patience_counter}个epoch无显著提升")
                        logger.info("=" * 80)
                        break  # 提前退出训练循环
            else:
                # 不使用早停，保持原逻辑（基于val_loss）
            if val_recon < best_val_loss:
                best_val_loss = val_recon
                    best_val_cosine = val_cosine
                    best_epoch = epoch
                    
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict() if optimizer is not None else None,
                    'val_loss': val_recon,
                    'val_cosine': val_cosine,
                    'rvq_config': {k: v for k, v in rvq_config.__dict__.items() if not k.startswith('_')},
                }
                save_path = Path(training_config.checkpoint_dir) / "grouped_rvq_best.pt"
                    save_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(checkpoint, save_path)
                logger.info(f"✅ 保存最佳模型: {save_path}")
        
        # 学习率调度（每epoch，仅用于非step级别的scheduler）
        if scheduler is not None and not use_step_scheduler:
            scheduler.step()
        
        # 定期保存检查点
        if epoch % training_config.save_interval == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict() if optimizer is not None else None,
                'val_loss': val_recon if epoch % training_config.eval_interval == 0 else None,
                'rvq_config': {k: v for k, v in rvq_config.__dict__.items() if not k.startswith('_')},
            }
            save_path = Path(training_config.checkpoint_dir) / f"grouped_rvq_epoch{epoch}.pt"
            torch.save(checkpoint, save_path)
            logger.info(f"保存检查点: {save_path}")
    
    logger.info("=" * 80)
    logger.info(f"✅ 训练完成! 最佳验证损失: {best_val_loss:.4f}")
    logger.info("=" * 80)
    
    if writer:
        writer.close()
    
    # 保存训练总结
    summary = {
        'best_val_loss': best_val_loss,
        'total_epochs': training_config.num_epochs,
        'rvq_config': {k: v for k, v in rvq_config.__dict__.items() if not k.startswith('_')},
    }
    summary_path = Path(training_config.results_dir) / "training_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"训练总结已保存: {summary_path}")


def main():
    """主函数"""
    import os
    
    # 加载配置
    config = get_default_config()
    grouped_rvq_config = config['grouped_rvq']
    data_config = config['data']
    training_config = config['training']
    
    # 测试模式：快速验证流程
    if os.environ.get('TEST_MODE', 'false').lower() == 'true':
        test_epochs = int(os.environ.get('TEST_EPOCHS', '3'))
        logger.info("=" * 80)
        logger.info(f"⚠️  测试模式：只运行 {test_epochs} epochs 验证流程")
        logger.info("=" * 80)
        
        # 覆盖配置
        from dataclasses import replace
        training_config = replace(training_config, num_epochs=test_epochs)
        data_config = replace(data_config, max_samples=1000)  # 只用1000个样本
        logger.info(f"  - 训练epochs: {test_epochs}")
        logger.info(f"  - 最大样本数: 1000")
    
    # 训练
    train_model(grouped_rvq_config, data_config, training_config)


if __name__ == "__main__":
    main()

