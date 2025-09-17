#!/usr/bin/env python3
"""
Emotion2Vec VQ-VAE 训练启动脚本
基于 VEVO 训练流程，适配 Emotion2Vec 特征
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path

# 添加路径
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))
sys.path.append(str(current_dir.parent / "vevo-code"))

import torch
from torch.utils.data import DataLoader

from emotion2vec_vqvae_trainer import (
    Emotion2VecVQVAETrainer, 
    Emotion2VecDataset, 
    Emotion2VecCollator,
    SimpleConfig,
    create_emotion2vec_config
)


def load_config_from_json(config_path):
    """从 JSON 文件加载配置"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = json.load(f)
    
    def dict_to_config(d):
        if isinstance(d, dict):
            return SimpleConfig(**{k: dict_to_config(v) for k, v in d.items()})
        else:
            return d
    
    return dict_to_config(config_dict)


def setup_logging(log_dir):
    """设置日志"""
    os.makedirs(log_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'train.log')),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)


def create_data_loaders(cfg):
    """创建数据加载器"""
    # 创建数据集
    train_dataset = Emotion2VecDataset(
        data_root=cfg.dataset.data_root,
        cfg=cfg,
        is_valid=False
    )
    
    # 创建验证数据集（使用部分训练数据）
    valid_dataset = Emotion2VecDataset(
        data_root=cfg.dataset.data_root,
        cfg=cfg,
        is_valid=True
    )
    
    # 如果数据集太大，限制验证集大小
    if len(valid_dataset) > 100:
        valid_indices = torch.randperm(len(valid_dataset))[:100]
        valid_dataset = torch.utils.data.Subset(valid_dataset, valid_indices)
    
    # 创建 collator
    collator = Emotion2VecCollator(cfg)
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.dataloader.num_worker,
        pin_memory=cfg.train.dataloader.pin_memory,
        collate_fn=collator,
        drop_last=True
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.train.dataloader.num_worker,
        pin_memory=cfg.train.dataloader.pin_memory,
        collate_fn=collator,
        drop_last=False
    )
    
    return train_loader, valid_loader


def train_model(args, cfg, logger):
    """训练模型"""
    # 创建训练器
    trainer = Emotion2VecVQVAETrainer(args, cfg)
    
    # 创建数据加载器
    train_loader, valid_loader = create_data_loaders(cfg)
    
    logger.info(f"Training dataset size: {len(train_loader.dataset)}")
    logger.info(f"Validation dataset size: {len(valid_loader.dataset)}")
    logger.info(f"Training steps per epoch: {len(train_loader)}")
    
    # 训练循环
    step = 0
    best_loss = float('inf')
    
    # 创建检查点目录
    checkpoint_dir = os.path.join(cfg.log_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    try:
        for epoch in range(cfg.train.epochs):
            logger.info(f"Starting epoch {epoch + 1}/{cfg.train.epochs}")
            
            # 训练阶段
            trainer.model.train() if hasattr(trainer.model, 'train') else None
            epoch_losses = []
            
            for batch_idx, batch in enumerate(train_loader):
                try:
                    # 训练步骤
                    loss, train_losses, train_stats = trainer._train_step(batch)
                    epoch_losses.append(loss)
                    step += 1
                    
                    # 记录日志
                    if step % cfg.train.save_summary_steps == 0:
                        avg_loss = sum(epoch_losses[-100:]) / len(epoch_losses[-100:])
                        logger.info(
                            f"Step {step}: Loss={loss:.4f}, Avg Loss={avg_loss:.4f}, "
                            f"Rec Loss={train_losses['rec_loss']:.4f}, "
                            f"Codebook Loss={train_losses['codebook_loss']:.4f}"
                        )
                        
                        if 'perplexity' in train_stats:
                            logger.info(f"  Perplexity: {train_stats['perplexity']:.2f}")
                    
                    # 验证
                    if step % cfg.train.valid_interval == 0:
                        logger.info("Running validation...")
                        trainer.model.eval() if hasattr(trainer.model, 'eval') else None
                        
                        valid_losses = []
                        with torch.no_grad():
                            for valid_batch in valid_loader:
                                try:
                                    valid_loss, _, _ = trainer._valid_step(valid_batch)
                                    valid_losses.append(valid_loss)
                                except Exception as e:
                                    logger.warning(f"Validation batch failed: {e}")
                                    continue
                        
                        if valid_losses:
                            avg_valid_loss = sum(valid_losses) / len(valid_losses)
                            logger.info(f"Validation Loss: {avg_valid_loss:.4f}")
                            
                            # 保存最佳模型
                            if avg_valid_loss < best_loss:
                                best_loss = avg_valid_loss
                                best_checkpoint_path = os.path.join(
                                    checkpoint_dir, "best_model.pt"
                                )
                                trainer.save_checkpoint(best_checkpoint_path)
                                logger.info(f"New best model saved: {avg_valid_loss:.4f}")
                        
                        trainer.model.train() if hasattr(trainer.model, 'train') else None
                    
                    # 保存检查点
                    if step % cfg.train.save_checkpoints_steps == 0:
                        checkpoint_path = os.path.join(
                            checkpoint_dir, f"checkpoint_step_{step}.pt"
                        )
                        trainer.save_checkpoint(checkpoint_path)
                        logger.info(f"Checkpoint saved at step {step}")
                    
                    # 检查是否达到最大步数
                    if step >= cfg.train.max_steps:
                        logger.info(f"Reached max steps {cfg.train.max_steps}")
                        break
                        
                except Exception as e:
                    logger.error(f"Training step {step} failed: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            if step >= cfg.train.max_steps:
                break
                
            # 记录每个 epoch 的平均损失
            if epoch_losses:
                avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
                logger.info(f"Epoch {epoch + 1} completed. Average loss: {avg_epoch_loss:.4f}")
    
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 保存最终模型
        final_checkpoint_path = os.path.join(checkpoint_dir, "final_model.pt")
        trainer.save_checkpoint(final_checkpoint_path)
        logger.info(f"Final model saved to {final_checkpoint_path}")


def main():
    parser = argparse.ArgumentParser(description="Train Emotion2Vec VQ-VAE")
    parser.add_argument(
        "--config", 
        type=str, 
        default="config/emotion2vec_vqvae_config.json",
        help="Path to config file"
    )
    parser.add_argument(
        "--data_root", 
        type=str, 
        default="./data",
        help="Path to training data"
    )
    parser.add_argument(
        "--exp_dir", 
        type=str, 
        default="./experiments/emotion2vec_vqvae",
        help="Experiment directory"
    )
    parser.add_argument(
        "--resume", 
        type=str, 
        default=None,
        help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--use_dummy", 
        action="store_true",
        help="Use dummy emotion2vec features for testing"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=None,
        help="Override batch size"
    )
    parser.add_argument(
        "--learning_rate", 
        type=float, 
        default=None,
        help="Override learning rate"
    )
    parser.add_argument(
        "--max_steps", 
        type=int, 
        default=None,
        help="Override max training steps"
    )
    
    args = parser.parse_args()
    
    # 创建实验目录
    os.makedirs(args.exp_dir, exist_ok=True)
    
    # 设置日志
    logger = setup_logging(args.exp_dir)
    logger.info("Starting Emotion2Vec VQ-VAE training")
    logger.info(f"Arguments: {args}")
    
    # 加载配置
    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = os.path.join(Path(__file__).parent, config_path)
    
    if os.path.exists(config_path):
        logger.info(f"Loading config from {config_path}")
        cfg = load_config_from_json(config_path)
    else:
        logger.info("Using default config")
        cfg = create_emotion2vec_config()
    
    # 覆盖配置参数
    cfg.dataset.data_root = args.data_root
    cfg.log_dir = args.exp_dir
    
    if args.use_dummy:
        cfg.model.emotion2vec.use_dummy = True
        logger.info("Using dummy emotion2vec features")
    
    if args.batch_size:
        cfg.train.batch_size = args.batch_size
        logger.info(f"Override batch size: {args.batch_size}")
    
    if args.learning_rate:
        cfg.train.adam.lr = args.learning_rate
        cfg.train.learning_rate = args.learning_rate
        logger.info(f"Override learning rate: {args.learning_rate}")
    
    if args.max_steps:
        cfg.train.max_steps = args.max_steps
        cfg.train.num_train_steps = args.max_steps
        cfg.train.total_training_steps = args.max_steps
        logger.info(f"Override max steps: {args.max_steps}")
    
    # 保存配置
    config_save_path = os.path.join(args.exp_dir, "config.json")
    with open(config_save_path, 'w', encoding='utf-8') as f:
        json.dump(cfg.__dict__, f, indent=2, default=str)
    logger.info(f"Config saved to {config_save_path}")
    
    # 设置随机种子
    if hasattr(cfg.train, 'random_seed'):
        torch.manual_seed(cfg.train.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(cfg.train.random_seed)
    
    # 添加实验名称
    args.exp_name = f"emotion2vec_vqvae_{cfg.model.repcodec.codebook_size}"
    
    # 开始训练
    train_model(args, cfg, logger)
    
    logger.info("Training completed!")


if __name__ == "__main__":
    main()
