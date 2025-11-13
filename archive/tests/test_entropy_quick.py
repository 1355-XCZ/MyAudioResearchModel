"""
快速测试熵模型训练流程（0.01数据，1个目标bpf）
确保完整流程能跑通，然后可以安心睡觉
"""

import torch
import os
from pathlib import Path

# 设置测试模式环境变量
os.environ['TEST_MODE'] = 'true'
os.environ['TEST_DATA_FRACTION'] = '0.01'  # 只用1%数据
os.environ['TEST_TARGET_BPF'] = '50'  # 只测试1个关键点

from config import get_default_config
from grouped_rvq import GroupedResidualVQ
from entropy_model import create_entropy_model
from data_loader import create_dataloaders
from dataclasses import replace
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def quick_test():
    """快速测试"""
    logger.info("="*60)
    logger.info("快速测试：0.01数据 + 1个目标bpf")
    logger.info("="*60)
    
    # 加载配置
    config = get_default_config()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 修改为测试配置
    test_data_config = replace(config['data'], 
                               train_data_fraction=0.01)  # 只用1%数据
    
    test_entropy_config = replace(config['entropy_model'],
                                  num_epochs=1,  # 只训练1个epoch
                                  target_bpf_grid=[50.0])  # 只测试1个关键点
    
    logger.info(f"测试配置: 数据量1%, 目标bpf=[50], epochs=1")
    
    # 1. 加载RVQ
    logger.info("加载RVQ checkpoint...")
    rvq_model = GroupedResidualVQ(config['grouped_rvq']).to(device)
    ckpt = torch.load('checkpoints/grouped_rvq_best.pt', map_location=device)
    rvq_model.load_state_dict(ckpt['model_state_dict'])
    logger.info("✅ RVQ已加载")
    
    # 2. 加载数据
    logger.info("加载测试数据...")
    extract_training_config = replace(
        config['training'],
        batch_size=test_entropy_config.batch_size,
        num_workers=0
    )
    extract_data_config = replace(
        test_data_config,
        max_frames=test_entropy_config.max_frames_per_sample
    )
    
    train_loader, _ = create_dataloaders(
        extract_data_config, extract_training_config, config['grouped_rvq'],
        use_bucketing=True, num_buckets=10
    )
    logger.info(f"✅ 数据已加载: {len(train_loader.dataset)}个样本")
    
    # 3. 提取索引（1个target_bpf）
    logger.info("提取索引（target_bpf=50）...")
    rvq_model.eval()
    all_indices = []
    
    with torch.no_grad():
        for i, batch in enumerate(train_loader):
            if i >= 10:  # 只测试10个batch
                break
            features = batch['features'].to(device)
            lengths = batch['lengths'].to(device)
            
            valid_bt = (torch.arange(features.size(1), device=device).unsqueeze(0) < lengths.unsqueeze(1))
            
            _, indices, _, _ = rvq_model(
                features, target_bpf=50.0, valid_mask=valid_bt
            )
            
            for j in range(features.size(0)):
                T_j = int(lengths[j].item())
                idx_j = indices[j, :T_j, :].to(torch.uint8).cpu()
                all_indices.append(idx_j)
    
    logger.info(f"✅ 提取完成: {len(all_indices)}个样本")
    
    # 4. 创建熵模型并训练1步
    logger.info("创建熵模型...")
    entropy_model = create_entropy_model(test_entropy_config, config['grouped_rvq']).to(device)
    
    optimizer = torch.optim.AdamW(
        [p for p in entropy_model.parameters() if p.requires_grad],
        lr=test_entropy_config.learning_rate
    )
    
    # 简单训练1个batch
    logger.info("测试训练1个batch...")
    entropy_model.train()
    
    # 准备1个batch数据
    batch_indices = [all_indices[i].to(torch.long) for i in range(min(8, len(all_indices)))]
    max_T = max(idx.shape[0] for idx in batch_indices)
    L = batch_indices[0].shape[1]
    B = len(batch_indices)
    
    indices_batch = torch.full((B, max_T, L), entropy_model.SKIP_ID, dtype=torch.long, device=device)
    for i, idx in enumerate(batch_indices):
        T_i = idx.shape[0]
        indices_batch[i, :T_i, :] = idx
    
    # 前向+反向
    optimizer.zero_grad()
    logits, targets = entropy_model(indices_batch, None)
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, entropy_model.V),
        targets.reshape(-1),
        reduction='mean'
    )
    loss.backward()
    optimizer.step()
    
    logger.info(f"✅ 训练测试成功: Loss={loss.item():.4f}")
    
    logger.info("")
    logger.info("="*60)
    logger.info("✅ 快速测试完全成功！")
    logger.info("完整流程可以跑通，可以安心睡觉了 😴")
    logger.info("="*60)

if __name__ == "__main__":
    quick_test()


