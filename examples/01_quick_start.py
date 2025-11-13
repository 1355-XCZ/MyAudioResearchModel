"""
快速开始示例：训练RVQ和熵模型

这个脚本展示如何使用本项目的核心功能。
"""

import torch
from config import GroupedRVQConfig, EntropyModelConfig
from grouped_rvq import GroupedRVQ
from entropy_model import EntropyModel

def main():
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. 创建分组RVQ模型
    print("\n1. Creating Grouped RVQ model...")
    rvq_config = GroupedRVQConfig()
    rvq_model = GroupedRVQ(rvq_config).to(device)
    print(f"   - Groups: {rvq_config.num_groups}")
    print(f"   - Layers per group: {rvq_config.num_layers_per_group}")
    print(f"   - Codebook size: {rvq_config.codebook_size}")
    
    # 2. 创建熵模型
    print("\n2. Creating Entropy Model...")
    entropy_config = EntropyModelConfig()
    entropy_model = EntropyModel(entropy_config).to(device)
    print(f"   - Model dimension: {entropy_config.d_model}")
    print(f"   - Transformer layers: {entropy_config.num_layers}")
    
    # 3. 示例：量化emotion2vec特征
    print("\n3. Example: Quantizing emotion2vec features...")
    batch_size = 2
    num_frames = 100
    feature_dim = 768
    
    # 模拟emotion2vec特征
    features = torch.randn(batch_size, num_frames, feature_dim).to(device)
    print(f"   Input shape: {features.shape}")
    
    # 量化
    with torch.no_grad():
        quantized, indices, commitment_loss = rvq_model.forward_without_entropy(
            features, 
            lambda_val=1.0
        )
    
    print(f"   Quantized shape: {quantized.shape}")
    print(f"   Indices shape: {indices.shape}")
    print(f"   Commitment loss: {commitment_loss.item():.4f}")
    
    # 4. 示例：使用熵模型编码
    print("\n4. Example: Entropy coding...")
    with torch.no_grad():
        # 创建有效掩码（所有帧都有效）
        valid_mask = torch.ones(batch_size, num_frames, dtype=torch.bool).to(device)
        
        # 计算比特数
        total_bits = entropy_model.compute_bits(indices, valid_mask)
        rate_bpf = total_bits / num_frames
        
    print(f"   Total bits: {total_bits:.2f}")
    print(f"   Rate (bits per frame): {rate_bpf:.2f}")
    
    print("\n✅ Quick start completed successfully!")
    print("\nNext steps:")
    print("  - See examples/02_train_rvq.py for RVQ training")
    print("  - See examples/03_train_entropy.py for entropy model training")
    print("  - See examples/04_evaluation.py for evaluation")

if __name__ == "__main__":
    main()

