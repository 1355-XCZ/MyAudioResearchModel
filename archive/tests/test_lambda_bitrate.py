"""
测试λ值与实际码率的对应关系
无需等待熵模型训练完成即可运行
"""

import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

from config import get_default_config
from grouped_rvq import GroupedResidualVQ
from data_loader import create_dataloaders


def test_lambda_bitrate_mapping(test_lambdas=None):
    """
    测试λ值对应的实际码率
    
    Args:
        test_lambdas: 要测试的λ值列表，None则使用config中的lambda_grid
    """
    config = get_default_config()
    grouped_rvq_config = config['grouped_rvq']
    entropy_model_config = config['entropy_model']
    data_config = config['data']
    training_config = config['training']
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 加载RVQ
    print("加载RVQ模型...")
    rvq_model = GroupedResidualVQ(grouped_rvq_config).to(device)
    rvq_checkpoint = torch.load('checkpoints/grouped_rvq_best.pt', map_location=device)
    rvq_model.load_state_dict(rvq_checkpoint['model_state_dict'])
    rvq_model.eval()
    print("✅ RVQ加载成功")
    
    # 加载少量测试数据（不需要全部）
    print("加载测试数据...")
    from dataclasses import replace
    test_data_config = replace(data_config, max_samples=100)  # 只用100个样本
    test_training_config = replace(training_config, batch_size=16)
    
    _, test_loader = create_dataloaders(
        test_data_config, test_training_config, grouped_rvq_config,
        use_bucketing=False
    )
    print(f"✅ 加载了测试数据")
    
    # 测试λ值
    if test_lambdas is None:
        test_lambdas = entropy_model_config.lambda_grid
    
    print(f"\n{'='*70}")
    print(f"测试 {len(test_lambdas)} 个λ值")
    print(f"{'='*70}")
    
    results = []
    
    for lam in tqdm(test_lambdas, desc="测试λ值"):
        total_bits = 0
        total_frames = 0
        skip_counts = 0
        total_codes = 0
        
        with torch.no_grad():
            for batch in test_loader:
                features = batch['features'].to(device)
                lengths = batch['lengths'].to(device)
                B, T, D = features.shape
                
                # 用简化ECVQ编码（不需要熵模型）
                _, indices, _, stats = rvq_model(
                    features,
                    lambda_rate=torch.tensor(lam, device=device),
                    entropy_model=None,  # 简化ECVQ，不依赖熵模型
                    valid_mask=(torch.arange(T, device=device).unsqueeze(0) < lengths.unsqueeze(1))
                )
                
                # 粗略估算码率（假设均匀分布）
                # 实际需要熵模型，这里只是估算
                num_codes = (indices >= 0).sum().item()  # 非SKIP的码字数
                num_tokens = indices.numel()
                skip_rate = 1 - num_codes / num_tokens
                
                # 理论bits（假设均匀分布）
                bits_per_code = np.log2(128)  # 码本大小
                total_bits += num_codes * bits_per_code
                total_frames += lengths.sum().item()
                
                skip_counts += (indices < 0).sum().item()
                total_codes += indices.numel()
        
        # 计算平均码率
        avg_bitrate_bpf = total_bits / total_frames if total_frames > 0 else 0
        skip_rate_pct = skip_counts / total_codes * 100 if total_codes > 0 else 0
        
        results.append({
            'lambda': lam,
            'bitrate_bpf': avg_bitrate_bpf,
            'skip_rate': skip_rate_pct
        })
        
        print(f"λ={lam:8.2f} → 码率≈{avg_bitrate_bpf:6.1f} bpf, SKIP率≈{skip_rate_pct:5.1f}%")
    
    print(f"\n{'='*70}")
    print("测试完成！")
    print(f"{'='*70}")
    
    # 保存结果
    import json
    with open('lambda_bitrate_test.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✅ 结果已保存: lambda_bitrate_test.json")
    
    return results


if __name__ == "__main__":
    import sys
    
    # 可以指定要测试的λ值
    if len(sys.argv) > 1:
        test_lambdas = [float(x) for x in sys.argv[1:]]
        print(f"测试指定的λ值: {test_lambdas}")
    else:
        test_lambdas = None  # 使用config中的全部λ值
        print("测试config中的所有λ值")
    
    results = test_lambda_bitrate_mapping(test_lambdas)

