"""
快速测试λ值与码率的对应关系（无需熵模型）
使用简化ECVQ估算每个λ值产生的SKIP率和码率
"""

import torch
from pathlib import Path
from tqdm import tqdm
import sys

sys.path.insert(0, str(Path(__file__).parent))

from config import get_default_config
from grouped_rvq import GroupedResidualVQ
from data_loader import create_dataloaders
from dataclasses import replace

def test_lambda_range():
    """测试lambda值范围"""
    print("="*60)
    print("Lambda值与码率关系测试（简化ECVQ）")
    print("="*60)
    
    # 加载配置
    config = get_default_config()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 加载RVQ checkpoint
    rvq_model = GroupedResidualVQ(config['grouped_rvq']).to(device)
    ckpt_path = Path('checkpoints/grouped_rvq_best.pt')
    
    if not ckpt_path.exists():
        print(f"❌ RVQ checkpoint不存在: {ckpt_path}")
        return
    
    checkpoint = torch.load(ckpt_path, map_location=device)
    rvq_model.load_state_dict(checkpoint['model_state_dict'])
    rvq_model.eval()
    print(f"✅ 已加载RVQ checkpoint (还原率: {checkpoint['val_cosine']*100:.2f}%)")
    
    # 创建测试数据集（只用500个样本快速测试）
    test_data_config = replace(config['data'], max_samples=500)
    test_training_config = replace(config['training'], batch_size=32)
    
    train_loader, _ = create_dataloaders(
        test_data_config, test_training_config, config['grouped_rvq'],
        use_bucketing=False
    )
    
    print(f"✅ 加载测试数据: {len(train_loader.dataset)}个样本")
    
    # 获取lambda值（从config）
    lambda_values = config['entropy_model'].lambda_grid
    print(f"\n测试 {len(lambda_values)} 个λ值...")
    print("")
    
    # 测试每个λ值
    results = []
    
    for lam in lambda_values:
        total_skips = 0
        total_codes = 0
        total_frames = 0
        
        with torch.no_grad():
            for batch in train_loader:
                features = batch['features'].to(device)
                lengths = batch['lengths'].to(device)
                
                B, T = features.shape[0], features.shape[1]
                valid_mask = (torch.arange(T, device=device).unsqueeze(0) < lengths.unsqueeze(1))
                
                # 简化ECVQ（无熵模型）
                _, indices, _, stats = rvq_model(
                    features,
                    lambda_rate=torch.tensor(lam, device=device),
                    entropy_model=None,  # ✅ 简化模式
                    valid_mask=valid_mask
                )
                
                total_skips += stats.get('total_skips', 0)
                total_codes += stats.get('total_codes', 0)
                total_frames += valid_mask.sum().item()
        
        # 计算统计
        skip_rate = total_skips / total_codes if total_codes > 0 else 0
        
        # 粗略估算码率（假设均匀分布，每个非SKIP码字7 bits）
        approx_bpf = (total_codes - total_skips) / total_frames * 7 if total_frames > 0 else 0
        
        results.append({
            'lambda': lam,
            'skip_rate': skip_rate,
            'approx_bpf': approx_bpf
        })
        
        print(f"λ={lam:8.2f}: SKIP率={skip_rate:5.1%}, 粗略码率≈{approx_bpf:6.1f} bpf")
        
        # 清理GPU缓存
        torch.cuda.empty_cache()
    
    # 总结
    print("")
    print("="*60)
    print("测试总结")
    print("="*60)
    
    bitrates = [r['approx_bpf'] for r in results]
    print(f"码率范围: {min(bitrates):.1f} - {max(bitrates):.1f} bpf")
    
    # 统计各区间的λ值数量
    count_0_20 = sum(1 for b in bitrates if b <= 20)
    count_20_50 = sum(1 for b in bitrates if 20 < b <= 50)
    count_50_100 = sum(1 for b in bitrates if 50 < b <= 100)
    count_100_300 = sum(1 for b in bitrates if 100 < b <= 300)
    count_300_plus = sum(1 for b in bitrates if b > 300)
    
    print(f"\n码率分布:")
    print(f"  0-20 bpf:     {count_0_20}个λ值")
    print(f"  20-50 bpf:    {count_20_50}个λ值")
    print(f"  50-100 bpf:   {count_50_100}个λ值")
    print(f"  100-300 bpf:  {count_100_300}个λ值")
    print(f"  300+ bpf:     {count_300_plus}个λ值")
    
    print("")
    print("⚠️ 注意: 这是简化ECVQ的粗略估算")
    print("   真实码率需要训练完熵模型后测量")
    print("")
    print("✅ 测试完成！")

if __name__ == "__main__":
    test_lambda_range()

