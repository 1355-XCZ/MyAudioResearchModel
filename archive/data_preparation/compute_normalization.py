"""
计算100h训练集（EN+ZH）的均值和标准差
用于特征归一化
"""

import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse


def compute_mean_std(data_dir, output_path, max_samples=None):
    """
    计算训练集特征的均值和标准差
    
    Args:
        data_dir: 数据目录（包含EN/和ZH/子目录）
        output_path: 输出路径（.npz）
        max_samples: 最大样本数（用于快速测试）
    """
    data_dir = Path(data_dir)
    
    # 收集所有特征文件（递归扫描EN/和ZH/子目录）
    feature_files = list(data_dir.rglob("*_ev2_frame.npy"))
    
    if max_samples:
        feature_files = feature_files[:max_samples]
    
    if len(feature_files) == 0:
        raise FileNotFoundError(f"❌ 未在 {data_dir} 找到 *_ev2_frame.npy 特征文件")
    
    print(f"找到 {len(feature_files)} 个特征文件")
    
    # 向量化计算均值和方差（文件级累积）
    sum_x = None
    sum_x2 = None
    n = 0
    
    for fpath in tqdm(feature_files, desc="计算统计量"):
        try:
            features = np.load(fpath, mmap_mode='r')  # (T, 768) 使用内存映射
            
            if features.ndim != 2:
                continue
            
            T, D = features.shape
            
            if sum_x is None:
                sum_x = np.zeros(D, dtype=np.float64)
                sum_x2 = np.zeros(D, dtype=np.float64)
            
            # 文件级向量化累积
            sum_x += features.sum(axis=0, dtype=np.float64)
            sum_x2 += (features.astype(np.float64) ** 2).sum(axis=0)
            n += T
        
        except Exception as e:
            print(f"加载失败: {fpath}, 错误: {e}")
            continue
    
    # 检查是否成功读取到帧
    if n == 0:
        raise RuntimeError("❌ 没有成功读取任何帧，无法计算 mean/std")
    
    # 计算均值和标准差
    mean = sum_x / n
    variance = np.maximum(sum_x2 / n - mean**2, 0.0)  # 防止数值误差导致负值
    std = np.sqrt(variance)
    
    # 保存
    np.savez(output_path, mean=mean.astype(np.float32), std=std.astype(np.float32))
    
    print(f"✅ 归一化参数已保存: {output_path}")
    print(f"   - 总帧数: {n}")
    print(f"   - 总样本数: {len(feature_files)}")
    print(f"   - Mean范围: [{mean.min():.4f}, {mean.max():.4f}]")
    print(f"   - Std范围: [{std.min():.4f}, {std.max():.4f}]")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, 
                       default='/data/gpfs/projects/punim2341/haoguangzhou/data/emilia_vevo_training_50h',
                       help='数据目录（包含EN/和ZH/子目录）')
    parser.add_argument('--output', type=str,
                       default='ev2_mean_std_100h_EN_ZH.npz',
                       help='输出文件路径')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='最大样本数（用于测试）')
    
    args = parser.parse_args()
    
    print("计算100h中英文数据的归一化参数")
    print(f"数据目录: {args.data_dir}")
    print(f"输出文件: {args.output}")
    
    compute_mean_std(args.data_dir, args.output, args.max_samples)


if __name__ == "__main__":
    main()

