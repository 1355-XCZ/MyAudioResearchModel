"""
长度分桶采样器 (Bucket Batch Sampler)
用于减少padding，提高码本训练效率
"""

import torch
from torch.utils.data import Sampler, Subset
import numpy as np
from typing import Iterator, List


class BucketBatchSampler(Sampler):
    """
    按序列长度分桶的批次采样器
    
    原理：
    1. 将数据集按长度排序
    2. 分成若干桶（bucket）
    3. 每个bucket内随机采样batch
    4. 保证同一batch内的序列长度接近，减少padding
    
    使用示例:
        sampler = BucketBatchSampler(
            dataset,
            batch_size=32,
            num_buckets=10,
            shuffle=True
        )
        loader = DataLoader(dataset, batch_sampler=sampler, ...)
    """
    
    def __init__(
        self,
        dataset,
        batch_size: int,
        num_buckets: int = 10,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: int = 42
    ):
        """
        Args:
            dataset: 数据集（需要有length信息）
            batch_size: 批次大小
            num_buckets: 桶的数量（越多，长度越接近，但随机性降低）
            shuffle: 是否在桶内打乱
            drop_last: 是否丢弃最后不足batch_size的数据
            seed: 随机种子
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_buckets = num_buckets
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0
        
        # 收集所有样本的长度
        self.lengths = self._get_lengths()
        
        # 按长度分桶
        self.buckets = self._create_buckets()
        
        # 计算总batch数
        self.num_batches = sum(
            len(bucket) // batch_size if drop_last else (len(bucket) + batch_size - 1) // batch_size
            for bucket in self.buckets
        )
    
    def _get_lengths(self) -> np.ndarray:
        """获取所有样本的长度（优化：避免Subset二次IO）"""
        # 优先从Subset的底层数据集读取缓存长度
        if isinstance(self.dataset, Subset) and hasattr(self.dataset.dataset, 'get_length'):
            base = self.dataset.dataset
            idxs = self.dataset.indices
            return np.array([base.get_length(j) for j in idxs], dtype=np.int64)
        
        # 回退方案
        lengths = []
        for i in range(len(self.dataset)):
            try:
                if hasattr(self.dataset, 'get_length'):
                    length = int(self.dataset.get_length(i))
                else:
                    # 加载样本并获取长度
                    sample = self.dataset[i]
                    if isinstance(sample, dict) and 'length' in sample:
                        length = int(sample['length'].item())
                    elif isinstance(sample, dict) and 'features' in sample:
                        length = int(sample['features'].shape[0])
                    else:
                        length = 100  # 默认值
            except Exception:
                length = 100  # fallback
            
            lengths.append(length)
        
        return np.array(lengths, dtype=np.int64)
    
    def _create_buckets(self) -> List[List[int]]:
        """按长度分桶（使用linspace均匀切分，更稳健）"""
        # 按长度排序的索引
        sorted_indices = np.argsort(self.lengths)
        
        # 使用linspace均匀切分，避免空桶
        edges = np.linspace(0, len(sorted_indices), num=self.num_buckets + 1, dtype=np.int64)
        buckets = []
        
        for i in range(len(edges) - 1):
            start, end = edges[i], edges[i + 1]
            if start < end:
                bucket = sorted_indices[start:end].tolist()
                buckets.append(bucket)
        
        return buckets
    
    def __iter__(self) -> Iterator[List[int]]:
        """生成批次"""
        # 设置随机种子（保证epoch间可复现）
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        
        all_batches = []
        
        # 逐桶生成batch
        for bucket in self.buckets:
            # 桶内打乱（如果需要）
            if self.shuffle:
                indices = torch.randperm(len(bucket), generator=g).tolist()
                bucket_shuffled = [bucket[i] for i in indices]
            else:
                bucket_shuffled = bucket
            
            # 分batch
            for i in range(0, len(bucket_shuffled), self.batch_size):
                batch = bucket_shuffled[i:i + self.batch_size]
                
                if len(batch) == self.batch_size or not self.drop_last:
                    all_batches.append(batch)
        
        # 打乱所有batch的顺序（保持桶内聚性，但batch间随机）
        if self.shuffle:
            batch_indices = torch.randperm(len(all_batches), generator=g).tolist()
            all_batches = [all_batches[i] for i in batch_indices]

        # DDP 支持：让每个 rank 取不同子序列（步长切片）
        # 修复：padding到world_size的整数倍，避免rank间批次数不一致导致死锁
        rank, world_size = 0, 1
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
            
            # 计算需要padding的批次数
            n = len(all_batches)
            pad = (-n) % world_size  # 需要补齐的批次数
            
            if pad > 0:
                # 复制前pad个batch填充（循环复用）
                all_batches += all_batches[:pad]
            
            # 现在每个rank得到相同数量的批次
            all_batches = all_batches[rank::world_size]

        return iter(all_batches)
    
    def __len__(self) -> int:
        """返回总batch数（DDP 友好：返回当前 rank 视角的批次数）"""
        import math
        n = self.num_batches
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()
            # 与 __iter__ 的padding逻辑保持一致
            # padding后总批次数: n + ((-n) % world_size)
            n_padded = n + ((-n) % world_size)
            # 每个rank得到的批次数严格相同
            return n_padded // world_size
        return n
    
    def set_epoch(self, epoch: int):
        """设置epoch（用于分布式训练和随机性控制）"""
        self.epoch = epoch


def test_bucket_sampler():
    """测试BucketBatchSampler"""
    from torch.utils.data import Dataset, DataLoader
    
    # 创建模拟数据集
    class DummyDataset(Dataset):
        def __init__(self, sizes):
            self.sizes = sizes
        
        def __len__(self):
            return len(self.sizes)
        
        def __getitem__(self, idx):
            return {
                'features': torch.zeros(self.sizes[idx], 10),
                'length': torch.tensor([self.sizes[idx]])
            }
    
    # 创建长度分布不均的数据集
    np.random.seed(42)
    sizes = np.random.randint(50, 200, size=100)
    dataset = DummyDataset(sizes)
    
    # 使用BucketBatchSampler
    sampler = BucketBatchSampler(
        dataset,
        batch_size=8,
        num_buckets=5,
        shuffle=True
    )
    
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=lambda x: x
    )
    
    print(f"总batch数: {len(loader)}")
    
    # 检查前几个batch的长度分布
    for i, batch in enumerate(loader):
        if i >= 5:
            break
        lengths = [item['length'].item() for item in batch]
        print(f"Batch {i}: 长度范围 [{min(lengths)}, {max(lengths)}], "
              f"标准差 {np.std(lengths):.1f}")
    
    # 对比标准DataLoader
    print("\n对比：标准DataLoader（无分桶）")
    standard_loader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=lambda x: x)
    for i, batch in enumerate(standard_loader):
        if i >= 5:
            break
        lengths = [item['length'].item() for item in batch]
        print(f"Batch {i}: 长度范围 [{min(lengths)}, {max(lengths)}], "
              f"标准差 {np.std(lengths):.1f}")


if __name__ == "__main__":
    test_bucket_sampler()

