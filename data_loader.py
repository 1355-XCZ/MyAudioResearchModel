"""
数据加载器 - 最小版
支持加载 emotion2vec 特征和情感标签
"""

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict
import logging
import random

logger = logging.getLogger(__name__)


def _worker_init_fn(worker_id):
    """
    DataLoader worker 初始化函数
    确保多进程下的随机性可控且不冲突
    """
    # 获取PyTorch的随机种子
    worker_seed = torch.initial_seed() % 2**32
    
    # 设置numpy和python的随机种子
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class EmotionDataset(Dataset):
    """
    emotion2vec 帧级特征 + 情感标签数据集
    """
    
    def __init__(
        self,
        data_dir: str,
        max_samples: Optional[int] = None,
        max_frames: int = 200,
        min_frames: int = 10,
        feature_dim: int = 768,
        supported_languages: Optional[list] = None,
        normalize: bool = True,
        mean_std_path: Optional[str] = None,
        feature_suffix: str = "_ev2_frame.npy",
        label_suffix: str = "_emotion.txt",
        emotion_label_map: Optional[dict] = None,
        seed: int = 42
    ):
        """
        Args:
            data_dir: 数据目录（如 emilia_vevo_training_50h）
            max_samples: 最大样本数（用于快速测试）
            max_frames: 最大帧数（截断）
            min_frames: 最小帧数（过滤）
            feature_dim: 特征维度（默认 768）
            supported_languages: 支持的语言列表（默认 ['EN', 'ZH']）
            normalize: 是否归一化特征
            mean_std_path: 归一化参数路径
            feature_suffix: 特征文件后缀
            label_suffix: 标签文件后缀
            emotion_label_map: 情感标签映射 {label_str: class_id}
            seed: 随机种子
        """
        self.data_dir = Path(data_dir)
        self.max_frames = max_frames
        self.min_frames = min_frames
        self.feature_dim = feature_dim
        self.normalize = normalize
        self.feature_suffix = feature_suffix
        self.label_suffix = label_suffix
        
        # 情感标签映射（支持别名和大小写不敏感）
        if emotion_label_map is None:
            # 默认映射（5类，兼容ESD）
            self.emotion_label_map = {
                'angry': 0, 'anger': 0, 'ang': 0,
                'happy': 1, 'happiness': 1, 'excited': 1, 'hap': 1,
                'neutral': 2, 'neu': 2, 'frustrated': 2,  # 与 DataConfig 保持一致
                'sad': 3, 'sadness': 3,
                'surprise': 4, 'surprised': 4, 'sur': 4,
            }
        else:
            self.emotion_label_map = emotion_label_map
        
        self.unknown_labels_count = 0  # 统计未知标签数
        
        # 特征归一化参数
        self.mean = None
        self.std = None
        if normalize and mean_std_path and Path(mean_std_path).exists():
            data = np.load(mean_std_path)
            self.mean = torch.from_numpy(data['mean']).float()  # (768,)
            self.std = torch.from_numpy(data['std']).float().clamp_min(1e-5)
            logger.info(f"✅ 加载归一化参数: {mean_std_path}")
        elif normalize and not mean_std_path:
            logger.warning("⚠️ normalize_features=True 但未设置 mean_std_path")
            logger.warning("   将跳过归一化。请先运行 compute_normalization.py 并设置路径。")
        elif normalize and mean_std_path and not Path(mean_std_path).exists():
            logger.warning(f"⚠️ normalize_features=True 但文件不存在: {mean_std_path}")
            logger.warning("   将跳过归一化。")
        
        # 收集所有特征文件并缓存长度
        self.feature_files = []
        self.lengths = []  # 缓存每个样本的帧长
        
        if supported_languages is None:
            # 无语言子目录，递归搜索整个data_dir
            frame_files = sorted(self.data_dir.rglob(f"*{self.feature_suffix}"))
            self.feature_files.extend(frame_files)
        else:
            # 有语言子目录
            for lang in supported_languages:
                lang_dir = self.data_dir / lang
                if not lang_dir.exists():
                    logger.warning(f"目录不存在: {lang_dir}")
                    continue
                
                # 帧级特征文件
                frame_files = sorted(lang_dir.glob(f"*{self.feature_suffix}"))
                self.feature_files.extend(frame_files)
        
        if len(self.feature_files) == 0:
            raise ValueError(f"未找到任何特征文件（后缀={self.feature_suffix}）: {data_dir}")
        
        # 过滤：检查帧数范围并缓存长度（超长样本保留，截断在__getitem__中处理）
        valid_files = []
        valid_lengths = []
        for fpath in self.feature_files:
            try:
                feat = np.load(fpath, mmap_mode='r')  # 内存映射，避免读入整个数组
                T = feat.shape[0]
                if T >= self.min_frames:  # 只按下限过滤
                    valid_files.append(fpath)
                    # 分桶用的长度：截断到max_frames，与实际batch长度一致
                    valid_lengths.append(min(T, self.max_frames))
            except:
                continue
        
        self.feature_files = valid_files
        self.lengths = valid_lengths
        
        # 洗牌（固定随机种子）- 对文件和长度同步洗牌
        np.random.seed(seed)
        perm = np.random.permutation(len(self.feature_files))
        self.feature_files = [self.feature_files[i] for i in perm]
        self.lengths = [self.lengths[i] for i in perm]
        
        # 限制样本数
        if max_samples is not None:
            self.feature_files = self.feature_files[:max_samples]
            self.lengths = self.lengths[:max_samples]
        
        logger.info(f"✅ EmotionDataset: {len(self.feature_files)} 个样本")
    
    def __len__(self):
        return len(self.feature_files)
    
    def get_length(self, idx: int) -> int:
        """获取样本长度（用于分桶采样器，避免二次IO）"""
        return int(self.lengths[idx])
    
    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        """
        返回:
            {
                'features': (T, 768) emotion2vec 帧级特征
                'label': (1,) 情感标签（如果有）
                'length': (1,) 实际帧数
            }
        """
        feature_file = self.feature_files[idx]
        
        try:
            # 加载特征（内存映射，避免读入整个数组）
            arr = np.load(feature_file)  # (T, 768) - 不使用内存映射以提高性能
            T = arr.shape[0]

            # 截断或填充
            if T > self.max_frames:
                # 随机截取
                start = np.random.randint(0, T - self.max_frames + 1)
                arr = arr[start:start + self.max_frames]  # 先切片
                T = self.max_frames

            features = torch.from_numpy(np.array(arr, copy=True)).float()
            
            # 特征归一化
            if self.normalize and self.mean is not None and self.std is not None:
                features = (features - self.mean) / self.std
            
            # 尝试加载情感标签（如果存在）- 使用统一映射
            label_file = feature_file.with_name(
                feature_file.name.replace(self.feature_suffix, self.label_suffix)
            )
            if label_file.exists():
                with open(label_file, 'r') as f:
                    label_str = f.read().strip().lower()  # 转小写
                    
                    # 使用统一的映射（支持别名）
                    if label_str in self.emotion_label_map:
                        label_id = self.emotion_label_map[label_str]
                    else:
                        # 未知标签：记录并使用neutral
                        self.unknown_labels_count += 1
                        label_id = 2  # Neutral
                        if self.unknown_labels_count <= 10:  # 只打印前10个
                            logger.warning(f"未知情感标签: '{label_str}' 在文件 {feature_file.name}，使用Neutral")
                    
                    label = torch.tensor([label_id], dtype=torch.long)
            else:
                label = torch.tensor([2], dtype=torch.long)  # 默认Neutral
            
            return {
                'features': features,  # (T, 768)
                'label': label,        # (1,)
                'length': torch.tensor([T], dtype=torch.long)
            }
        
        except Exception as e:
            logger.error(f"加载失败: {feature_file}, 错误: {e}")
            # 返回零向量作为fallback
            return {
                'features': torch.zeros(self.min_frames, self.feature_dim, dtype=torch.float32),
                'label': torch.tensor([2], dtype=torch.long),
                'length': torch.tensor([self.min_frames], dtype=torch.long)
            }


def collate_fn(batch):
    """
    处理变长序列：padding
    
    Args:
        batch: List of dicts
    
    Returns:
        {
            'features': (B, T_max, 768) 填充后的特征
            'labels': (B,) 情感标签
            'lengths': (B,) 实际长度
        }
    """
    # 获取最大长度和 dtype
    lengths = torch.cat([item['length'] for item in batch])
    max_len = lengths.max().item()
    feature_dim = batch[0]['features'].shape[1]
    dtype = batch[0]['features'].dtype  # 保留原始 dtype
    
    # Padding
    batch_size = len(batch)
    features = torch.zeros(batch_size, max_len, feature_dim, dtype=dtype)
    labels = torch.cat([item['label'] for item in batch])  # (B,)
    
    for i, item in enumerate(batch):
        T = item['features'].shape[0]
        features[i, :T] = item['features']
    
    return {
        'features': features,  # (B, T_max, 768)
        'labels': labels,      # (B,)
        'lengths': lengths     # (B,)
    }


def create_dataloaders(
    data_config,
    training_config,
    rvq_config,
    use_bucketing: bool = False,
    num_buckets: int = 10
) -> Tuple[DataLoader, DataLoader]:
    """
    创建训练和验证数据加载器
    
    Args:
        data_config: DataConfig实例
        training_config: TrainingConfig实例
        rvq_config: GroupedRVQConfig实例
        use_bucketing: 是否使用长度分桶（减少padding）
        num_buckets: 桶的数量
    
    Returns:
        train_loader, val_loader
    """
    # 创建完整数据集
    full_dataset = EmotionDataset(
        data_dir=data_config.train_data_path,
        max_samples=data_config.max_samples,
        max_frames=data_config.max_frames,
        min_frames=data_config.min_frames,
        feature_dim=rvq_config.feature_dim,
        supported_languages=data_config.supported_languages,
        normalize=data_config.normalize_features,
        mean_std_path=data_config.mean_std_path,
        feature_suffix=data_config.feature_suffix,
        label_suffix=data_config.label_suffix,
        emotion_label_map=data_config.emotion_label_map,  # 使用统一的标签映射
        seed=data_config.seed
    )
    
    # 计算分割大小
    total_size = len(full_dataset)
    train_size = int(data_config.train_split * total_size)
    val_size = total_size - train_size
    
    # 分割数据集
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(data_config.seed)
    )
    
    logger.info(f"✅ 训练集: {len(train_dataset)} 个样本")
    logger.info(f"✅ 验证集: {len(val_dataset)} 个样本")
    
    # DataLoader 通用参数（避免 prefetch_factor=None 在单进程时报错）
    persistent_workers = training_config.num_workers > 0
    
    # 构建通用 kwargs
    common_kwargs = {
        'num_workers': training_config.num_workers,
        'pin_memory': training_config.pin_memory and torch.cuda.is_available(),  # 只有 CUDA 场景下才有收益
        'collate_fn': collate_fn,
        'worker_init_fn': _worker_init_fn if training_config.num_workers > 0 else None,
        'persistent_workers': persistent_workers,
    }
    
    # 只在多进程时添加 prefetch_factor
    if persistent_workers:
        common_kwargs['prefetch_factor'] = 2  # 视 I/O 能力可调到 3
    
    # 创建数据加载器
    if use_bucketing:
        # 使用长度分桶采样器
        try:
            from .bucket_sampler import BucketBatchSampler
        except ImportError:
            from bucket_sampler import BucketBatchSampler
        
        train_sampler = BucketBatchSampler(
            train_dataset,
            batch_size=training_config.batch_size,
            num_buckets=num_buckets,
            shuffle=True,
            drop_last=True,
            seed=data_config.seed
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            **common_kwargs
        )
        
        logger.info(f"✅ 使用长度分桶采样器 (num_buckets={num_buckets})")
    else:
        # 标准DataLoader
        train_loader = DataLoader(
            train_dataset,
            batch_size=training_config.batch_size,
            shuffle=True,
            drop_last=True,
            **common_kwargs
        )
    
    # 验证集不需要分桶
    val_loader = DataLoader(
        val_dataset,
        batch_size=training_config.batch_size,
        shuffle=False,
        drop_last=False,
        **common_kwargs
    )
    
    return train_loader, val_loader

