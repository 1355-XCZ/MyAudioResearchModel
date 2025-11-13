"""
配置管理 - 发布版情感RVQ信息瓶颈实验
"""

from dataclasses import dataclass, field
from typing import Optional, List
import os


@dataclass
class GroupedRVQConfig:
    """分组RVQ模型配置"""
    feature_dim: int = 768              # emotion2vec 特征维度
    num_groups: int = 12                # 分组数
    group_dim: int = 64                 # 每组维度 (768/12=64)
    
    # 组内细层
    num_fine_layers: int = 16           # 每组细量化层数
    fine_codebook_size: int = 128       # 细层码本大小
    
    # SKIP机制
    enable_skip: bool = True            # 启用SKIP机制
    
    # RVQ训练参数（EMA模式）
    decay: float = 0.99                 # EMA衰减率
    commitment_weight: float = 0.25     # commitment loss权重
    kmeans_init: bool = True            # K-means初始化
    kmeans_iters: int = 10              # K-means迭代次数
    threshold_ema_dead_code: float = 2.0

    # 训练控制（注：num_epochs在TrainingConfig中配置）
    eval_interval: int = 1              # 每个epoch都验证一次
    save_interval: int = 5              # 每5个epoch保存一次
    
    # 早停机制
    enable_early_stopping: bool = True   # 启用早停
    early_stopping_patience: int = 10    # 容忍的epoch数（无提升则停止）
    early_stopping_min_delta: float = 0.0001  # 最小改进阈值（0.01%还原率）
    
    @property
    def total_codebook_size_with_skip(self) -> int:
        """包含SKIP的码本大小"""
        if self.enable_skip:
            return self.fine_codebook_size + 1  # +1 for SKIP
        return self.fine_codebook_size
    
    @property
    def max_bits_per_frame(self) -> float:
        """理论最大bits/frame"""
        import numpy as np
        return self.num_groups * self.num_fine_layers * np.log2(self.fine_codebook_size)


@dataclass
class EntropyModelConfig:
    """无条件熵模型配置 - 仅 q(z)"""
    # 模型结构
    d_model: int = 256                  # Transformer隐藏维度
    num_layers: int = 6                 # Transformer层数
    num_heads: int = 8                  # 注意力头数
    d_ff: int = 1024                    # FFN隐藏维度
    dropout: float = 0.1
    
    # 训练参数
    batch_size: int = 16                # 批次大小（16层RVQ需要降低）
    num_epochs: int = 5                 # 训练epoch数
    train_data_fraction: float = 1.0    # 使用训练数据的比例（1.0=100%）
    
    # 早停机制
    enable_early_stopping: bool = True   # 启用早停
    early_stopping_patience: int = 2     # 容忍的epoch数（无提升则停止）
    early_stopping_min_delta: float = 0.01  # 最小改进阈值（Loss下降1%）
    
    # 性能优化
    enable_optimizations: bool = True   # 启用性能优化
    max_frames_per_sample: int = 128     # 裁剪长序列（降低显存）
    use_amp: bool = True                # 混合精度
    use_tf32: bool = True               # TF32加速
    
    def get_effective_config(self):
        """返回实际优化配置"""
        if not self.enable_optimizations:
            return {
                'max_frames_per_sample': 0,
                'use_amp': False,
                'use_tf32': False,
            }
        return {
            'max_frames_per_sample': self.max_frames_per_sample,
            'use_amp': self.use_amp,
            'use_tf32': self.use_tf32,
        }
    
    # 上下文（帧内AR）
    max_seq_len: int = 100              # 帧内最大序列长度
    causal: bool = True                 # 因果掩码
    
    # SKIP位置加权
    skip_weight: float = 0.5            # SKIP权重（平衡正常编码与SKIP）
    
    # 目标bpf网格（用于索引提取，分位数门控）- 朋友方案
    target_bpf_grid: List[float] = field(default_factory=lambda: [
        # 极低码率 (0-20 bpf)：密集 ⭐
        2, 5, 10, 15, 20, 25,
        # 低码率 (20-100 bpf)：密集 ⭐ 用户关心
        30, 35, 40, 50, 60, 80, 100,
        # 中高码率 (100-300 bpf)：中等
        150, 200, 300,
        # 高码率 (300-1344 bpf)：稀疏
        500, 1344,
        # 无量化baseline（原始特征）
        float('inf')  # 表示不量化，直接用原始特征
    ])  # 19个点（18个量化点 + 1个无量化baseline）

    # 训练优化
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 4000
    max_grad_norm: float = 1.0


@dataclass
class RateControlConfig:
    """码率控制配置"""
    # 目标码率列表（bps）
    target_rates_bps: List[int] = field(default_factory=lambda: [
        500,   # 10 bpf
        1000,  # 20 bpf
        2000,  # 40 bpf
        4000,  # 80 bpf
        8000   # 160 bpf
    ])

    # λ搜索范围
    lambda_min: float = 1e-4
    lambda_max: float = 16.0
    lambda_init: float = 0.5

    # 二分搜索参数
    rate_tolerance_bpf: float = 1.0  # 容差：1 bpf（未收敛可接受）
    max_binary_search_iters: int = 20
    
    # 对偶更新参数
    dual_lr: float = 0.01
    dual_momentum: float = 0.9
    dual_update_interval: int = 100
    
    # 帧率
    frame_rate_hz: float = 50.0
    
    @property
    def frame_duration_sec(self) -> float:
        """帧时长（秒）"""
        return 1.0 / self.frame_rate_hz
    
    def bps_to_bpf(self, bps: float) -> float:
        """bps → bits/frame"""
        return bps * self.frame_duration_sec
    
    def bpf_to_bps(self, bpf: float) -> float:
        """bits/frame → bps"""
        return bpf / self.frame_duration_sec


@dataclass
class EvaluationConfig:
    """评估配置"""
    # 评估数据集列表（使用面向对象设计）
    dataset_names: List[str] = field(default_factory=lambda: ['IEMOCAP', 'RAVDESS', 'ESD'])
    
    def __post_init__(self):
        """如果rate_sweep_rates_bpf为None，使用训练时的target_bpf_grid"""
        if self.rate_sweep_rates_bpf is None:
            # 从EntropyModelConfig获取target_bpf_grid
            # 需要在运行时设置
            pass
    
    # 方法1: 码率扫描（主要方法）
    enable_rate_sweep: bool = True
    # 先验知识：50 BPF以下是关键区域，重点密集测试
    rate_sweep_rates_bpf: List[float] = field(default_factory=lambda: [
        5, 10, 15, 20, 25, 30, 40, 50,  # 重点：5-50 BPF密集采样
        100, 200,  # 高码率参考点
        float('inf')  # 无量化baseline
    ])
    # 35个测试点，0-100 bpf区间有26个值（密集）⭐
    
    # 方法2: 层数扫描（辅助方法）
    enable_layer_sweep: bool = True
    # 先验知识：5层/组后变化不大，重点测试1-5层/组
    layer_sweep_layers: List[int] = field(default_factory=lambda: [
        12*1,  # 12层（12组×1层/组）
        12*2,  # 24层（12组×2层/组）
        12*3,  # 36层（12组×3层/组）
        12*4,  # 48层（12组×4层/组）
        12*5,  # 60层（12组×5层/组）
    ])  # 5个点，重点测试低层数区域
    
    # emotion2vec分类器配置
    emotion2vec_model: str = "iic/emotion2vec_plus_base"
    emotion2vec_hub: str = "ms"  # modelscope
    
    # 结果输出
    save_predictions: bool = True
    save_confusion_matrix: bool = True
    save_plots: bool = True


@dataclass
class DataConfig:
    """数据配置"""
    # 数据根目录（从环境变量或配置文件读取）
    data_root: Optional[str] = None
    
    # 训练数据（100h中英文）
    train_data_dir: str = "training_data"
    
    # 特征文件命名模式
    feature_suffix: str = "_ev2_frame.npy"
    label_suffix: str = "_emotion.txt"  # 标签文件后缀（训练数据无标签，评估时使用）
    
    # 数据分割
    train_split: float = 0.9
    seed: int = 1344871  # 统一随机种子
    
    # 数据处理
    max_samples: Optional[int] = None   # 最大样本数（用于快速测试）
    max_frames: int = 256               # 最大帧数（索引提取阶段使用，防止OOM）
    min_frames: int = 10                # 最小帧数
    normalize_features: bool = True     # 是否归一化特征
    mean_std_path: Optional[str] = "ev2_mean_std_100h_EN_ZH.npz"  # 归一化参数路径（待计算）
    supported_languages: Optional[List[str]] = field(default_factory=lambda: ['EN', 'ZH'])  # 使用100h中英文数据
    
    # 情感标签映射（训练数据无标签，用于评估时的占位）
    @property
    def emotion_label_map(self) -> dict:
        """情感标签映射（占位，训练数据无标签）"""
        return {}  # 训练数据无标签，返回空字典
    
    @property
    def train_data_path(self) -> str:
        return os.path.join(self.data_root, self.train_data_dir)


@dataclass
class TrainingConfig:
    """训练配置"""
    # 通用训练参数
    batch_size: int = 32  # 用于索引提取阶段
    num_epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    
    # 数据采样
    use_bucketing: bool = True
    num_buckets: int = 10
    
    # 优化器
    optimizer: str = "adamw"
    betas: tuple = (0.9, 0.999)
    eps: float = 1e-8
    
    # 学习率调度
    scheduler: str = "cosine"
    warmup_steps: int = 4000
    
    # 设备
    device: str = "cuda"
    num_workers: int = 4
    pin_memory: bool = True
    
    # 输出目录（使用项目内目录）
    output_dir: str = "checkpoints"
    exp_name: Optional[str] = None
    
    # Checkpoint控制
    resume_from_checkpoint: bool = False  # 是否从checkpoint恢复
    load_rvq_checkpoint: bool = True      # 熵模型训练时是否加载RVQ
    
    # 监控
    log_interval: int = 50
    eval_interval: int = 1
    save_interval: int = 5
    
    # TensorBoard
    use_tensorboard: bool = False  # 简化版不启用tensorboard
    
    def __post_init__(self):
        """初始化后处理"""
        if self.exp_name is None:
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.exp_name = f"exp_{timestamp}"
    
    @property
    def checkpoint_dir(self) -> str:
        return self.output_dir
    
    @property
    def log_dir(self) -> str:
        return os.path.join(self.output_dir, "logs")
    
    @property
    def results_dir(self) -> str:
        return os.path.join(self.output_dir, "results")


@dataclass
class SlurmConfig:
    """Slurm作业配置"""
    account: str = "punim2341"
    partition: str = "gpu-h100"
    
    nodes: int = 1
    ntasks: int = 1
    cpus_per_task: int = 4
    gpus: int = 1
    memory: str = "32G"
    time_limit: str = "08:00:00"
    
    project_root: str = "/data/gpfs/projects/punim2341/haoguangzhou/voice/MyAudioResearchModel"
    log_dir: str = "/data/gpfs/projects/punim2341/haoguangzhou/logs"
    venv_path: str = "/data/gpfs/projects/punim2341/haoguangzhou/venvs/vevo-source-fix"
    
    modules: List[str] = field(default_factory=lambda: [
        "GCCcore/11.3.0",
        "Python/3.10.4",
        "GCC/11.3.0",
        "CUDA/11.8.0",
        "cuDNN/8.7.0.84-CUDA-11.8.0"
    ])


def get_default_config():
    """获取默认配置"""
    return {
        'grouped_rvq': GroupedRVQConfig(),
        'entropy_model': EntropyModelConfig(),
        'rate_control': RateControlConfig(),
        'evaluation': EvaluationConfig(),
        'data': DataConfig(),
        'training': TrainingConfig(),
        'slurm': SlurmConfig(),
    }

