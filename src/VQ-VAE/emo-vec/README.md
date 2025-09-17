# Emotion2Vec VQ-VAE 码本训练器

这是一个基于 VEVO 训练框架的 Emotion2Vec 特征码本训练器。它将原本用于 HuBERT 特征的 VQ-VAE 码本训练适配为使用 Emotion2Vec 特征。

## 功能特性

- **完全兼容 VEVO 架构**: 严格按照 VEVO 的训练方式，只替换特征提取器
- **Emotion2Vec 特征提取**: 支持真实的 Emotion2Vec 模型和 dummy 模式用于测试
- **灵活配置**: 支持多种码本大小、批次大小等参数配置
- **完整训练流程**: 包含训练、验证、检查点保存等完整功能
- **跨平台支持**: 提供 Linux/macOS 和 Windows 的启动脚本

## 文件结构

```
emo-vec/
├── emotion2vec_extractor.py          # Emotion2Vec 特征提取器
├── emotion2vec_trainer.py            # 原始训练器（已存在）
├── emotion2vec_vqvae_trainer.py      # 新的 VQ-VAE 训练器
├── train_emotion2vec_vqvae.py        # 训练启动脚本
├── train.sh                          # Linux/macOS 启动脚本
├── train.bat                         # Windows 启动脚本
├── config/
│   └── emotion2vec_vqvae_config.json # 训练配置文件
└── README.md                         # 本文件
```

## 安装依赖

确保已安装以下依赖：

```bash
# 基础依赖
torch>=1.9.0
torchaudio>=0.9.0
numpy
pyyaml

# VEVO 相关依赖（如果使用完整 VEVO 功能）
accelerate
transformers

# Emotion2Vec 相关依赖
# (根据你的 emotion2vec_extractor.py 实现而定)
```

## 快速开始

### 1. 使用 Dummy 特征测试

```bash
# Linux/macOS
chmod +x train.sh
./train.sh --data_root ./test_data --max_steps 1000

# Windows
train.bat --data_root .\test_data --max_steps 1000
```

### 2. 使用真实 Emotion2Vec 特征

```bash
# Linux/macOS
./train.sh --data_root /path/to/your/audio/data --no_dummy --max_steps 50000

# Windows
train.bat --data_root C:\path\to\your\audio\data --no_dummy --max_steps 50000
```

### 3. 直接使用 Python

```bash
python train_emotion2vec_vqvae.py \
    --config config/emotion2vec_vqvae_config.json \
    --data_root ./data \
    --exp_dir ./experiments/emotion2vec_vqvae \
    --use_dummy \
    --batch_size 8 \
    --max_steps 50000
```

## 配置说明

### 主要配置参数

```json
{
    "model": {
        "emotion2vec": {
            "model_name": "emotion2vec_base",    // Emotion2Vec 模型名称
            "use_dummy": true,                   // 是否使用 dummy 特征
            "feature_dim": 1024,                 // 特征维度
            "sample_rate": 16000,                // 采样率
            "normalize": true                    // 是否归一化
        },
        "repcodec": {
            "codebook_size": 1024,               // 码本大小
            "codebook_num": 1,                   // 码本数量
            "code_dim": 512                      // 码向量维度
        }
    },
    "train": {
        "batch_size": 8,                         // 批次大小
        "max_steps": 50000,                      // 最大训练步数
        "learning_rate": 1e-4,                   // 学习率
        "save_checkpoints_steps": 2000,          // 检查点保存间隔
        "valid_interval": 2000                   // 验证间隔
    }
}
```

### 数据格式

支持以下数据格式：

1. **音频文件夹**: 指定包含音频文件的目录路径
2. **文件列表**: 指定包含音频文件路径列表的文本文件

支持的音频格式：`.wav`, `.flac`, `.mp3`, `.ogg`

## 训练流程

1. **特征提取**: 使用 Emotion2Vec 从音频中提取特征
2. **VQ-VAE 编码**: 将特征编码到码本空间
3. **重构**: 从码本重构特征
4. **损失计算**: 计算重构损失和码本损失
5. **反向传播**: 更新模型参数

## 输出文件

训练过程中会生成以下文件：

```
experiments/emotion2vec_vqvae/
├── config.json                    # 保存的训练配置
├── train.log                      # 训练日志
└── checkpoints/
    ├── best_model.pt              # 最佳模型
    ├── final_model.pt             # 最终模型
    └── checkpoint_step_*.pt       # 定期保存的检查点
```

## 与原始 VEVO 的差异

| 方面 | VEVO (原始) | Emotion2Vec VQ-VAE |
|------|-------------|-------------------|
| 特征提取器 | HuBERT | Emotion2Vec |
| 特征维度 | 768 (HuBERT) | 1024 (Emotion2Vec) |
| 归一化 | HuBERT 统计量 | Emotion2Vec 统计量 |
| 数据集 | Emilia Dataset | 通用音频数据集 |

## 使用示例

### 训练小规模码本

```bash
python train_emotion2vec_vqvae.py \
    --config config/emotion2vec_vqvae_config.json \
    --data_root ./small_dataset \
    --batch_size 4 \
    --max_steps 10000 \
    --use_dummy
```

### 训练大规模码本

```bash
python train_emotion2vec_vqvae.py \
    --config config/emotion2vec_vqvae_config.json \
    --data_root /large/audio/dataset \
    --batch_size 16 \
    --max_steps 100000 \
    --no_dummy
```

### 从检查点恢复训练

```bash
python train_emotion2vec_vqvae.py \
    --config config/emotion2vec_vqvae_config.json \
    --resume ./experiments/emotion2vec_vqvae/checkpoints/checkpoint_step_20000.pt
```

## 故障排除

### 常见问题

1. **导入错误**: 确保 VEVO 代码路径正确添加到 PYTHONPATH
2. **内存不足**: 减小批次大小或序列长度
3. **特征提取失败**: 检查 Emotion2Vec 模型是否正确安装

### 调试模式

使用 dummy 特征进行快速测试：

```python
cfg.model.emotion2vec.use_dummy = True
cfg.debug = True  # 限制数据集大小
```

## 扩展功能

### 自定义特征提取器

可以通过修改 `emotion2vec_extractor.py` 来支持其他特征提取器：

```python
def create_custom_extractor(model_name, **kwargs):
    # 实现自定义特征提取逻辑
    pass
```

### 多码本训练

修改配置文件中的 `codebook_num` 参数：

```json
"repcodec": {
    "codebook_num": 4,  // 使用4个码本
    "codebook_size": 1024
}
```

## 性能优化

1. **批次大小**: 根据 GPU 内存调整批次大小
2. **数据加载**: 调整 `num_worker` 参数
3. **梯度累积**: 使用 `gradient_accumulation_step`
4. **混合精度**: 启用 VEVO 的混合精度训练

## 贡献指南

欢迎提交 issue 和 pull request 来改进这个项目。

## 许可证

本项目基于 MIT 许可证，与原始 VEVO 项目保持一致。