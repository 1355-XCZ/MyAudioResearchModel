# VEVO 码本训练源码备份

这个目录包含了从 Amphion-VevoDev 项目中复制的所有 VEVO 相关源码，用于参考和理解 VEVO 的码本训练实现。

## 目录结构

### 核心模块
- `vevo/` - VEVO 的核心实现
  - `vevo_repcodec.py` - VEVO 码本编解码器实现（包含 VectorQuantize 和 ResidualVQ）
  - `vqvae_trainer.py` - VEVO 码本训练器
  - `vevo_utils.py` - VEVO 工具函数
  - `infer_*.py` - 各种推理脚本

### 配置文件
- `vevo/config/` - 训练和模型配置
  - `hubert_large_l18_c32.yaml` - HuBERT 特征提取配置
  - `*.json` - 各种模型配置文件

### 基础组件
- `base/` - 基础训练器和数据集类
  - `base_trainer.py` - 基础训练器
  - `base_dataset.py` - 基础数据集
  - `emilia_dataset.py` - Emilia 数据集

### 工具函数
- `utils/` - 各种工具函数
  - `hubert.py` - HuBERT 特征提取
  - `audio.py` - 音频处理
  - `data_utils.py` - 数据处理工具

### 数据处理
- `processors/` - 各种特征提取器
  - `content_extractor.py` - 内容特征提取
  - `audio_features_extractor.py` - 音频特征提取

## 关键实现

### VEVO 码本训练的核心特点：
1. **EMA 更新**：使用指数移动平均更新码本，而非梯度下降
2. **Commitment Loss**：只使用 commitment loss，没有 codebook loss
3. **Residual VQ**：使用残差量化提高表征能力
4. **HuBERT 特征**：基于 HuBERT 的语音表征进行量化

### 主要文件说明：
- `vevo_repcodec.py` - 包含 VectorQuantize 类（EMA更新）和 ResidualVQ 类
- `vqvae_trainer.py` - 训练流程，展示如何结合重构损失和量化损失
- `vevo_utils.py` - 推理管道和工具函数

这些代码将作为构建 emotion2vec 版本码本训练器的参考基础。

