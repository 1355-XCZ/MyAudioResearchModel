# Issue #3: 评估需要预提取特征

**优先级**: 🟡 中  
**状态**: ⚠️ 待确认策略  
**影响**: 评估流程的便捷性

## 问题描述

评估脚本假设存在预提取的emotion2vec特征文件：

```python
# evaluation/method_rate_sweep.py 第60行
features_path = audio_path.replace('.wav', '_ev2_frame.npy')

if not Path(features_path).exists():
    logger.warning(f"特征文件不存在，跳过: {features_path}")
    continue  # ⚠️ 直接跳过，不提取
```

## 当前行为

- ✅ 如果有 `*_ev2_frame.npy`：正常加载评估
- ❌ 如果没有特征文件：跳过样本，给出警告

## 解决方案选项

### 选项A: 保持现状（推荐给发表版本）
- **优点**: 代码简洁，避免在线提取开销
- **缺点**: 需要预先准备特征
- **适用**: 科学实验（特征可复用）

### 选项B: 添加在线特征提取
```python
if not Path(features_path).exists():
    # 使用emotion2vec从音频提取特征
    features = extract_emotion2vec_features(audio_path)
    # 可选：缓存到文件
    np.save(features_path, features)
```
- **优点**: 用户友好，无需预处理
- **缺点**: 增加运行时间，重复计算

### 选项C: 提供独立的特征提取脚本
```bash
# 用户先运行特征提取
python extract_features.py --dataset IEMOCAP --output features/

# 然后运行评估
python run_evaluation.py
```
- **优点**: 清晰分离，特征可复用
- **缺点**: 需要额外步骤

## 建议

**发表版本**: 保持选项A
- 科学实验通常需要预提取特征（确保可重复性）
- 在README中说明需要预提取特征

**后续优化**: 可选添加选项C的特征提取脚本

## 需要的操作

1. 在README中添加"评估前提"部分
2. 说明需要预提取emotion2vec特征
3. （可选）提供特征提取脚本示例

