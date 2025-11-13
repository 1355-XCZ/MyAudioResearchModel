# Issue #5: 情感标签映射策略待确认

**优先级**: 🔴 高  
**状态**: ⚠️ 待用户确认  
**影响**: 评估准确率计算

## 需要确认的映射

### 1. IEMOCAP - frustrated

**当前映射**: frustrated → angry

**数据分析结果**（1,849个样本）：
| emotion2vec预测 | 比例 |
|----------------|------|
| neutral | 49.5% ⭐ |
| angry | 33.8% |
| sad | 9.1% |

**结论**: frustrated在情感空间上更接近neutral（低激活度），而非angry（高激活度）

**建议**: frustrated → **neutral**

**影响**: 修改后IEMOCAP准确率可能从74.16%提升到75-78%

### 2. IEMOCAP - excited

**当前映射**: excited → happy

**数据分析结果**（1,041个样本）：
| emotion2vec预测 | 比例 |
|----------------|------|
| happy | 88.9% ⭐ |
| neutral | 8.5% |

**结论**: 映射正确

**建议**: 保持 excited → happy

### 3. RAVDESS - calm

**当前映射**: calm → neutral

**数据分析结果**（192个样本）：
| emotion2vec预测 | 比例 |
|----------------|------|
| neutral | 92.2% ⭐ |

**结论**: 映射正确

**建议**: 保持 calm → neutral

## 决策选项

### 选项A: 数据驱动映射（推荐）
基于emotion2vec实际分类分布：
```python
# datasets/iemocap_dataset.py
'fru': 'neutral',  # 从angry改为neutral
'exc': 'happy',    # 保持
```

### 选项B: 语义映射
基于情感语义相似性（当前设置）：
```python
'fru': 'angry',  # 保持
'exc': 'happy',  # 保持
```

### 选项C: 过滤有争议类别
```python
def get_emotion_filter(self):
    # 排除frustrated和excited
    return ['ang', 'hap', 'sad', 'neu']  # 只用4类核心情感
```

## 修复位置

`datasets/iemocap_dataset.py` 第47行：
```python
def get_emotion_mapping(self) -> Dict[str, str]:
    return {
        # ...
        'fru': 'angry',  # TODO: 改为'neutral'？
        'exc': 'happy',  # TODO: 保持
        # ...
    }
```

## 参考文献

- `data_analysis/EMOTION2VEC_CLASSIFICATION_ANALYSIS_REPORT.md`
- `data_analysis/EMOTION_ALIGNMENT_SUMMARY.md`

## 用户决策

请用户确认采用哪个选项（A/B/C），然后修改对应文件。
