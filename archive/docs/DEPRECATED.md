# 废弃文件标记

本文档列出项目中已废弃或可安全删除的文件，方便后续清理。

## ❌ 已废弃的优化代码（README明确说明已撤回）

```
evaluation/lambda_calibration.py
```
**原因**: λ标定表优化，收益<1%，已撤回
**状态**: 可安全删除
**建议**: 移到archive/或直接删除

```
evaluation/method_rate_sweep_batch.py
```
**原因**: 批处理优化，真ECVQ无法批处理，反而变慢，已撤回
**状态**: 可安全删除
**建议**: 移到archive/或直接删除

## 🔄 重复文件

```
extract_eval_features.py
```
**重复于**: extract_evaluation_features.py
**状态**: 可安全删除
**建议**: 删除extract_eval_features.py，保留extract_evaluation_features.py

```
run_evaluation.py
run_evaluation.slurm
```
**重复于**: run_evaluation_iemocap.py, run_evaluation_ravdess.py, run_evaluation_esd.py
**原因**: 旧版串行评估，已被分数据集并行版本替代
**状态**: 可安全删除

```
run_evaluation_test.py
run_evaluation_test.slurm
```
**原因**: 测试版本，已有正式版本
**状态**: 可安全删除

## 🧪 临时测试脚本（开发完成后可归档）

```
test_setup.py
test_classification_accuracy.py
test_entropy_quick.py
test_lambda_bitrate.py
test_lambda_range.py

run_test_entropy_quick.slurm
run_test_lambda.slurm
run_train_rvq_test.slurm
```
**原因**: 开发和调试时使用的临时测试
**状态**: 功能已验证，可归档
**建议**: 移到archive/tests/

## ✅ 验证脚本（功能完成后可归档）

```
verify_extracted_features.py
verify_features_classification.py
verify_sampled_accuracy.py

run_verify_features.slurm
run_verify_sampled.slurm
```
**原因**: 一次性验证功能，已完成验证
**状态**: 可归档
**建议**: 移到archive/verification/

## 🏃 快速测试脚本（开发时使用）

```
run_evaluation_esd_quick.py
run_evaluation_esd_quick_v2.py

run_eval_esd_quick.slurm
run_eval_esd_quick_v2.slurm
```
**原因**: 开发时快速验证流程（20样本/情感）
**状态**: 可归档（保留用于快速测试）
**建议**: 移到archive/quick_tests/

## 🔧 一次性数据生成脚本（已完成）

```
generate_iemocap_labels.py
regenerate_ravdess_labels.py
compute_normalization.py

run_compute_normalization.slurm
```
**原因**: 一次性运行生成数据/标签/归一化参数
**状态**: 已完成，可归档
**建议**: 移到archive/data_preparation/

## 📁 可清理的结果目录

```
evaluation_quick_results/
```
**原因**: 旧版快速测试结果
**状态**: 已被evaluation_quick_v2_results/替代
**建议**: 直接删除

```
evaluation_test_results/
```
**原因**: 测试评估结果
**状态**: 测试用，非正式结果
**建议**: 直接删除

## 💾 可删除的中间检查点

```
checkpoints/grouped_rvq_epoch5.pt
checkpoints/grouped_rvq_epoch10.pt
checkpoints/grouped_rvq_epoch15.pt
checkpoints/grouped_rvq_epoch20.pt
checkpoints/grouped_rvq_epoch25.pt
checkpoints/grouped_rvq_epoch30.pt
```
**原因**: 中间epoch检查点，已有best模型
**状态**: grouped_rvq_best.pt已包含最佳模型
**建议**: 直接删除，节省空间

## 📚 可归档的文档和分析

```
issues/
```
**内容**: 已解决的问题记录
**状态**: 历史记录，可归档
**建议**: 移到archive/issues/

```
analyze/
```
**内容**: 分析结果和图表
**状态**: 历史分析，可归档
**建议**: 移到archive/analysis/

```
HANDOVER.md
LAMBDA_TESTING_WITHOUT_ENTROPY.md
```
**内容**: 交接文档和测试文档
**状态**: 历史文档，可归档
**建议**: 移到archive/docs/

## 🗂️ 推荐的清理脚本

### 方案1: 完全删除（最激进，节省最多空间）

```bash
cd /data/gpfs/projects/punim2341/haoguangzhou/voice/MyAudioResearchModel/src/Amphion/models/vc/my_publish_emo_rvq_bottleneck

# 删除废弃的优化代码
rm evaluation/lambda_calibration.py
rm evaluation/method_rate_sweep_batch.py

# 删除重复文件
rm extract_eval_features.py
rm run_evaluation.py run_evaluation.slurm
rm run_evaluation_test.py run_evaluation_test.slurm

# 删除测试脚本
rm test_*.py
rm run_test_*.slurm

# 删除验证脚本
rm verify_*.py
rm run_verify_*.slurm

# 删除快速测试脚本
rm run_evaluation_esd_quick*.py
rm run_eval_esd_quick*.slurm

# 删除数据生成脚本
rm generate_iemocap_labels.py
rm regenerate_ravdess_labels.py
rm run_compute_normalization.slurm

# 删除旧结果目录
rm -rf evaluation_quick_results/
rm -rf evaluation_test_results/

# 删除中间检查点
rm checkpoints/grouped_rvq_epoch*.pt

# 删除文档和分析
rm -rf issues/
rm -rf analyze/
rm HANDOVER.md LAMBDA_TESTING_WITHOUT_ENTROPY.md
```

### 方案2: 归档保留（推荐，保留历史记录）

```bash
cd /data/gpfs/projects/punim2341/haoguangzhou/voice/MyAudioResearchModel/src/Amphion/models/vc/my_publish_emo_rvq_bottleneck

# 创建归档目录
mkdir -p archive/{deprecated,tests,verification,quick_tests,data_preparation,docs,analysis,issues}

# 归档废弃代码
mv evaluation/lambda_calibration.py archive/deprecated/
mv evaluation/method_rate_sweep_batch.py archive/deprecated/

# 归档测试脚本
mv test_*.py archive/tests/
mv run_test_*.slurm archive/tests/

# 归档验证脚本
mv verify_*.py archive/verification/
mv run_verify_*.slurm archive/verification/

# 归档快速测试
mv run_evaluation_esd_quick*.py archive/quick_tests/
mv run_eval_esd_quick*.slurm archive/quick_tests/

# 归档数据生成脚本
mv generate_iemocap_labels.py archive/data_preparation/
mv regenerate_ravdess_labels.py archive/data_preparation/
mv run_compute_normalization.slurm archive/data_preparation/

# 归档文档
mv issues/ archive/
mv analyze/ archive/analysis/
mv HANDOVER.md archive/docs/
mv LAMBDA_TESTING_WITHOUT_ENTROPY.md archive/docs/

# 删除重复文件（无需保留）
rm extract_eval_features.py
rm run_evaluation.py run_evaluation.slurm
rm run_evaluation_test.py run_evaluation_test.slurm

# 删除旧结果（无需保留）
rm -rf evaluation_quick_results/
rm -rf evaluation_test_results/

# 删除中间检查点（无需保留）
rm checkpoints/grouped_rvq_epoch*.pt
```

### 方案3: 最小清理（最保守，仅删除明显无用的）

```bash
cd /data/gpfs/projects/punim2341/haoguangzhou/voice/MyAudioResearchModel/src/Amphion/models/vc/my_publish_emo_rvq_bottleneck

# 仅删除重复和过时的文件
rm extract_eval_features.py
rm -rf evaluation_quick_results/
rm -rf evaluation_test_results/

# 删除中间检查点（保留best即可）
rm checkpoints/grouped_rvq_epoch*.pt
```

## 📊 空间节省估算

- **Python缓存**: ~50-100MB （已删除✅）
- **中间检查点**: ~3-5GB
- **旧评估结果**: ~500MB-1GB
- **归档脚本和文档**: ~10-20MB

**总计**: 方案1可节省 ~4-6GB，方案2可节省 ~3.5-5.5GB，方案3可节省 ~3.5-5GB

---
最后更新: 2025-11-13
**建议**: 使用方案2（归档保留），既保留历史记录又清理了主目录

