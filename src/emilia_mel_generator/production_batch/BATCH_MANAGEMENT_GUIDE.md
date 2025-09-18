# 大规模数据集分批处理指南

## 🎯 **问题解决**

针对**集群服务器很难一次性全部执行完成**的问题，我们设计了健壮的分批处理系统：

- ✅ **自动分批**: 将大数据集分割成可管理的小batch
- ✅ **断点续传**: 任务中断后可从断点继续
- ✅ **并行处理**: 同时运行多个batch任务
- ✅ **自动重试**: 失败的batch自动重试
- ✅ **状态监控**: 实时监控所有batch状态

## 🚀 **使用方式**

### **1. 启动分批任务管理**

```bash
cd production_batch

# 使用默认配置（每批500样本，最多3个并发）
./submit_batch_manager.sh

# 自定义配置
./submit_batch_manager.sh -s 1000 -j 5 -H 25.0
```

### **2. 监控任务状态**

```bash
# 实时监控（每60秒更新）
./submit_batch_manager.sh monitor

# 查看当前状态
./submit_batch_manager.sh status

# 快速状态
python monitor_tasks.py --mode quick
```

### **3. 任务管理操作**

```bash
# 合并完成的结果
./submit_batch_manager.sh merge

# 重启任务管理（会继续未完成的batch）
./submit_batch_manager.sh restart
```

## 📊 **分批策略**

### **自动分批逻辑**
```
总数据: 英文50小时 + 中文50小时 ≈ 10,000-20,000个样本
分批: 每批500样本 = 20-40个batch
并发: 最多3个batch同时运行
时间: 每个batch约1-2小时
```

### **健壮性设计**
- **断点续传**: 每个batch独立，中断后只需重跑失败的batch
- **状态持久化**: 所有状态保存到文件，重启后自动恢复
- **自动重试**: 失败的batch最多重试3次
- **并发控制**: 避免资源争抢，提高成功率

## 📁 **输出结构**

```
/data/gpfs/projects/punim2341/haoguangzhou/emilia_training_data/
├── batch_0000/                    # 第1个batch的输出
│   ├── mels/
│   ├── emotion_features/
│   └── batch_status.json
├── batch_0001/                    # 第2个batch的输出
│   └── ...
├── final_mels/                    # 合并后的最终mel文件
├── final_emotion_features/        # 合并后的最终情感特征
├── final_reports/                 # 最终报告
├── batch_info/                    # batch管理信息
│   ├── dataset_analysis.json
│   └── batch_plans.json
└── task_manager_state.json        # 任务管理器状态
```

## 🔄 **典型工作流程**

### **第一次启动**
```bash
# 1. 启动批任务管理器
./submit_batch_manager.sh

# 2. 监控进度
./submit_batch_manager.sh monitor
```

### **任务中断后恢复**
```bash
# 1. 检查状态
./submit_batch_manager.sh status

# 2. 重启任务管理（自动继续未完成的batch）
./submit_batch_manager.sh restart

# 3. 继续监控
./submit_batch_manager.sh monitor
```

### **手动干预**
```bash
# 查看SLURM队列
squeue -u haoguangz

# 取消特定作业
scancel JOB_ID

# 重启任务管理器
./submit_batch_manager.sh restart

# 手动合并结果
./submit_batch_manager.sh merge
```

## 📈 **监控界面示例**

```
================================================================================
📊 Emilia训练数据生成 - 任务状态
================================================================================
🕐 最后更新: 2025-09-18T15:30:00
📦 总batch数: 25
✅ 已完成: 15
🔄 运行中: 3
❌ 失败: 2
⏳ 待处理: 5
📈 总进度: 60.0%

🔄 运行中的作业:
  Batch 16: Job 12345 - RUNNING (01:23:45)
  Batch 17: Job 12346 - RUNNING (00:45:12)
  Batch 18: Job 12347 - PENDING (00:00:00)

❌ 失败的batch:
  Batch 12: 2/3 次尝试
  Batch 14: 1/3 次尝试

📁 输出文件统计:
  总文件数: 15,230
================================================================================
```

## ⚡ **快速命令参考**

```bash
# 启动分批处理
./submit_batch_manager.sh

# 监控状态
./submit_batch_manager.sh monitor

# 快速查看进度
python monitor_tasks.py --mode quick

# 查看SLURM队列
squeue -u haoguangz

# 查看特定batch日志
tail -f /data/gpfs/projects/punim2341/haoguangzhou/logs/emilia-batch-0001-*.out

# 重启失败的batch
./submit_batch_manager.sh restart

# 合并最终结果
./submit_batch_manager.sh merge
```

## 🛠️ **故障排除**

### **常见问题**

1. **作业提交失败**
   ```bash
   # 检查SLURM资源
   sinfo -p gpu-a100-short
   
   # 检查磁盘空间
   df -h /data/gpfs/projects/punim2341/haoguangzhou/
   ```

2. **batch任务失败**
   ```bash
   # 查看失败日志
   tail -f /data/gpfs/projects/punim2341/haoguangzhou/logs/emilia-batch-*-*.err
   
   # 重启特定batch
   ./submit_batch_manager.sh restart
   ```

3. **任务管理器无响应**
   ```bash
   # 检查状态文件
   cat /data/gpfs/projects/punim2341/haoguangzhou/emilia_training_data/task_manager_state.json
   
   # 手动监控
   ./submit_batch_manager.sh monitor
   ```

## 🎉 **优势**

- **🔧 健壮性**: 单个batch失败不影响整体进度
- **⚡ 效率**: 并行处理，充分利用集群资源
- **🔄 可恢复**: 任何时候中断都可以继续
- **📊 透明**: 实时状态监控，进度一目了然
- **🎯 智能**: 自动重试、自动合并、自动恢复

现在你可以放心地处理大规模数据集，不用担心集群限制或任务中断！🚀
