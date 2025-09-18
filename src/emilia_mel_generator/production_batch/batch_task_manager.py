#!/usr/bin/env python3
"""
批任务管理器 - 支持大规模数据集的分批处理
解决集群服务器无法一次性处理完整数据集的问题

核心功能：
1. 将大数据集分割成多个可管理的batch
2. 支持断点续传，任务中断后可继续
3. 并行提交多个batch任务
4. 自动合并最终结果
5. 健壮的错误处理和重试机制
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
import time
import math

class BatchTaskManager:
    """批任务管理器"""
    
    def __init__(self, 
                 dataset_path: str,
                 output_base_path: str,
                 cache_path: str,
                 samples_per_batch: int = 500,
                 max_concurrent_jobs: int = 3,
                 hours_per_lang: float = 50.0):
        """
        初始化批任务管理器
        
        Args:
            dataset_path: 数据集路径
            output_base_path: 输出基础路径
            cache_path: 缓存路径
            samples_per_batch: 每个batch的样本数
            max_concurrent_jobs: 最大并发作业数
            hours_per_lang: 每种语言的目标时长
        """
        self.dataset_path = Path(dataset_path)
        self.output_base_path = Path(output_base_path)
        self.cache_path = Path(cache_path)
        self.samples_per_batch = samples_per_batch
        self.max_concurrent_jobs = max_concurrent_jobs
        self.hours_per_lang = hours_per_lang
        
        # 任务管理
        self.task_state_file = self.output_base_path / "task_manager_state.json"
        self.batch_info_dir = self.output_base_path / "batch_info"
        self.batch_info_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建输出目录
        self.output_base_path.mkdir(parents=True, exist_ok=True)
        
        print(f"✅ 批任务管理器初始化完成")
        print(f"  数据集: {self.dataset_path}")
        print(f"  输出: {self.output_base_path}")
        print(f"  每批样本数: {self.samples_per_batch}")
        print(f"  最大并发: {self.max_concurrent_jobs}")
        print(f"  目标时长: 每种语言 {self.hours_per_lang} 小时")
    
    def analyze_dataset(self) -> Dict:
        """分析数据集，计算需要的batch数量"""
        print("📊 分析数据集...")
        
        try:
            from datasets import load_dataset, load_from_disk
            
            # 加载数据集
            if self.dataset_path.exists():
                dataset = load_from_disk(str(self.dataset_path))
            else:
                dataset = load_dataset("amphion/Emilia-Dataset", cache_dir=str(self.cache_path))
                dataset.save_to_disk(str(self.dataset_path))
            
            # 分析数据集
            total_samples = 0
            lang_stats = {'en': {'count': 0, 'duration': 0}, 'zh': {'count': 0, 'duration': 0}}
            
            for split_name in dataset.keys():
                for item in dataset[split_name]:
                    language = item.get('language', 'en').lower()
                    if language in ['en', 'zh']:
                        audio_data = item['audio']
                        duration = len(audio_data['array']) / audio_data['sampling_rate']
                        
                        lang_stats[language]['count'] += 1
                        lang_stats[language]['duration'] += duration
                        total_samples += 1
            
            # 计算需要的样本数
            target_seconds_per_lang = self.hours_per_lang * 3600
            needed_samples = {}
            
            for lang in ['en', 'zh']:
                current_duration = lang_stats[lang]['duration']
                if current_duration >= target_seconds_per_lang:
                    # 按比例计算需要的样本数
                    ratio = target_seconds_per_lang / current_duration
                    needed_samples[lang] = int(lang_stats[lang]['count'] * ratio)
                else:
                    needed_samples[lang] = lang_stats[lang]['count']
            
            total_needed = sum(needed_samples.values())
            num_batches = math.ceil(total_needed / self.samples_per_batch)
            
            analysis = {
                'total_available_samples': total_samples,
                'total_needed_samples': total_needed,
                'lang_stats': lang_stats,
                'needed_samples': needed_samples,
                'num_batches': num_batches,
                'samples_per_batch': self.samples_per_batch,
                'analysis_time': datetime.now().isoformat()
            }
            
            print(f"📊 数据集分析完成:")
            print(f"  可用样本: {total_samples}")
            print(f"  需要样本: {total_needed}")
            print(f"  计划batch数: {num_batches}")
            for lang, stats in lang_stats.items():
                hours = stats['duration'] / 3600
                print(f"  {lang.upper()}: {stats['count']} 样本, {hours:.1f} 小时")
            
            # 保存分析结果
            with open(self.batch_info_dir / "dataset_analysis.json", 'w') as f:
                json.dump(analysis, f, indent=2)
            
            return analysis
            
        except Exception as e:
            print(f"❌ 数据集分析失败: {e}")
            raise
    
    def create_batch_plans(self, analysis: Dict) -> List[Dict]:
        """创建batch执行计划"""
        print("📋 创建batch执行计划...")
        
        num_batches = analysis['num_batches']
        total_needed = analysis['total_needed_samples']
        
        batch_plans = []
        
        for batch_id in range(num_batches):
            start_idx = batch_id * self.samples_per_batch
            end_idx = min(start_idx + self.samples_per_batch, total_needed)
            
            batch_plan = {
                'batch_id': batch_id,
                'start_idx': start_idx,
                'end_idx': end_idx,
                'sample_count': end_idx - start_idx,
                'status': 'pending',
                'output_dir': str(self.output_base_path / f"batch_{batch_id:04d}"),
                'job_id': None,
                'start_time': None,
                'end_time': None,
                'attempts': 0,
                'max_attempts': 3
            }
            
            batch_plans.append(batch_plan)
        
        print(f"📋 创建了 {len(batch_plans)} 个batch计划")
        
        # 保存batch计划
        with open(self.batch_info_dir / "batch_plans.json", 'w') as f:
            json.dump(batch_plans, f, indent=2)
        
        return batch_plans
    
    def load_task_state(self) -> Dict:
        """加载任务状态"""
        if self.task_state_file.exists():
            try:
                with open(self.task_state_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        
        return {
            'batch_plans': [],
            'completed_batches': [],
            'failed_batches': [],
            'running_jobs': {},
            'last_update': datetime.now().isoformat()
        }
    
    def save_task_state(self, state: Dict):
        """保存任务状态"""
        state['last_update'] = datetime.now().isoformat()
        with open(self.task_state_file, 'w') as f:
            json.dump(state, f, indent=2)
    
    def submit_batch_job(self, batch_plan: Dict) -> Optional[str]:
        """提交单个batch作业"""
        batch_id = batch_plan['batch_id']
        output_dir = batch_plan['output_dir']
        
        # 创建batch专用的输出目录
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # 创建batch专用的SLURM脚本
        slurm_script = self._create_batch_slurm_script(batch_plan)
        
        try:
            # 提交作业
            result = subprocess.run(
                ['sbatch', slurm_script],
                capture_output=True,
                text=True,
                check=True
            )
            
            # 提取作业ID
            job_id = result.stdout.strip().split()[-1]
            
            print(f"✅ Batch {batch_id} 已提交，作业ID: {job_id}")
            
            # 清理临时脚本
            Path(slurm_script).unlink()
            
            return job_id
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Batch {batch_id} 提交失败: {e}")
            print(f"错误输出: {e.stderr}")
            return None
    
    def _create_batch_slurm_script(self, batch_plan: Dict) -> str:
        """为单个batch创建SLURM脚本"""
        batch_id = batch_plan['batch_id']
        output_dir = batch_plan['output_dir']
        
        # 创建临时SLURM脚本
        script_path = f"/tmp/emilia_batch_{batch_id}_{int(time.time())}.slurm"
        
        script_content = f'''#!/bin/bash
#SBATCH --partition=gpu-a100-short
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 04:00:00
#SBATCH -J emilia-batch-{batch_id:04d}
#SBATCH -o /data/gpfs/projects/punim2341/haoguangzhou/logs/emilia-batch-{batch_id:04d}-%j.out
#SBATCH -e /data/gpfs/projects/punim2341/haoguangzhou/logs/emilia-batch-{batch_id:04d}-%j.err
#SBATCH --mail-user=haoguangz@student.unimelb.edu.au
#SBATCH --mail-type=FAIL
#SBATCH -A punim2341

echo "🎯 Emilia Batch {batch_id:04d} 开始"
echo "作业ID: $SLURM_JOB_ID"
echo "样本范围: {batch_plan['start_idx']}-{batch_plan['end_idx']}"
echo "时间: $(date)"

# 设置环境
PROJECT_ROOT="/data/gpfs/projects/punim2341/haoguangzhou/MyAudioResearchModel"
cd $PROJECT_ROOT

source /data/gpfs/projects/punim2341/haoguangzhou/miniconda3/bin/activate
conda activate marm5

export PYTHONPATH=$PROJECT_ROOT/src:$PYTHONPATH

# 创建batch状态文件
echo '{{"batch_id": {batch_id}, "job_id": "'$SLURM_JOB_ID'", "status": "running", "start_time": "'$(date -Iseconds)'"}}' > "{output_dir}/batch_status.json"

# 运行batch处理
python src/emilia_mel_generator/production_batch/generate_training_data.py \\
    --dataset_path "{self.dataset_path}" \\
    --output_path "{output_dir}" \\
    --cache_path "{self.cache_path}" \\
    --device cuda \\
    --hours_per_lang {self.hours_per_lang} \\
    --batch_start_idx {batch_plan['start_idx']} \\
    --batch_end_idx {batch_plan['end_idx']} \\
    --batch_id {batch_id}

exit_code=$?

# 更新batch状态
if [ $exit_code -eq 0 ]; then
    echo '{{"batch_id": {batch_id}, "job_id": "'$SLURM_JOB_ID'", "status": "completed", "start_time": "'$(date -Iseconds)'", "end_time": "'$(date -Iseconds)'", "exit_code": '$exit_code'}}' > "{output_dir}/batch_status.json"
    echo "✅ Batch {batch_id:04d} 完成"
else
    echo '{{"batch_id": {batch_id}, "job_id": "'$SLURM_JOB_ID'", "status": "failed", "start_time": "'$(date -Iseconds)'", "end_time": "'$(date -Iseconds)'", "exit_code": '$exit_code'}}' > "{output_dir}/batch_status.json"
    echo "❌ Batch {batch_id:04d} 失败，退出代码: $exit_code"
fi

exit $exit_code
'''
        
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        return script_path
    
    def check_job_status(self, job_id: str) -> str:
        """检查SLURM作业状态"""
        try:
            result = subprocess.run(
                ['squeue', '-j', job_id, '-h', '-o', '%T'],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()  # PENDING, RUNNING, COMPLETED, etc.
            else:
                # 作业不在队列中，可能已完成或失败
                return 'NOT_FOUND'
                
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
            return 'UNKNOWN'
    
    def monitor_and_manage_jobs(self, state: Dict):
        """监控和管理正在运行的作业"""
        print("👀 监控正在运行的作业...")
        
        running_jobs = state.get('running_jobs', {})
        completed_this_round = []
        failed_this_round = []
        
        for batch_id, job_info in list(running_jobs.items()):\n            job_id = job_info['job_id']\n            job_status = self.check_job_status(job_id)\n            \n            if job_status in ['COMPLETED', 'NOT_FOUND']:\n                # 检查batch输出目录的状态文件\n                batch_output_dir = Path(job_info['output_dir'])\n                status_file = batch_output_dir / \"batch_status.json\"\n                \n                if status_file.exists():\n                    try:\n                        with open(status_file, 'r') as f:\n                            batch_status = json.load(f)\n                        \n                        if batch_status.get('status') == 'completed':\n                            completed_this_round.append(batch_id)\n                            state['completed_batches'].append(batch_id)\n                            print(f\"✅ Batch {batch_id} 完成\")\n                        else:\n                            failed_this_round.append(batch_id)\n                            state['failed_batches'].append(batch_id)\n                            print(f\"❌ Batch {batch_id} 失败\")\n                    except:\n                        failed_this_round.append(batch_id)\n                        state['failed_batches'].append(batch_id)\n                        print(f\"❌ Batch {batch_id} 状态文件损坏\")\n                else:\n                    failed_this_round.append(batch_id)\n                    state['failed_batches'].append(batch_id)\n                    print(f\"❌ Batch {batch_id} 没有状态文件\")\n                \n                # 从运行列表中移除\n                del running_jobs[batch_id]\n                \n            elif job_status in ['CANCELLED', 'FAILED', 'TIMEOUT']:\n                failed_this_round.append(batch_id)\n                state['failed_batches'].append(batch_id)\n                del running_jobs[batch_id]\n                print(f\"❌ Batch {batch_id} 作业失败: {job_status}\")\n                \n            elif job_status in ['PENDING', 'RUNNING']:\n                print(f\"🔄 Batch {batch_id} 运行中: {job_status}\")\n                \n            else:\n                print(f\"❓ Batch {batch_id} 状态未知: {job_status}\")\n        \n        return completed_this_round, failed_this_round\n    \n    def retry_failed_batches(self, state: Dict, batch_plans: List[Dict]) -> int:\n        \"\"\"重试失败的batch\"\"\"\n        failed_batches = state.get('failed_batches', [])\n        retried_count = 0\n        \n        for batch_id in failed_batches[:]:\n            batch_plan = next((bp for bp in batch_plans if bp['batch_id'] == batch_id), None)\n            if not batch_plan:\n                continue\n            \n            # 检查重试次数\n            if batch_plan['attempts'] >= batch_plan['max_attempts']:\n                print(f\"⚠️ Batch {batch_id} 已达到最大重试次数，跳过\")\n                continue\n            \n            # 检查并发限制\n            if len(state.get('running_jobs', {})) >= self.max_concurrent_jobs:\n                break\n            \n            print(f\"🔄 重试 Batch {batch_id} (第 {batch_plan['attempts'] + 1} 次尝试)\")\n            \n            job_id = self.submit_batch_job(batch_plan)\n            if job_id:\n                batch_plan['attempts'] += 1\n                state['running_jobs'][str(batch_id)] = {\n                    'job_id': job_id,\n                    'output_dir': batch_plan['output_dir'],\n                    'start_time': datetime.now().isoformat()\n                }\n                state['failed_batches'].remove(batch_id)\n                retried_count += 1\n        \n        return retried_count\n    \n    def submit_new_batches(self, state: Dict, batch_plans: List[Dict]) -> int:\n        \"\"\"提交新的batch作业\"\"\"\n        completed = set(state.get('completed_batches', []))\n        running = set(state.get('running_jobs', {}).keys())\n        failed = set(state.get('failed_batches', []))\n        \n        submitted_count = 0\n        \n        for batch_plan in batch_plans:\n            batch_id = batch_plan['batch_id']\n            \n            # 跳过已完成、运行中或失败的batch\n            if str(batch_id) in completed or str(batch_id) in running or batch_id in failed:\n                continue\n            \n            # 检查并发限制\n            if len(state.get('running_jobs', {})) >= self.max_concurrent_jobs:\n                break\n            \n            print(f\"🚀 提交 Batch {batch_id}\")\n            \n            job_id = self.submit_batch_job(batch_plan)\n            if job_id:\n                state.setdefault('running_jobs', {})[str(batch_id)] = {\n                    'job_id': job_id,\n                    'output_dir': batch_plan['output_dir'],\n                    'start_time': datetime.now().isoformat()\n                }\n                submitted_count += 1\n            else:\n                state.setdefault('failed_batches', []).append(batch_id)\n        \n        return submitted_count\n    \n    def merge_batch_results(self, state: Dict) -> bool:\n        \"\"\"合并所有完成的batch结果\"\"\"\n        print(\"🔗 合并batch结果...\")\n        \n        completed_batches = state.get('completed_batches', [])\n        if not completed_batches:\n            print(\"⚠️ 没有完成的batch需要合并\")\n            return False\n        \n        try:\n            # 创建最终输出目录\n            final_mels_dir = self.output_base_path / \"final_mels\"\n            final_emotions_dir = self.output_base_path / \"final_emotion_features\"\n            final_reports_dir = self.output_base_path / \"final_reports\"\n            \n            final_mels_dir.mkdir(exist_ok=True)\n            final_emotions_dir.mkdir(exist_ok=True)\n            final_reports_dir.mkdir(exist_ok=True)\n            \n            # 合并文件\n            total_merged = 0\n            merge_stats = {'en': 0, 'zh': 0}\n            \n            for batch_id in completed_batches:\n                batch_output_dir = self.output_base_path / f\"batch_{batch_id:04d}\"\n                \n                # 合并mel文件\n                batch_mels_dir = batch_output_dir / \"mels\"\n                if batch_mels_dir.exists():\n                    for mel_file in batch_mels_dir.glob(\"*.npz\"):\n                        target_file = final_mels_dir / mel_file.name\n                        if not target_file.exists():\n                            import shutil\n                            shutil.copy2(mel_file, target_file)\n                            total_merged += 1\n                            \n                            # 统计语言\n                            if '_en_' in mel_file.name:\n                                merge_stats['en'] += 1\n                            elif '_zh_' in mel_file.name:\n                                merge_stats['zh'] += 1\n                \n                # 合并情感特征文件\n                batch_emotions_dir = batch_output_dir / \"emotion_features\"\n                if batch_emotions_dir.exists():\n                    for emotion_file in batch_emotions_dir.glob(\"*.npz\"):\n                        target_file = final_emotions_dir / emotion_file.name\n                        if not target_file.exists():\n                            import shutil\n                            shutil.copy2(emotion_file, target_file)\n            \n            # 生成最终报告\n            final_report = {\n                'merge_info': {\n                    'completed_batches': len(completed_batches),\n                    'total_files_merged': total_merged,\n                    'language_stats': merge_stats,\n                    'merge_time': datetime.now().isoformat()\n                },\n                'batch_details': state\n            }\n            \n            with open(final_reports_dir / \"final_merge_report.json\", 'w') as f:\n                json.dump(final_report, f, indent=2)\n            \n            print(f\"✅ 合并完成: {total_merged} 个文件\")\n            print(f\"📊 语言分布: EN={merge_stats['en']}, ZH={merge_stats['zh']}\")\n            \n            return True\n            \n        except Exception as e:\n            print(f\"❌ 合并失败: {e}\")\n            return False\n    \n    def run_batch_management(self):\n        \"\"\"运行批任务管理\"\"\"\n        print(\"=\" * 80)\n        print(\"🚀 启动批任务管理系统\")\n        print(\"=\" * 80)\n        \n        try:\n            # 1. 分析数据集\n            analysis = self.analyze_dataset()\n            \n            # 2. 创建batch计划\n            batch_plans = self.create_batch_plans(analysis)\n            \n            # 3. 加载任务状态\n            state = self.load_task_state()\n            state['batch_plans'] = batch_plans\n            \n            print(f\"\\n📋 任务概览:\")\n            print(f\"  总batch数: {len(batch_plans)}\")\n            print(f\"  已完成: {len(state.get('completed_batches', []))}\")\n            print(f\"  运行中: {len(state.get('running_jobs', {}))}\")\n            print(f\"  失败: {len(state.get('failed_batches', []))}\")\n            \n            # 4. 主管理循环\n            while True:\n                # 监控当前作业\n                completed, failed = self.monitor_and_manage_jobs(state)\n                \n                # 重试失败的batch\n                retried = self.retry_failed_batches(state, batch_plans)\n                \n                # 提交新的batch\n                submitted = self.submit_new_batches(state, batch_plans)\n                \n                # 保存状态\n                self.save_task_state(state)\n                \n                # 检查是否全部完成\n                total_completed = len(state.get('completed_batches', []))\n                total_batches = len(batch_plans)\n                \n                print(f\"\\n📊 当前状态: {total_completed}/{total_batches} 完成\")\n                \n                if total_completed == total_batches:\n                    print(\"🎉 所有batch已完成！\")\n                    break\n                \n                if submitted == 0 and retried == 0 and len(state.get('running_jobs', {})) == 0:\n                    print(\"⚠️ 没有更多作业可提交，可能需要手动干预\")\n                    break\n                \n                # 等待一段时间再检查\n                print(f\"⏳ 等待 60 秒后继续监控...\")\n                time.sleep(60)\n            \n            # 5. 合并结果\n            if state.get('completed_batches'):\n                self.merge_batch_results(state)\n            \n            print(\"=\" * 80)\n            print(\"✅ 批任务管理完成！\")\n            print(\"=\" * 80)\n            \n            return True\n            \n        except KeyboardInterrupt:\n            print(\"\\n⚠️ 用户中断任务管理\")\n            self.save_task_state(state)\n            return False\n        except Exception as e:\n            print(f\"❌ 批任务管理失败: {e}\")\n            return False


def parse_arguments():\n    \"\"\"解析命令行参数\"\"\"\n    parser = argparse.ArgumentParser(description=\"批任务管理器\")\n    \n    parser.add_argument(\"--dataset_path\", type=str, required=True,\n                       help=\"数据集路径\")\n    parser.add_argument(\"--output_base_path\", type=str, required=True,\n                       help=\"输出基础路径\")\n    parser.add_argument(\"--cache_path\", type=str, required=True,\n                       help=\"缓存路径\")\n    parser.add_argument(\"--samples_per_batch\", type=int, default=500,\n                       help=\"每个batch的样本数\")\n    parser.add_argument(\"--max_concurrent_jobs\", type=int, default=3,\n                       help=\"最大并发作业数\")\n    parser.add_argument(\"--hours_per_lang\", type=float, default=50.0,\n                       help=\"每种语言的目标时长\")\n    parser.add_argument(\"--action\", type=str, \n                       choices=['start', 'monitor', 'merge', 'status'],\n                       default='start',\n                       help=\"执行动作\")\n    \n    return parser.parse_args()\n\n\ndef main():\n    \"\"\"主函数\"\"\"\n    args = parse_arguments()\n    \n    manager = BatchTaskManager(\n        dataset_path=args.dataset_path,\n        output_base_path=args.output_base_path,\n        cache_path=args.cache_path,\n        samples_per_batch=args.samples_per_batch,\n        max_concurrent_jobs=args.max_concurrent_jobs,\n        hours_per_lang=args.hours_per_lang\n    )\n    \n    if args.action == 'start':\n        success = manager.run_batch_management()\n        return 0 if success else 1\n    elif args.action == 'monitor':\n        # 只监控，不提交新作业\n        state = manager.load_task_state()\n        manager.monitor_and_manage_jobs(state)\n        manager.save_task_state(state)\n        return 0\n    elif args.action == 'merge':\n        # 只合并结果\n        state = manager.load_task_state()\n        success = manager.merge_batch_results(state)\n        return 0 if success else 1\n    elif args.action == 'status':\n        # 显示状态\n        state = manager.load_task_state()\n        print(json.dumps(state, indent=2))\n        return 0\n\n\nif __name__ == \"__main__\":\n    sys.exit(main())
