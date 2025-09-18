#!/usr/bin/env python3
"""
任务监控脚本
实时监控批处理任务的状态，提供友好的状态显示
"""

import json
import time
import subprocess
from pathlib import Path
import argparse
from datetime import datetime
import sys

def get_slurm_queue_status():
    """获取SLURM队列状态"""
    try:
        result = subprocess.run(
            ['squeue', '-u', 'haoguangz', '-h', '-o', '%i,%j,%T,%M,%R'],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode == 0:
            jobs = {}
            for line in result.stdout.strip().split('\n'):
                if line:
                    parts = line.split(',')
                    if len(parts) >= 5:
                        job_id, job_name, status, time_used, reason = parts
                        jobs[job_id] = {
                            'name': job_name,
                            'status': status,
                            'time_used': time_used,
                            'reason': reason
                        }
            return jobs
        
    except:
        pass
    
    return {}

def display_task_status(output_base_path: str):
    """显示任务状态"""
    output_dir = Path(output_base_path)
    task_state_file = output_dir / "task_manager_state.json"
    
    if not task_state_file.exists():
        print("❌ 未找到任务状态文件")
        print("💡 请先运行批任务管理器: ./submit_batch_manager.sh")
        return
    
    try:
        with open(task_state_file, 'r') as f:
            state = json.load(f)
    except:
        print("❌ 任务状态文件损坏")
        return
    
    # 获取SLURM队列状态
    slurm_jobs = get_slurm_queue_status()
    
    # 显示总体状态
    batch_plans = state.get('batch_plans', [])
    completed = state.get('completed_batches', [])
    running_jobs = state.get('running_jobs', {})
    failed = state.get('failed_batches', [])
    
    print("=" * 80)
    print("📊 Emilia训练数据生成 - 任务状态")
    print("=" * 80)
    print(f"🕐 最后更新: {state.get('last_update', 'Unknown')}")
    print(f"📦 总batch数: {len(batch_plans)}")
    print(f"✅ 已完成: {len(completed)}")
    print(f"🔄 运行中: {len(running_jobs)}")
    print(f"❌ 失败: {len(failed)}")
    print(f"⏳ 待处理: {len(batch_plans) - len(completed) - len(running_jobs) - len(failed)}")
    
    if len(batch_plans) > 0:
        progress = len(completed) / len(batch_plans) * 100
        print(f"📈 总进度: {progress:.1f}%")
    
    # 显示运行中的作业
    if running_jobs:
        print(f"\\n🔄 运行中的作业:")
        for batch_id, job_info in running_jobs.items():
            job_id = job_info['job_id']
            start_time = job_info.get('start_time', 'Unknown')
            
            # 检查SLURM状态
            slurm_status = "UNKNOWN"
            if job_id in slurm_jobs:
                slurm_status = slurm_jobs[job_id]['status']
                time_used = slurm_jobs[job_id]['time_used']
                print(f"  Batch {batch_id}: Job {job_id} - {slurm_status} ({time_used})")
            else:
                print(f"  Batch {batch_id}: Job {job_id} - 不在队列中")
    
    # 显示失败的batch
    if failed:
        print(f"\\n❌ 失败的batch:")
        for batch_id in failed:
            batch_plan = next((bp for bp in batch_plans if bp['batch_id'] == batch_id), None)
            if batch_plan:
                attempts = batch_plan.get('attempts', 0)
                max_attempts = batch_plan.get('max_attempts', 3)
                print(f"  Batch {batch_id}: {attempts}/{max_attempts} 次尝试")
    
    # 显示最近完成的batch
    if completed:
        recent_completed = completed[-5:]  # 显示最近5个
        print(f"\\n✅ 最近完成的batch:")
        for batch_id in recent_completed:
            print(f"  Batch {batch_id}")
    
    # 显示输出统计
    if completed:
        print(f"\\n📁 输出文件统计:")
        total_files = 0
        for batch_id in completed:
            batch_output_dir = output_dir / f"batch_{batch_id:04d}"
            if batch_output_dir.exists():
                mel_files = len(list((batch_output_dir / "mels").glob("*.npz")))
                emotion_files = len(list((batch_output_dir / "emotion_features").glob("*.npz")))
                total_files += mel_files + emotion_files
        
        print(f"  总文件数: {total_files}")
    
    print("=" * 80)

def monitor_loop(output_base_path: str, interval: int = 60):
    """监控循环"""
    print(f"👀 开始监控任务状态 (每 {interval} 秒更新)")
    print("按 Ctrl+C 退出监控")
    
    try:
        while True:
            print(f"\\n🔄 更新时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            display_task_status(output_base_path)
            
            print(f"\\n⏳ 等待 {interval} 秒...")
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print("\\n👋 监控已停止")

def quick_status(output_base_path: str):
    """快速状态显示"""
    output_dir = Path(output_base_path)
    task_state_file = output_dir / "task_manager_state.json"
    
    if not task_state_file.exists():
        print("❌ 未找到任务状态文件")
        return
    
    try:
        with open(task_state_file, 'r') as f:
            state = json.load(f)
        
        batch_plans = state.get('batch_plans', [])
        completed = len(state.get('completed_batches', []))
        running = len(state.get('running_jobs', {}))
        failed = len(state.get('failed_batches', []))
        
        if len(batch_plans) > 0:
            progress = completed / len(batch_plans) * 100
            print(f"📊 进度: {completed}/{len(batch_plans)} ({progress:.1f}%) | 运行中: {running} | 失败: {failed}")
        else:
            print("📊 还未开始任务")
            
    except:
        print("❌ 状态文件读取失败")

def main():
    parser = argparse.ArgumentParser(description="任务监控脚本")
    parser.add_argument("--output_path", type=str, 
                       default="/data/gpfs/projects/punim2341/haoguangzhou/emilia_training_data",
                       help="输出路径")
    parser.add_argument("--mode", type=str, choices=['status', 'monitor', 'quick'],
                       default='status', help="监控模式")
    parser.add_argument("--interval", type=int, default=60,
                       help="监控间隔（秒）")
    
    args = parser.parse_args()
    
    if args.mode == 'status':
        display_task_status(args.output_path)
    elif args.mode == 'monitor':
        monitor_loop(args.output_path, args.interval)
    elif args.mode == 'quick':
        quick_status(args.output_path)

if __name__ == "__main__":
    main()
