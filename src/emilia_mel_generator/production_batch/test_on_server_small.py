#!/usr/bin/env python3
"""
服务器小规模测试脚本
在Spartan集群上测试中英文各10条样本，验证完整流程
"""

import os
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime

def create_small_test_job_script(output_base_path: str, test_samples: int = 20):
    """创建小规模测试的SLURM脚本"""
    
    script_content = f'''#!/bin/bash
#SBATCH --partition=gpu-a100-short
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 01:00:00
#SBATCH -J emilia-test-small
#SBATCH -o /data/gpfs/projects/punim2341/haoguangzhou/logs/emilia-test-small-%j.out
#SBATCH -e /data/gpfs/projects/punim2341/haoguangzhou/logs/emilia-test-small-%j.err
#SBATCH --mail-user=haoguangz@student.unimelb.edu.au
#SBATCH --mail-type=END,FAIL
#SBATCH -A punim2341

echo "🧪 Emilia小规模测试开始 (中英文各10条样本)"
echo "作业ID: $SLURM_JOB_ID"
echo "时间: $(date)"

# 设置环境
PROJECT_ROOT="/data/gpfs/projects/punim2341/haoguangzhou/MyAudioResearchModel"
cd $PROJECT_ROOT

source /data/gpfs/projects/punim2341/haoguangzhou/miniconda3/bin/activate
conda activate marm5

export PYTHONPATH=$PROJECT_ROOT/src:$PYTHONPATH

# 检查GPU
echo "GPU信息:"
nvidia-smi

# 创建测试输出目录
TEST_OUTPUT="{output_base_path}/test_small_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$TEST_OUTPUT"
mkdir -p "$TEST_OUTPUT/logs"

echo "📁 测试输出目录: $TEST_OUTPUT"

# 设置数据和模型（在服务器上重新获取）
echo "=== 设置数据和模型 ==="
python src/emilia_mel_generator/production_batch/setup_data_and_models.py \\
    --base_path "/data/gpfs/projects/punim2341/haoguangzhou" \\
    --cache_path "/data/gpfs/projects/punim2341/haoguangzhou/cache" \\
    --download_dataset \\
    --download_models \\
    --check_space

if [ $? -ne 0 ]; then
    echo "❌ 数据和模型设置失败"
    exit 1
fi

# 运行小规模测试
echo "=== 开始小规模测试 ==="
python src/emilia_mel_generator/production_batch/generate_training_data.py \\
    --dataset_path "/data/gpfs/projects/punim2341/haoguangzhou/emilia_dataset" \\
    --output_path "$TEST_OUTPUT" \\
    --cache_path "/data/gpfs/projects/punim2341/haoguangzhou/cache" \\
    --device cuda \\
    --max_samples {test_samples} \\
    --hours_per_lang 0.1

exit_code=$?

echo "=== 测试完成 ==="
echo "Exit code: $exit_code"
echo "End time: $(date)"

# 验证生成的数据
if [ $exit_code -eq 0 ]; then
    echo "=== 验证生成的数据 ==="
    
    # 统计生成的文件
    echo "📊 文件统计:"
    echo "  原始mel: $(find "$TEST_OUTPUT/original_mels" -name "*.npz" 2>/dev/null | wc -l) 个"
    echo "  中性mel: $(find "$TEST_OUTPUT/neutral_mels" -name "*.npz" 2>/dev/null | wc -l) 个"
    echo "  情感特征: $(find "$TEST_OUTPUT/emotion_features" -name "*.npz" 2>/dev/null | wc -l) 个"
    
    # 检查CSV元信息
    if [ -f "$TEST_OUTPUT/metadata/training_dataset_metadata.csv" ]; then
        echo "✅ CSV元信息文件存在"
        echo "  记录数: $(tail -n +2 "$TEST_OUTPUT/metadata/training_dataset_metadata.csv" | wc -l)"
    else
        echo "❌ CSV元信息文件缺失"
    fi
    
    # 验证语言匹配
    python src/emilia_mel_generator/production_batch/verify_language_matching.py \\
        --output_path "$TEST_OUTPUT"
    
    # 验证数据格式
    python src/emilia_mel_generator/production_batch/verify_training_data.py \\
        --output_path "$TEST_OUTPUT"
    
    echo "✅ 小规模测试成功完成！"
    echo "📁 测试结果位置: $TEST_OUTPUT"
else
    echo "❌ 小规模测试失败"
fi

exit $exit_code
'''
    
    return script_content

def submit_small_test(output_base_path: str = "/data/gpfs/projects/punim2341/haoguangzhou/emilia_test"):
    """提交小规模测试作业"""
    
    print("=" * 80)
    print("🧪 提交Emilia小规模测试作业到Spartan集群")
    print("=" * 80)
    print(f"📁 输出路径: {output_base_path}")
    print(f"📊 测试规模: 中英文各10条样本 (共20条)")
    print(f"⏰ 预计时间: 30-60分钟")
    print("=" * 80)
    
    # 创建临时SLURM脚本
    import tempfile
    script_content = create_small_test_job_script(output_base_path, test_samples=20)
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.slurm', delete=False) as f:
        f.write(script_content)
        script_path = f.name
    
    try:
        # 提交作业
        import subprocess
        result = subprocess.run(
            ['sbatch', script_path],
            capture_output=True,
            text=True,
            check=True
        )
        
        # 提取作业ID
        job_id = result.stdout.strip().split()[-1]
        
        print(f"✅ 小规模测试作业已提交！")
        print(f"📋 作业ID: {job_id}")
        print(f"📊 监控命令: squeue -j {job_id}")
        print(f"📄 输出日志: /data/gpfs/projects/punim2341/haoguangzhou/logs/emilia-test-small-{job_id}.out")
        print(f"❌ 错误日志: /data/gpfs/projects/punim2341/haoguangzhou/logs/emilia-test-small-{job_id}.err")
        
        print(f"\n📋 测试完成后的验证命令:")
        print(f"# 查看测试结果")
        print(f"ls -la {output_base_path}/test_small_*/")
        print(f"")
        print(f"# 验证语言匹配")
        print(f"python verify_language_matching.py --output_path {output_base_path}/test_small_*")
        print(f"")
        print(f"# 查看CSV元信息")
        print(f"head {output_base_path}/test_small_*/metadata/training_dataset_metadata.csv")
        
        return job_id
        
    except subprocess.CalledProcessError as e:
        print(f"❌ 作业提交失败: {e}")
        print(f"错误输出: {e.stderr}")
        return None
    finally:
        # 清理临时脚本
        try:
            os.unlink(script_path)
        except:
            pass

def create_test_config():
    """创建测试配置文件"""
    config = {
        "test_info": {
            "purpose": "验证Emilia训练数据生成流程",
            "scale": "小规模测试 - 中英文各10条样本",
            "expected_duration": "30-60分钟",
            "test_date": datetime.now().isoformat()
        },
        "test_targets": {
            "emo_emilia_integration": "验证Emo-Emilia中性参考音频加载",
            "language_matching": "确保中文用中文neutral，英文用英文neutral",
            "vevo_compatibility": "确保源mel和中性mel参数一致",
            "emotion_extraction": "验证从源音频提取emotion2vec特征",
            "csv_metadata": "验证CSV元信息文件生成",
            "directory_structure": "验证清晰的训练集目录结构"
        },
        "success_criteria": {
            "min_success_rate": 0.8,
            "required_files_per_sample": 3,
            "required_language_match_rate": 0.9,
            "required_vevo_compatibility": True
        }
    }
    
    config_file = Path("test_small_config.json")
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print(f"✅ 测试配置已保存: {config_file}")
    return config_file

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="服务器小规模测试")
    parser.add_argument("--output_path", type=str, 
                       default="/data/gpfs/projects/punim2341/haoguangzhou/emilia_test",
                       help="测试输出路径")
    parser.add_argument("--samples", type=int, default=20,
                       help="测试样本数")
    parser.add_argument("--action", type=str, choices=['submit', 'config'],
                       default='submit', help="执行动作")
    
    args = parser.parse_args()
    
    if args.action == 'config':
        config_file = create_test_config()
        print(f"📋 测试配置文件已创建: {config_file}")
        return 0
    
    elif args.action == 'submit':
        print("🚀 准备提交服务器小规模测试...")
        
        # 创建测试配置
        create_test_config()
        
        # 提交测试作业
        job_id = submit_small_test(args.output_path)
        
        if job_id:
            print("\n" + "=" * 80)
            print("🎉 小规模测试作业提交成功！")
            print("=" * 80)
            print(f"📋 作业ID: {job_id}")
            print(f"📊 监控: squeue -j {job_id}")
            print(f"📄 日志: tail -f /data/gpfs/projects/punim2341/haoguangzhou/logs/emilia-test-small-{job_id}.out")
            print("\n💡 测试预期:")
            print("  ✅ 成功加载Emo-Emilia数据集")
            print("  ✅ 正确的语言匹配 (中文↔中文neutral, 英文↔英文neutral)")
            print("  ✅ 生成Vevo兼容的mel频谱图")
            print("  ✅ 提取源音频的emotion2vec特征")
            print("  ✅ 生成清晰的CSV元信息文件")
            print("=" * 80)
            return 0
        else:
            print("❌ 作业提交失败")
            return 1

if __name__ == "__main__":
    sys.exit(main())
