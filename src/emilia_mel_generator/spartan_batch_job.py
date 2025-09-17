"""
Spartan 集群批处理脚本
用于在墨尔本大学HPC集群上运行大规模数据提取
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import torch

# 添加项目路径
sys.path.append('/data/gpfs/projects/punim2341/haoguangzhou/MyAudioResearchModel/src')

from emilia_mel_generator import CorrectedDataGenerator

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_environment():
    """设置Spartan环境"""
    logger.info("设置Spartan环境...")
    
    # 检查GPU可用性
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        current_device = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(current_device)
        logger.info(f"GPU可用: {gpu_count}个设备, 当前: {gpu_name}")
    else:
        logger.warning("GPU不可用，将使用CPU")
    
    # 设置环境变量
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 使用第一个GPU
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # 避免tokenizer警告
    
    # 创建输出目录
    output_base = Path("/data/gpfs/projects/punim2341/haoguangzhou/emilia_dataset")
    output_base.mkdir(parents=True, exist_ok=True)
    
    return output_base


def run_batch_processing(batch_id: int, 
                        total_batches: int,
                        output_base: Path,
                        target_hours_per_lang: int = 50,
                        k_variants: int = 1) -> bool:
    """
    运行批处理
    
    Args:
        batch_id: 批次ID (0-based)
        total_batches: 总批次数
        output_base: 输出基础目录
        target_hours_per_lang: 每语言目标小时数
        k_variants: K值
        
    Returns:
        是否成功
    """
    logger.info(f"开始批次 {batch_id+1}/{total_batches}...")
    
    try:
        # 计算这个批次的目标小时数
        hours_per_batch = target_hours_per_lang / total_batches
        
        # 创建批次特定的输出目录
        batch_output_dir = output_base / f"batch_{batch_id:02d}"
        
        # 创建数据生成器
        generator = CorrectedDataGenerator(
            output_dir=str(batch_output_dir),
            target_hours_per_lang=int(hours_per_batch),
            k_neutral_variants=k_variants
        )
        
        # 运行生成
        result = generator.run()
        
        if result["success"]:
            logger.info(f"✅ 批次 {batch_id+1} 完成!")
            logger.info(f"📊 生成统计: {result['generation_stats']}")
            return True
        else:
            logger.error(f"❌ 批次 {batch_id+1} 失败: {result['error']}")
            return False
            
    except Exception as e:
        logger.error(f"批次 {batch_id+1} 处理失败: {e}")
        return False


def merge_batch_results(output_base: Path, total_batches: int) -> bool:
    """
    合并所有批次的结果
    
    Args:
        output_base: 输出基础目录
        total_batches: 总批次数
        
    Returns:
        是否成功
    """
    logger.info("合并批次结果...")
    
    try:
        # 创建合并目录
        merged_dir = output_base / "merged_dataset"
        merged_dir.mkdir(exist_ok=True)
        
        # 合并每种语言的数据
        for lang in ["EN", "ZH"]:
            lang_merged_dir = merged_dir / lang
            lang_merged_dir.mkdir(exist_ok=True)
            
            file_count = 0
            
            # 遍历所有批次
            for batch_id in range(total_batches):
                batch_dir = output_base / f"batch_{batch_id:02d}" / lang
                
                if batch_dir.exists():
                    # 复制文件到合并目录
                    for file_path in batch_dir.glob("*"):
                        if file_path.is_file():
                            # 重命名文件避免冲突
                            new_name = f"batch{batch_id:02d}_{file_path.name}"
                            new_path = lang_merged_dir / new_name
                            
                            # 复制文件
                            import shutil
                            shutil.copy2(file_path, new_path)
                            file_count += 1
            
            logger.info(f"{lang}: 合并了 {file_count} 个文件")
        
        # 创建合并后的CSV文件列表
        # TODO: 实现CSV合并逻辑
        
        logger.info(f"✅ 批次结果合并完成: {merged_dir}")
        return True
        
    except Exception as e:
        logger.error(f"合并失败: {e}")
        return False


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Spartan集群数据处理脚本")
    
    parser.add_argument("--batch-id", type=int, required=True,
                       help="批次ID (0-based)")
    parser.add_argument("--total-batches", type=int, default=10,
                       help="总批次数")
    parser.add_argument("--target-hours", type=int, default=50,
                       help="每语言目标小时数")
    parser.add_argument("--k-variants", type=int, default=1,
                       help="K值 (中性变体数)")
    parser.add_argument("--merge-only", action="store_true",
                       help="只执行合并操作")
    
    args = parser.parse_args()
    
    # 设置环境
    output_base = setup_environment()
    
    if args.merge_only:
        # 只执行合并
        success = merge_batch_results(output_base, args.total_batches)
    else:
        # 执行批处理
        success = run_batch_processing(
            batch_id=args.batch_id,
            total_batches=args.total_batches,
            output_base=output_base,
            target_hours_per_lang=args.target_hours,
            k_variants=args.k_variants
        )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
