"""
Emilia Mel Generator - 主执行脚本
精简版，只保留核心功能
"""

import argparse
import logging
import sys
from pathlib import Path

try:
    from .corrected_data_generator import CorrectedDataGenerator
    from .config import load_config, save_config_template, get_config_summary
    from .utils import run_full_validation
except ImportError:
    from corrected_data_generator import CorrectedDataGenerator
    from config import load_config, save_config_template, get_config_summary
    from utils import run_full_validation

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="Emilia 增强数据生成工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 生成增强数据集 (推荐)
  python main.py --generate --target-hours 50 --k-variants 5
  
  # 验证配置
  python main.py --validate
  
  # 显示配置信息
  python main.py --show-config
        """
    )
    
    # 主要操作
    parser.add_argument("--generate", action="store_true",
                       help="生成增强数据集")
    parser.add_argument("--test-local", action="store_true",
                       help="本地小规模测试 (10条音频)")
    parser.add_argument("--validate", action="store_true",
                       help="验证配置")
    parser.add_argument("--show-config", action="store_true",
                       help="显示配置摘要")
    parser.add_argument("--create-config", type=str, metavar="PATH",
                       help="创建配置模板")
    
    # 参数选项
    parser.add_argument("--output-dir", type=str, 
                       default="data/user_specified_dataset",
                       help="输出目录")
    parser.add_argument("--target-hours", type=int, default=50,
                       help="每种语言的目标小时数")
    parser.add_argument("--k-variants", type=int, default=1,
                       help="每个原始音频的中性变体数量K")
    parser.add_argument("--num-test-samples", type=int, default=10,
                       help="本地测试样本数量")
    
    # 日志选项
    parser.add_argument("--verbose", action="store_true",
                       help="详细输出")
    parser.add_argument("--quiet", action="store_true",
                       help="静默模式")
    
    args = parser.parse_args()
    
    # 设置日志级别
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    elif args.quiet:
        logging.getLogger().setLevel(logging.WARNING)
    
    # 执行操作
    success = True
    
    if args.generate:
        print("🚀 生成增强数据集...")
        try:
            generator = CorrectedDataGenerator(
                output_dir=args.output_dir,
                target_hours_per_lang=args.target_hours,
                k_neutral_variants=args.k_variants
            )
            
            result = generator.run()
            
            if result["success"]:
                print("✅ 数据集生成成功！")
                print(f"📊 数据格式: {result['data_format']['tuple_format']}")
                print(f"📁 CSV文件: {result['csv_file_path']}")
                print(f"🔄 增强倍数: {result['data_format']['augmentation_factor']}")
            else:
                print(f"❌ 生成失败: {result['error']}")
                success = False
                
        except Exception as e:
            logger.error(f"生成失败: {e}")
            success = False
            
    elif args.test_local:
        print("🧪 开始本地小规模测试...")
        try:
            from .test_local_small import LocalTestGenerator
            
            tester = LocalTestGenerator(
                output_dir=f"{args.output_dir}_test",
                num_test_samples=args.num_test_samples,
                k_variants=args.k_variants
            )
            
            result = tester.run_test()
            
            if result.get("success", True):
                print("✅ 本地测试成功！")
                print("🎵 请检查验证音频:")
                print(f"  {args.output_dir}_test/verification_audio/")
                print("📋 确认中性mel重建音频保持音色但去除情感")
            else:
                print(f"❌ 测试失败: {result['error']}")
                success = False
                
        except Exception as e:
            logger.error(f"本地测试失败: {e}")
            success = False
            
    elif args.validate:
        print("🔍 验证配置...")
        success = run_full_validation(load_config())
        
    elif args.show_config:
        print("📋 配置摘要:")
        summary = get_config_summary()
        
        print("\n🎵 Mel 配置:")
        for key, value in summary["unified_mel_config"].items():
            print(f"  {key}: {value}")
        
        print("\n📊 衍生属性:")
        for key, value in summary["derived_properties"].items():
            print(f"  {key}: {value}")
            
    elif args.create_config:
        print(f"📝 创建配置模板: {args.create_config}")
        save_config_template(args.create_config)
        
    else:
        parser.print_help()
        success = False
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()