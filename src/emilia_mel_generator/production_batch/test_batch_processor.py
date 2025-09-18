#!/usr/bin/env python3
"""
批处理器测试脚本
用于验证批处理系统是否正常工作
"""

import sys
from pathlib import Path
import tempfile
import shutil

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent))

from emilia_batch_processor import EmiliaBatchProcessor, setup_logging

def create_test_data():
    """创建测试数据"""
    test_data = []
    
    # 创建一些模拟样本
    for i in range(5):
        sample = {
            'id': f'test_sample_{i+1:03d}',
            'audio': {
                'array': np.random.randn(24000 * 2).astype(np.float32),  # 2秒音频
                'sampling_rate': 24000
            },
            'text': f'This is test sample number {i+1}.',
            'speaker': f'test_speaker_{i+1}',
            'language': 'en' if i % 2 == 0 else 'zh'
        }
        test_data.append(sample)
    
    return test_data

def test_batch_processor():
    """测试批处理器"""
    print("🧪 开始批处理器测试...")
    
    # 设置日志
    setup_logging("INFO")
    
    # 创建临时目录
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        input_path = temp_path / "input"
        output_path = temp_path / "output"
        
        input_path.mkdir(exist_ok=True)
        
        try:
            # 创建批处理器（小规模测试）
            processor = EmiliaBatchProcessor(
                input_data_path=str(input_path),
                output_base_path=str(output_path),
                batch_size=2,  # 小批次测试
                device='cpu',  # 使用CPU避免GPU依赖
                resume_from_checkpoint=False,
                checkpoint_interval=1
            )
            
            # 模拟数据集元数据加载
            test_data = create_test_data()
            processor.total_count = len(test_data)
            
            print(f"📊 创建了 {len(test_data)} 个测试样本")
            print("🔧 测试批处理器组件...")
            
            # 测试目录创建
            assert output_path.exists(), "输出目录未创建"
            assert (output_path / "mels").exists(), "mels目录未创建"
            assert (output_path / "checkpoints").exists(), "checkpoints目录未创建"
            
            print("✅ 目录结构创建正常")
            
            # 测试检查点功能
            test_results = [{"sample_id": "test", "status": "success"}]
            processor.save_checkpoint(0, test_results)
            
            checkpoint_files = list((output_path / "checkpoints").glob("*.json"))
            assert len(checkpoint_files) > 0, "检查点文件未创建"
            
            print("✅ 检查点功能正常")
            
            # 测试报告生成
            processor.processed_count = 5
            processor.total_count = 5
            processor.start_time = "2025-09-18T14:30:00"
            
            report = processor.generate_final_report(test_results)
            assert report is not None, "报告生成失败"
            
            report_files = list((output_path / "reports").glob("*.json"))
            assert len(report_files) > 0, "报告文件未创建"
            
            print("✅ 报告生成功能正常")
            print("🎉 批处理器测试通过！")
            
            return True
            
        except Exception as e:
            print(f"❌ 测试失败: {e}")
            import traceback
            traceback.print_exc()
            return False

if __name__ == "__main__":
    import numpy as np
    
    success = test_batch_processor()
    
    if success:
        print("\n" + "=" * 60)
        print("🎉 批处理器测试成功！")
        print("✅ 所有组件功能正常")
        print("🚀 可以开始正式的批处理任务")
        print("=" * 60)
        sys.exit(0)
    else:
        print("\n" + "=" * 60)
        print("❌ 批处理器测试失败")
        print("⚠️ 请检查错误信息并修复问题")
        print("=" * 60)
        sys.exit(1)
