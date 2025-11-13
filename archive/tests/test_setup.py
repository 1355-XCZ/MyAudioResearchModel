"""
项目设置测试脚本
验证所有依赖和配置是否正确
"""

import sys
from pathlib import Path

# 添加路径
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

def test_imports():
    """测试所有导入"""
    print("测试模块导入...")
    
    try:
        # 配置
        from config import get_default_config
        print("  ✓ config")
        
        # 核心模型
        from grouped_rvq import GroupedResidualVQ
        print("  ✓ grouped_rvq")
        
        from entropy_model import AutoregressiveEntropyModel, create_entropy_model
        print("  ✓ entropy_model")
        
        from rate_controller import RateController
        print("  ✓ rate_controller")
        
        from data_loader import create_dataloaders
        print("  ✓ data_loader")
        
        # 数据集
        from datasets import EmotionDataset, IEMOCAPDataset, RAVDESSDataset, ESDDataset
        print("  ✓ datasets")
        
        # 评估（手动导入，避免与Amphion.evaluation冲突）
        import sys
        import importlib.util
        eval_dir = Path(__file__).parent / 'evaluation'
        
        # emotion_classifier
        spec = importlib.util.spec_from_file_location("emotion_classifier_local", eval_dir / "emotion_classifier.py")
        emotion_classifier_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(emotion_classifier_module)
        EmotionClassifierV2 = emotion_classifier_module.EmotionClassifierV2
        
        # method_rate_sweep
        spec = importlib.util.spec_from_file_location("method_rate_sweep_local", eval_dir / "method_rate_sweep.py")
        rate_sweep_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(rate_sweep_module)
        
        # method_layer_sweep
        spec = importlib.util.spec_from_file_location("method_layer_sweep_local", eval_dir / "method_layer_sweep.py")
        layer_sweep_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(layer_sweep_module)
        
        # analyzer
        spec = importlib.util.spec_from_file_location("analyzer_local", eval_dir / "analyzer.py")
        analyzer_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(analyzer_module)
        
        print("  ✓ evaluation (手动加载)")
        
        print("\n✅ 所有模块导入成功！")
        return True
        
    except Exception as e:
        print(f"\n❌ 导入失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_config():
    """测试配置"""
    print("\n测试配置...")
    
    try:
        from config import get_default_config
        
        config = get_default_config()
        
        print(f"  数据路径: {config['data'].train_data_path}")
        print(f"  支持语言: {config['data'].supported_languages}")
        print(f"  归一化参数: {config['data'].mean_std_path}")
        print(f"  RVQ分组数: {config['grouped_rvq'].num_groups}")
        print(f"  RVQ层数: {config['grouped_rvq'].num_fine_layers}")
        print(f"  熵模型: d_model={config['entropy_model'].d_model}, layers={config['entropy_model'].num_layers}")
        print(f"  码率扫描点数: {len(config['evaluation'].rate_sweep_rates_bpf)}")
        print(f"  层数扫描点数: {len(config['evaluation'].layer_sweep_layers)}")
        
        print("\n✅ 配置加载成功！")
        return True
        
    except Exception as e:
        print(f"\n❌ 配置测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_data_paths():
    """测试数据路径"""
    print("\n测试数据路径...")
    
    paths_to_check = {
        '训练数据（100h）': '/data/gpfs/projects/punim2341/haoguangzhou/data/emilia_vevo_training_50h',
        '训练数据-英文': '/data/gpfs/projects/punim2341/haoguangzhou/data/emilia_vevo_training_50h/EN',
        '训练数据-中文': '/data/gpfs/projects/punim2341/haoguangzhou/data/emilia_vevo_training_50h/ZH',
    }
    
    all_exist = True
    for name, path in paths_to_check.items():
        exists = Path(path).exists()
        symbol = "✓" if exists else "✗"
        print(f"  {symbol} {name}: {path}")
        if not exists:
            all_exist = False
    
    if all_exist:
        print("\n✅ 所有数据路径存在！")
    else:
        print("\n⚠️ 部分数据路径不存在")
    
    return all_exist


def test_directory_structure():
    """测试目录结构"""
    print("\n测试目录结构...")
    
    base_dir = Path(__file__).parent
    
    required_files = [
        'config.py',
        'compute_normalization.py',
        'grouped_rvq.py',
        'entropy_model.py',
        'rate_controller.py',
        'data_loader.py',
        'bucket_sampler.py',
        'train_rvq.py',
        'train_entropy.py',
        'run_evaluation.py',
        'README.md',
    ]
    
    required_dirs = [
        'datasets',
        'evaluation',
        'checkpoints',
    ]
    
    all_exist = True
    
    for file in required_files:
        exists = (base_dir / file).exists()
        symbol = "✓" if exists else "✗"
        print(f"  {symbol} {file}")
        if not exists:
            all_exist = False
    
    for dir in required_dirs:
        exists = (base_dir / dir).exists()
        symbol = "✓" if exists else "✗"
        print(f"  {symbol} {dir}/")
        if not exists:
            all_exist = False
    
    if all_exist:
        print("\n✅ 目录结构完整！")
    else:
        print("\n⚠️ 部分文件/目录缺失")
    
    return all_exist


def main():
    """运行所有测试"""
    print("="*80)
    print("情感RVQ信息瓶颈项目 - 设置测试")
    print("="*80)
    
    results = []
    
    # 测试目录结构
    results.append(("目录结构", test_directory_structure()))
    
    # 测试导入
    results.append(("模块导入", test_imports()))
    
    # 测试配置
    results.append(("配置加载", test_config()))
    
    # 测试数据路径
    results.append(("数据路径", test_data_paths()))
    
    # 总结
    print("\n" + "="*80)
    print("测试总结")
    print("="*80)
    
    for name, result in results:
        symbol = "✅" if result else "❌"
        print(f"{symbol} {name}")
    
    all_passed = all(r[1] for r in results)
    
    if all_passed:
        print("\n✅ 所有测试通过！项目设置正确。")
    else:
        print("\n⚠️ 部分测试未通过，请检查上述问题。")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

