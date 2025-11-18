#!/usr/bin/env python3
"""
测试所有依赖是否正确安装

用于验证环境配置是否完整
"""

import sys
import importlib

# 定义所有需要的依赖包
REQUIRED_PACKAGES = {
    # 核心深度学习
    'torch': 'PyTorch (深度学习框架)',
    'numpy': 'NumPy (数值计算)',
    
    # 音频和模型
    'funasr': 'FunASR (Emotion2Vec)',
    'modelscope': 'ModelScope (模型下载)',
    'torchaudio': 'TorchAudio (音频处理)',
    
    # 机器学习
    'sklearn': 'scikit-learn (评估指标)',
    'scipy': 'SciPy (统计分析)',
    
    # RVQ
    'vector_quantize_pytorch': 'vector-quantize-pytorch (VQ实现)',
    
    # 数据处理
    'tqdm': 'tqdm (进度条)',
    'yaml': 'PyYAML (配置文件)',
    
    # 可视化
    'matplotlib': 'Matplotlib (绘图)',
    'seaborn': 'Seaborn (统计图)',
    'pandas': 'Pandas (数据处理)',
    
    # Transformer相关
    'einops': 'einops (张量操作)',
    'transformers': 'Transformers (预训练模型)',
}


def check_package(package_name, description):
    """检查单个包是否可导入"""
    try:
        module = importlib.import_module(package_name)
        version = getattr(module, '__version__', 'unknown')
        print(f"✅ {description:40s} v{version}")
        return True
    except ImportError as e:
        print(f"❌ {description:40s} NOT FOUND")
        print(f"   错误: {e}")
        return False


def check_cuda():
    """检查CUDA是否可用"""
    try:
        import torch
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            cuda_version = torch.version.cuda
            print(f"✅ CUDA可用: {device_name} (CUDA {cuda_version})")
            return True
        else:
            print(f"⚠️  CUDA不可用 (将使用CPU)")
            return False
    except Exception as e:
        print(f"❌ CUDA检查失败: {e}")
        return False


def check_project_structure():
    """检查项目结构"""
    from pathlib import Path
    
    print("\n" + "="*80)
    print("检查项目结构")
    print("="*80)
    
    required_files = [
        'config.py',
        'grouped_rvq.py',
        'entropy_model.py',
        'rate_controller.py',
        'run_evaluation.py',
        'reproduce_experiments.py',
        'prepare_evaluation_subset.py',
        'checkpoints/grouped_rvq_best.pt',
        'checkpoints/entropy_model_best.pt',
    ]
    
    required_dirs = [
        'datasets',
        'evaluation',
        'scripts',
    ]
    
    all_ok = True
    
    for file in required_files:
        if Path(file).exists():
            size = Path(file).stat().st_size
            print(f"✅ {file:45s} ({size:,} bytes)")
        else:
            print(f"❌ {file:45s} NOT FOUND")
            all_ok = False
    
    for dir_name in required_dirs:
        if Path(dir_name).is_dir():
            files = len(list(Path(dir_name).glob('*.py')))
            print(f"✅ {dir_name + '/':<45s} ({files} Python files)")
        else:
            print(f"❌ {dir_name + '/':<45s} NOT FOUND")
            all_ok = False
    
    return all_ok


def main():
    print("="*80)
    print("依赖检查工具")
    print("="*80)
    print()
    
    # 检查Python版本
    python_version = sys.version.split()[0]
    print(f"Python版本: {python_version}")
    
    if sys.version_info < (3, 8):
        print("❌ Python版本过低，需要 >= 3.8")
        sys.exit(1)
    else:
        print("✅ Python版本符合要求")
    
    print()
    print("="*80)
    print("检查依赖包")
    print("="*80)
    
    # 检查所有包
    failed_packages = []
    for package, description in REQUIRED_PACKAGES.items():
        if not check_package(package, description):
            failed_packages.append(package)
    
    print()
    print("="*80)
    print("检查GPU支持")
    print("="*80)
    check_cuda()
    
    print()
    structure_ok = check_project_structure()
    
    # 总结
    print()
    print("="*80)
    print("检查总结")
    print("="*80)
    
    if failed_packages:
        print(f"❌ {len(failed_packages)}个包缺失:")
        for pkg in failed_packages:
            print(f"   - {pkg}")
        print()
        print("请运行以下命令安装缺失的包:")
        print("   pip install -r requirements.txt")
        sys.exit(1)
    else:
        print("✅ 所有依赖包已安装")
    
    if not structure_ok:
        print("⚠️  项目结构不完整，请检查缺失的文件")
        sys.exit(1)
    else:
        print("✅ 项目结构完整")
    
    print()
    print("🎉 环境配置完成，可以运行实验！")
    print()
    print("下一步:")
    print("  1. 准备数据: python3 prepare_evaluation_subset.py")
    print("  2. 运行实验: python3 reproduce_experiments.py --mode all")
    print("  或一键运行: bash run_reproduce.sh")


if __name__ == '__main__':
    main()

