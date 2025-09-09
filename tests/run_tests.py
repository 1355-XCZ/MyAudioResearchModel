"""
测试运行器
提供统一的测试入口和测试套件管理
"""
import os
import sys

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def setup_test_environment():
    """设置测试环境"""
    print("🔧 设置测试环境...")
    
    # 创建测试所需的目录
    test_dirs = [
        "data/test",
        "data/train/audio", 
        "data/test/esd/audio",
        "tests/outputs"
    ]
    
    for dir_path in test_dirs:
        os.makedirs(dir_path, exist_ok=True)
        print(f"   ✅ 创建目录: {dir_path}")
    
    # 检查配置文件
    if os.path.exists('config/base_config.yaml'):
        print("   ✅ 配置文件存在")
    else:
        print("   ❌ 配置文件不存在")
        return False
    
    return True


def check_dependencies():
    """检查依赖安装状态"""
    print("\n🔍 检查依赖安装状态...")
    
    required_packages = [
        ('librosa', '音频处理'),
        ('soundfile', '音频文件读写'),
        ('numpy', '数值计算'),
        ('torch', '深度学习框架'),
        ('whisper', 'Whisper语音识别'),
        ('pypinyin', '中文音素转换'),
        ('yaml', '配置文件解析')
    ]
    
    optional_packages = [
        ('modelscope', 'Emotion2Vec支持'),
        ('funasr', 'Emotion2Vec备选'),
        ('transformers', 'HuggingFace模型')
    ]
    
    missing_required = []
    missing_optional = []
    
    # 检查必需依赖
    for package, description in required_packages:
        try:
            __import__(package)
            print(f"   ✅ {package} - {description}")
        except ImportError:
            print(f"   ❌ {package} - {description} (缺失)")
            missing_required.append(package)
    
    # 检查可选依赖
    for package, description in optional_packages:
        try:
            __import__(package)
            print(f"   ✅ {package} - {description}")
        except ImportError:
            print(f"   ⚠️ {package} - {description} (可选，未安装)")
            missing_optional.append(package)
    
    if missing_required:
        print(f"\n❌ 缺少必需依赖: {missing_required}")
        print("请运行: pip install -r requirements.txt")
        return False
    
    if missing_optional:
        print(f"\n⚠️ 缺少可选依赖: {missing_optional}")
        print("某些功能可能无法使用")
    
    return True


def run_preprocessing_tests():
    """运行预处理测试"""
    print("\n" + "="*60)
    print("🧪 运行预处理模块测试")
    print("="*60)
    
    try:
        from tests.test_data_processing import run_all_preprocessing_tests
        run_all_preprocessing_tests()
        return True
    except Exception as e:
        print(f"❌ 预处理测试失败: {e}")
        return False


def run_model_tests():
    """运行模型测试"""
    print("\n" + "="*60)
    print("🧪 运行模型模块测试")
    print("="*60)
    
    try:
        from tests.test_models import run_all_model_tests
        run_all_model_tests()
        return True
    except Exception as e:
        print(f"❌ 模型测试失败: {e}")
        return False


def run_pipeline_tests():
    """运行流水线测试"""
    print("\n" + "="*60)
    print("🧪 运行流水线测试")
    print("="*60)
    
    try:
        from tests.test_pipeline import run_all_pipeline_tests
        run_all_pipeline_tests()
        return True
    except Exception as e:
        print(f"❌ 流水线测试失败: {e}")
        return False


def run_quick_test():
    """快速测试：只测试核心功能"""
    print("\n🚀 快速测试模式")
    print("="*60)
    
    try:
        from tests.test_data_processing import test_audio_loading, test_complete_preprocessing
        from tests.test_pipeline import test_pipeline_creation
        
        quick_tests = [
            ("音频加载", test_audio_loading),
            ("流水线创建", test_pipeline_creation),
            ("完整预处理", test_complete_preprocessing)
        ]
        
        results = {}
        for test_name, test_func in quick_tests:
            print(f"\n--- {test_name} ---")
            results[test_name] = test_func()
        
        # 汇总快速测试结果
        success_count = sum(results.values())
        total_count = len(results)
        
        print(f"\n🏆 快速测试结果: {success_count}/{total_count} 通过")
        
        return success_count == total_count
        
    except Exception as e:
        print(f"❌ 快速测试失败: {e}")
        return False


def main():
    """主测试函数"""
    print("🧪 情感音频建模系统 - 测试套件")
    print("="*60)
    
    # 设置环境
    if not setup_test_environment():
        print("❌ 测试环境设置失败")
        return
    
    # 检查依赖
    if not check_dependencies():
        print("❌ 依赖检查失败")
        return
    
    print("\n请选择测试模式:")
    print("1. 快速测试 (推荐)")
    print("2. 预处理模块测试")
    print("3. 模型模块测试")
    print("4. 流水线测试")
    print("5. 完整测试套件")
    
    choice = input("\n请输入选择 (1-5): ").strip()
    
    if choice == '1':
        run_quick_test()
    elif choice == '2':
        run_preprocessing_tests()
    elif choice == '3':
        run_model_tests()
    elif choice == '4':
        run_pipeline_tests()
    elif choice == '5':
        print("\n🚀 运行完整测试套件...")
        run_preprocessing_tests()
        run_model_tests()
        run_pipeline_tests()
    else:
        print("❌ 无效选择")
        return
    
    print("\n🏁 测试完成！")


if __name__ == "__main__":
    main()
