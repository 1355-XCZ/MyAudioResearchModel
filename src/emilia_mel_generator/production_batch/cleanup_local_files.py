#!/usr/bin/env python3
"""
本地文件清理脚本
用于清理测试生成的大文件，为git提交做准备
"""

import os
import shutil
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def cleanup_local_files():
    """清理本地测试文件"""
    
    # 项目根目录
    project_root = Path(__file__).parent.parent.parent.parent
    
    logger.info(f"🧹 开始清理本地测试文件...")
    logger.info(f"📁 项目根目录: {project_root}")
    
    # 要清理的目录和文件模式
    cleanup_patterns = [
        # emilia_mel_generator目录下的测试输出
        "src/emilia_mel_generator/test_output_*",
        "src/emilia_mel_generator/emilia_cache",
        "src/emilia_mel_generator/final_vevo_test_output",
        "src/emilia_mel_generator/vevo_emilia_integration",
        "src/emilia_mel_generator/temp_*",
        
        # 音频文件
        "src/emilia_mel_generator/*.wav",
        "src/emilia_mel_generator/*.ogg",
        "src/emilia_mel_generator/*.mp3",
        
        # 数据文件
        "src/emilia_mel_generator/*.npy",
        "src/emilia_mel_generator/*.npz",
        "src/emilia_mel_generator/*.pkl",
        "src/emilia_mel_generator/*.tar",
        
        # 项目根目录的测试文件
        "test_audio*.wav",
        "test_audio*.ogg",
        "outputs",
        "exp",
        
        # Amphion缓存
        "src/Amphion/ckpts",
        "src/Amphion/logs",
        "src/Amphion/exp_output",
        "src/Amphion/temp",
        
        # VQ-VAE缓存
        "src/VQ-VAE/*/ckpt",
        "src/VQ-VAE/*/logs",
        "src/VQ-VAE/*/output",
        
        # FlowSE静态文件
        "src/FlowSE-*/static",
        "src/FlowSE-*/datalist/*.wav",
        "src/FlowSE-*/datalist/*.json",
        
        # Python缓存
        "**/__pycache__",
        "**/*.pyc",
        "**/*.pyo",
    ]
    
    total_freed = 0
    cleaned_count = 0
    
    for pattern in cleanup_patterns:
        pattern_path = project_root / pattern
        
        # 处理通配符模式
        if '*' in pattern:
            parent_dir = project_root
            parts = pattern.split('/')
            
            for part in parts[:-1]:
                if '*' not in part:
                    parent_dir = parent_dir / part
            
            if parent_dir.exists():
                import glob
                matches = glob.glob(str(project_root / pattern))
                
                for match in matches:
                    match_path = Path(match)
                    if match_path.exists():
                        size = get_dir_size(match_path) if match_path.is_dir() else match_path.stat().st_size
                        
                        if match_path.is_dir():\n                            shutil.rmtree(match_path)\n                            logger.info(f\"🗑️  删除目录: {match_path.relative_to(project_root)} ({size/1024/1024:.1f}MB)\")\n                        else:\n                            match_path.unlink()\n                            logger.info(f\"🗑️  删除文件: {match_path.relative_to(project_root)} ({size/1024/1024:.1f}MB)\")\n                        \n                        total_freed += size\n                        cleaned_count += 1\n        else:\n            # 直接路径\n            if pattern_path.exists():\n                size = get_dir_size(pattern_path) if pattern_path.is_dir() else pattern_path.stat().st_size\n                \n                if pattern_path.is_dir():\n                    shutil.rmtree(pattern_path)\n                    logger.info(f\"🗑️  删除目录: {pattern_path.relative_to(project_root)} ({size/1024/1024:.1f}MB)\")\n                else:\n                    pattern_path.unlink()\n                    logger.info(f\"🗑️  删除文件: {pattern_path.relative_to(project_root)} ({size/1024/1024:.1f}MB)\")\n                \n                total_freed += size\n                cleaned_count += 1\n    \n    logger.info(\"=====================================\")\n    logger.info(f\"✅ 清理完成！\")\n    logger.info(f\"📊 清理统计:\")\n    logger.info(f\"  删除项目: {cleaned_count}\")\n    logger.info(f\"  释放空间: {total_freed/1024/1024/1024:.2f}GB\")\n    logger.info(\"=====================================\")\n    \n    return cleaned_count, total_freed\n\ndef get_dir_size(path: Path) -> int:\n    \"\"\"获取目录大小\"\"\"\n    total = 0\n    try:\n        for entry in path.rglob('*'):\n            if entry.is_file():\n                total += entry.stat().st_size\n    except (OSError, PermissionError):\n        pass\n    return total\n\ndef main():\n    \"\"\"主函数\"\"\"\n    print(\"=====================================\")\n    print(\"🧹 本地测试文件清理脚本\")\n    print(\"=====================================\")\n    print(\"⚠️  警告: 这将删除所有测试生成的文件！\")\n    print(\"包括: 音频文件、模型权重、缓存、测试输出等\")\n    print(\"=====================================\")\n    \n    # 确认清理\n    try:\n        confirm = input(\"确认要清理这些文件吗？(y/N): \")\n        if confirm.lower() != 'y':\n            print(\"❌ 清理已取消\")\n            return 1\n    except KeyboardInterrupt:\n        print(\"\\n❌ 清理已取消\")\n        return 1\n    \n    try:\n        cleaned_count, total_freed = cleanup_local_files()\n        \n        if cleaned_count > 0:\n            print(f\"\\n🎉 清理成功！释放了 {total_freed/1024/1024/1024:.2f}GB 空间\")\n            print(\"\\n📋 后续步骤:\")\n            print(\"1. git add .\")\n            print(\"2. git commit -m 'Clean up test files and add production batch system'\")\n            print(\"3. git push\")\n        else:\n            print(\"\\n✅ 没有找到需要清理的文件\")\n        \n        return 0\n        \n    except Exception as e:\n        logger.error(f\"❌ 清理失败: {e}\")\n        return 1\n\nif __name__ == \"__main__\":\n    import sys\n    sys.exit(main())
