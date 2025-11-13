#!/usr/bin/env python3
"""
自动清理脚本：移除硬编码和幽灵功能

Usage:
    python cleanup_code.py --check     # 检查问题
    python cleanup_code.py --fix       # 自动修复
    python cleanup_code.py --report    # 生成报告
"""

import re
import argparse
from pathlib import Path
from typing import List, Tuple, Dict

class CodeCleaner:
    def __init__(self, project_root: str = "."):
        self.root = Path(project_root)
        self.issues = []
        
    def find_hardcoded_paths(self) -> List[Tuple[Path, int, str]]:
        """查找硬编码的路径"""
        pattern = r'/data/gpfs/projects/punim2341'
        issues = []
        
        for py_file in self.root.glob("*.py"):
            if py_file.name.startswith('.'):
                continue
                
            with open(py_file, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    if re.search(pattern, line):
                        issues.append((py_file, line_num, line.strip()))
        
        return issues
    
    def find_unused_parameters(self, file_path: Path) -> List[str]:
        """查找未使用的参数"""
        unused = []
        
        with open(file_path, 'r') as f:
            content = f.read()
        
        # 检查 use_coarse
        if 'use_coarse:' in content and content.count('use_coarse') <= 3:
            unused.append('use_coarse (defined but rarely used)')
        
        # 检查 use_tensorboard
        if 'use_tensorboard: bool = False' in content:
            unused.append('use_tensorboard (always False)')
        
        # 检查 num_coarse_layers
        if 'num_coarse_layers' in content and content.count('num_coarse_layers') <= 2:
            unused.append('num_coarse_layers (unused due to use_coarse=False)')
        
        return unused
    
    def check_ghost_functions(self) -> Dict[str, List[str]]:
        """检查幽灵功能"""
        ghosts = {}
        
        config_file = self.root / "config.py"
        if config_file.exists():
            with open(config_file, 'r') as f:
                content = f.read()
            
            ghosts['config.py'] = []
            
            # Coarse layers
            if 'use_coarse: bool = False' in content:
                ghosts['config.py'].append("Coarse quantization (always disabled)")
            
            # TensorBoard
            if 'use_tensorboard: bool = False' in content:
                ghosts['config.py'].append("TensorBoard (always disabled)")
            
            # Empty emotion_label_map
            if 'return {}  # 训练数据无标签' in content:
                ghosts['config.py'].append("emotion_label_map (always returns empty dict)")
        
        return ghosts
    
    def generate_cleanup_suggestions(self) -> str:
        """生成清理建议"""
        report = []
        report.append("=" * 60)
        report.append("代码清理建议")
        report.append("=" * 60)
        
        # 硬编码路径
        report.append("\n## 1. 硬编码路径")
        hardcoded = self.find_hardcoded_paths()
        if hardcoded:
            report.append(f"发现 {len(hardcoded)} 处硬编码路径：")
            for file, line_num, line in hardcoded[:5]:  # 只显示前5个
                report.append(f"  - {file.name}:{line_num} - {line[:60]}...")
            if len(hardcoded) > 5:
                report.append(f"  ... 还有 {len(hardcoded) - 5} 处")
        else:
            report.append("✅ 未发现硬编码路径")
        
        # 幽灵功能
        report.append("\n## 2. 幽灵功能")
        ghosts = self.check_ghost_functions()
        if ghosts:
            for file, issues in ghosts.items():
                if issues:
                    report.append(f"\n{file}:")
                    for issue in issues:
                        report.append(f"  ❌ {issue}")
        
        # 未使用参数
        report.append("\n## 3. 未使用的参数")
        config_file = self.root / "config.py"
        if config_file.exists():
            unused = self.find_unused_parameters(config_file)
            if unused:
                for param in unused:
                    report.append(f"  ❌ {param}")
            else:
                report.append("  ✅ 未发现未使用的参数")
        
        # 建议
        report.append("\n## 建议的清理步骤")
        report.append("1. 将硬编码路径移至 configs/default.yaml")
        report.append("2. 移除 use_coarse 相关代码")
        report.append("3. 移除 use_tensorboard 相关代码")
        report.append("4. 清理 emotion_label_map 空实现")
        report.append("5. 更新评估脚本使用配置系统")
        
        report.append("\n" + "=" * 60)
        
        return "\n".join(report)
    
    def run_check(self):
        """运行检查"""
        print(self.generate_cleanup_suggestions())
    
    def create_cleanup_config(self):
        """创建清理配置文件"""
        cleanup_yaml = """# 清理配置
# 这个文件列出了需要移除的硬编码和幽灵功能

hardcoded_paths:
  - pattern: '/data/gpfs/projects/punim2341'
    replace_with: '${DATA_ROOT}'
    files:
      - config.py
      - run_evaluation_*.py
      - extract_evaluation_features.py

ghost_parameters:
  config.py:
    - use_coarse
    - num_coarse_layers
    - coarse_codebook_size
    - use_tensorboard
    
  grouped_rvq.py:
    - coarse_* (if use_coarse is removed)

unused_methods:
  - DataConfig.emotion_label_map
"""
        
        output_file = self.root / "cleanup_config.yaml"
        with open(output_file, 'w') as f:
            f.write(cleanup_yaml)
        
        print(f"✅ 创建清理配置: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='代码清理工具')
    parser.add_argument('--check', action='store_true', help='检查问题')
    parser.add_argument('--report', action='store_true', help='生成详细报告')
    parser.add_argument('--create-config', action='store_true', help='创建清理配置')
    
    args = parser.parse_args()
    
    cleaner = CodeCleaner()
    
    if args.check or args.report:
        cleaner.run_check()
    
    if args.create_config:
        cleaner.create_cleanup_config()
    
    if not any([args.check, args.report, args.create_config]):
        parser.print_help()


if __name__ == "__main__":
    main()

