#!/usr/bin/env python3
"""
训练数据集访问工具
提供便捷的API来访问和管理生成的训练数据
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json

class EmiliaTrainingDataset:
    """Emilia训练数据集访问器"""
    
    def __init__(self, dataset_path: str):
        """
        初始化数据集访问器
        
        Args:
            dataset_path: 训练数据集根目录路径
        """
        self.dataset_path = Path(dataset_path)
        self.metadata_path = self.dataset_path / "metadata"
        
        # 检查数据集完整性
        self._validate_dataset()
        
        # 加载元信息
        self.metadata_df = self._load_metadata()
        
        print(f"✅ 训练数据集加载完成")
        print(f"📁 数据集路径: {self.dataset_path}")
        print(f"📊 样本总数: {len(self.metadata_df)}")
        
        # 统计信息
        self._print_statistics()
    
    def _validate_dataset(self):
        """验证数据集完整性"""
        required_dirs = ['original_mels', 'neutral_mels', 'emotion_features', 'metadata']
        
        for dir_name in required_dirs:
            dir_path = self.dataset_path / dir_name
            if not dir_path.exists():
                raise FileNotFoundError(f"缺少必需目录: {dir_path}")
        
        # 检查CSV文件
        csv_file = self.metadata_path / "training_dataset_metadata.csv"
        if not csv_file.exists():
            raise FileNotFoundError(f"缺少元信息文件: {csv_file}")
    
    def _load_metadata(self) -> pd.DataFrame:
        """加载CSV元信息"""
        csv_file = self.metadata_path / "training_dataset_metadata.csv"
        return pd.read_csv(csv_file)
    
    def _print_statistics(self):
        """打印数据集统计信息"""
        # 语言分布
        lang_stats = self.metadata_df['language'].value_counts()
        print(f"📊 语言分布:")
        for lang, count in lang_stats.items():
            duration = self.metadata_df[self.metadata_df['language'] == lang]['duration_seconds'].sum()
            hours = duration / 3600
            print(f"  {lang.upper()}: {count} 样本, {hours:.1f} 小时")
        
        # 时长统计
        total_duration = self.metadata_df['duration_seconds'].sum() / 3600
        avg_duration = self.metadata_df['duration_seconds'].mean()
        print(f"📊 时长统计:")
        print(f"  总时长: {total_duration:.1f} 小时")
        print(f"  平均时长: {avg_duration:.1f} 秒")
    
    def get_sample_by_id(self, sample_id: str) -> Optional[Dict]:
        """根据sample_id获取样本信息"""
        sample_row = self.metadata_df[self.metadata_df['sample_id'] == sample_id]
        
        if sample_row.empty:
            return None
        
        return sample_row.iloc[0].to_dict()
    
    def load_sample_data(self, sample_id: str) -> Optional[Dict]:
        """加载样本的所有数据文件"""
        sample_info = self.get_sample_by_id(sample_id)
        if not sample_info:
            return None
        
        try:
            # 加载源mel
            original_mel_path = self.dataset_path / "original_mels" / sample_info['original_mel_file']
            original_mel_data = np.load(original_mel_path)
            
            # 加载中性mel
            neutral_mel_path = self.dataset_path / "neutral_mels" / sample_info['neutral_mel_file']
            neutral_mel_data = np.load(neutral_mel_path)
            
            # 加载情感特征
            emotion_path = self.dataset_path / "emotion_features" / sample_info['emotion_features_file']
            emotion_data = np.load(emotion_path)
            
            return {
                'sample_info': sample_info,
                'original_mel': original_mel_data['mel'],
                'neutral_mel': neutral_mel_data['mel'],
                'emotion_utterance': emotion_data['utterance'],
                'emotion_frame': emotion_data['frame'],
                'metadata': {
                    'original_mel_meta': dict(original_mel_data),
                    'neutral_mel_meta': dict(neutral_mel_data),
                    'emotion_meta': dict(emotion_data)
                }
            }
            
        except Exception as e:
            print(f"❌ 加载样本数据失败 {sample_id}: {e}")
            return None
    
    def get_samples_by_language(self, language: str) -> pd.DataFrame:
        """获取指定语言的所有样本"""
        return self.metadata_df[self.metadata_df['language'] == language.lower()]
    
    def get_samples_by_duration_range(self, min_duration: float, max_duration: float) -> pd.DataFrame:
        """获取指定时长范围的样本"""
        return self.metadata_df[
            (self.metadata_df['duration_seconds'] >= min_duration) &
            (self.metadata_df['duration_seconds'] <= max_duration)
        ]
    
    def create_train_val_split(self, val_ratio: float = 0.1, random_seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """创建训练/验证集分割"""
        # 按语言分别分割，确保语言平衡
        train_dfs = []
        val_dfs = []
        
        for language in ['en', 'zh']:
            lang_data = self.get_samples_by_language(language)
            
            # 随机分割
            lang_data_shuffled = lang_data.sample(frac=1, random_state=random_seed)
            val_size = int(len(lang_data_shuffled) * val_ratio)
            
            val_data = lang_data_shuffled[:val_size]
            train_data = lang_data_shuffled[val_size:]
            
            train_dfs.append(train_data)
            val_dfs.append(val_data)
        
        train_df = pd.concat(train_dfs, ignore_index=True)
        val_df = pd.concat(val_dfs, ignore_index=True)
        
        print(f"📊 数据集分割:")
        print(f"  训练集: {len(train_df)} 样本")
        print(f"  验证集: {len(val_df)} 样本")
        
        return train_df, val_df
    
    def export_split_csvs(self, output_dir: Optional[str] = None):
        """导出训练/验证集CSV文件"""
        if output_dir is None:
            output_dir = self.metadata_path
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        train_df, val_df = self.create_train_val_split()
        
        # 保存分割后的CSV
        train_df.to_csv(output_dir / "train_split.csv", index=False)
        val_df.to_csv(output_dir / "val_split.csv", index=False)
        
        print(f"✅ 分割CSV已保存:")
        print(f"  训练集: {output_dir}/train_split.csv")
        print(f"  验证集: {output_dir}/val_split.csv")
    
    def verify_data_integrity(self) -> bool:
        """验证数据完整性"""
        print("🔍 验证数据完整性...")
        
        missing_files = []
        corrupted_files = []
        
        for _, row in self.metadata_df.iterrows():
            sample_id = row['sample_id']
            
            # 检查文件是否存在
            files_to_check = [
                self.dataset_path / "original_mels" / row['original_mel_file'],
                self.dataset_path / "neutral_mels" / row['neutral_mel_file'],
                self.dataset_path / "emotion_features" / row['emotion_features_file']
            ]
            
            for file_path in files_to_check:
                if not file_path.exists():
                    missing_files.append(str(file_path))
                else:
                    # 尝试加载文件验证完整性
                    try:
                        data = np.load(file_path)
                        # 基本验证
                        if 'mel' in data and len(data['mel'].shape) != 2:
                            corrupted_files.append(str(file_path))
                    except:
                        corrupted_files.append(str(file_path))
        
        # 报告结果
        if not missing_files and not corrupted_files:
            print("✅ 数据完整性验证通过")
            return True
        else:
            if missing_files:
                print(f"❌ 缺失文件: {len(missing_files)} 个")
            if corrupted_files:
                print(f"❌ 损坏文件: {len(corrupted_files)} 个")
            return False


def main():
    """示例使用"""
    import argparse
    
    parser = argparse.ArgumentParser(description="训练数据集访问工具")
    parser.add_argument("--dataset_path", type=str, required=True,
                       help="训练数据集路径")
    parser.add_argument("--action", type=str, 
                       choices=['info', 'verify', 'split', 'sample'],
                       default='info',
                       help="执行动作")
    parser.add_argument("--sample_id", type=str, default=None,
                       help="样本ID（用于sample动作）")
    
    args = parser.parse_args()
    
    try:
        dataset = EmiliaTrainingDataset(args.dataset_path)
        
        if args.action == 'info':
            print("📊 数据集信息已显示")
            
        elif args.action == 'verify':
            success = dataset.verify_data_integrity()
            return 0 if success else 1
            
        elif args.action == 'split':
            dataset.export_split_csvs()
            
        elif args.action == 'sample':
            if not args.sample_id:
                print("❌ 请提供sample_id")
                return 1
            
            sample_data = dataset.load_sample_data(args.sample_id)
            if sample_data:
                print(f"✅ 样本 {args.sample_id} 数据:")
                print(f"  语言: {sample_data['sample_info']['language']}")
                print(f"  时长: {sample_data['sample_info']['duration_seconds']:.1f}s")
                print(f"  源mel形状: {sample_data['original_mel'].shape}")
                print(f"  中性mel形状: {sample_data['neutral_mel'].shape}")
                print(f"  情感特征形状: utterance={sample_data['emotion_utterance'].shape}, frame={sample_data['emotion_frame'].shape}")
            else:
                print(f"❌ 未找到样本: {args.sample_id}")
                return 1
        
        return 0
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())
