"""
结果分析器
生成可视化图表和统计报告
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


class ResultAnalyzer:
    """结果分析器"""
    
    def __init__(self, output_dir: str):
        """
        Args:
            output_dir: 输出目录
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置绘图样式
        sns.set_style('whitegrid')
        plt.rcParams['figure.dpi'] = 300
        plt.rcParams['font.size'] = 10
    
    def plot_rate_vs_accuracy(self, results_dict: Dict, save_name: str = 'rate_vs_accuracy.png'):
        """
        绘制准确率 vs 码率曲线（分情感）
        
        Args:
            results_dict: 多个数据集的结果字典
            save_name: 保存文件名
        """
        for dataset_name, results in results_dict.items():
            # 提取码率点数据并排序
            rate_data = []
            for key, rate_point in results['rate_points'].items():
                if 'target_rate_bpf' in rate_point and len(rate_point.get('predictions', [])) > 0:
                    target_rate = rate_point['target_rate_bpf']
                    # 跳过inf（无量化baseline）
                    if target_rate != float('inf'):
                        rate_data.append({
                            'rate_bpf': target_rate,  # 使用目标码率，不是平均码率
                            'predictions': rate_point['predictions'],
                            'ground_truths': rate_point['ground_truths']
                        })
            
            # 按码率排序
            rate_data.sort(key=lambda x: x['rate_bpf'])
            
            if not rate_data:
                continue
            
            # 计算每个情感的准确率
            emotion_mapping = results.get('emotion_mapping', {})
            all_emotions = set()
            for gt in rate_data[0]['ground_truths']:
                all_emotions.add(gt)
            all_emotions = sorted(all_emotions)
            
            # 创建图表
            fig, ax = plt.subplots(figsize=(12, 7))
            
            for emotion in all_emotions:
                rates = []
                accuracies = []
                
                for rd in rate_data:
                    # 计算该情感的准确率
                    emotion_preds = [p for p, g in zip(rd['predictions'], rd['ground_truths']) if g == emotion]
                    emotion_gts = [g for g in rd['ground_truths'] if g == emotion]
                    
                    if len(emotion_gts) > 0:
                        correct = sum([1 for p, g in zip(emotion_preds, emotion_gts) if p == g])
                        acc = correct / len(emotion_gts)
                        rates.append(rd['rate_bpf'])
                        accuracies.append(acc)
                
                if rates:
                    ax.plot(rates, accuracies, marker='o', label=emotion, linewidth=2, markersize=6)
            
            ax.set_xlabel('Rate (BPF)', fontsize=13, fontweight='bold')
            ax.set_ylabel('Accuracy', fontsize=13, fontweight='bold')
            ax.set_title(f'{dataset_name}: Accuracy vs Rate (Per-Emotion)', fontsize=15, fontweight='bold')
            ax.legend(loc='best', fontsize=11)
            ax.grid(True, alpha=0.3)
            ax.set_ylim([0, 1.05])
            
            save_path = self.output_dir / f'{dataset_name}_rate_vs_accuracy.png'
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"✅ 保存图表: {save_path}")
    
    def plot_rate_vs_confidence(self, results_dict: Dict, save_name: str = 'rate_vs_confidence.png'):
        """
        绘制置信度 vs 码率曲线
        
        Args:
            results_dict: 多个数据集的结果字典
            save_name: 保存文件名
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for dataset_name, results in results_dict.items():
            rates = []
            confidences = []
            
            for key, rate_point in results['rate_points'].items():
                if 'avg_rate_bpf' in rate_point and 'avg_confidence' in rate_point:
                    rates.append(rate_point['avg_rate_bpf'])
                    confidences.append(rate_point['avg_confidence'])
            
            if rates:
                ax.plot(rates, confidences, marker='s', label=dataset_name, linewidth=2)
        
        ax.set_xlabel('Rate (bits/frame)', fontsize=12)
        ax.set_ylabel('Confidence', fontsize=12)
        ax.set_title('Confidence vs Rate', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        save_path = self.output_dir / save_name
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        
        logger.info(f"✅ 保存图表: {save_path}")
    
    def plot_layer_vs_accuracy(self, results_dict: Dict, save_name: str = 'layer_vs_accuracy.png'):
        """
        绘制准确率 vs 层数曲线
        
        Args:
            results_dict: 多个数据集的结果字典
            save_name: 保存文件名
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for dataset_name, results in results_dict.items():
            # 提取层数点数据并排序
            layer_data = []
            for key, layer_point in results['layer_points'].items():
                if 'num_layers' in layer_point and len(layer_point.get('predictions', [])) > 0:
                    layer_data.append({
                        'num_layers': layer_point['num_layers'],
                        'predictions': layer_point['predictions'],
                        'ground_truths': layer_point['ground_truths']
                    })
            
            # 按层数排序
            layer_data.sort(key=lambda x: x['num_layers'])
            
            if not layer_data:
                continue
            
            # 提取所有情感类别
            all_emotions = set()
            for gt in layer_data[0]['ground_truths']:
                all_emotions.add(gt)
            all_emotions = sorted(all_emotions)
            
            # 创建图表
            fig, ax = plt.subplots(figsize=(12, 7))
            
            for emotion in all_emotions:
                layers = []
                accuracies = []
                
                for ld in layer_data:
                    # 计算该情感的准确率
                    emotion_preds = [p for p, g in zip(ld['predictions'], ld['ground_truths']) if g == emotion]
                    emotion_gts = [g for g in ld['ground_truths'] if g == emotion]
                    
                    if len(emotion_gts) > 0:
                        correct = sum([1 for p, g in zip(emotion_preds, emotion_gts) if p == g])
                        acc = correct / len(emotion_gts)
                        layers.append(ld['num_layers'])
                        accuracies.append(acc)
                
                if layers:
                    ax.plot(layers, accuracies, marker='s', label=emotion, linewidth=2, markersize=6)
            
            ax.set_xlabel('Number of Layers (12 Groups × M Layers/Group)', fontsize=13, fontweight='bold')
            ax.set_ylabel('Accuracy', fontsize=13, fontweight='bold')
            ax.set_title(f'{dataset_name}: Accuracy vs Layers (Per-Emotion)', fontsize=15, fontweight='bold')
            ax.legend(loc='best', fontsize=11)
            ax.grid(True, alpha=0.3)
            ax.set_ylim([0, 1.05])
            
            save_path = self.output_dir / f'{dataset_name}_layer_vs_accuracy.png'
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"✅ 保存图表: {save_path}")
    
    def plot_combined_comparison(self, rate_results: Dict, layer_results: Dict, 
                                save_name: str = 'combined_comparison.png'):
        """
        绘制码率方法和层数方法的对比图
        
        Args:
            rate_results: 码率扫描结果
            layer_results: 层数扫描结果
            save_name: 保存文件名
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # 左图：码率方法
        for dataset_name, results in rate_results.items():
            rates = []
            accuracies = []
            
            for key, rate_point in results['rate_points'].items():
                if 'avg_rate_bpf' in rate_point and 'accuracy' in rate_point:
                    rates.append(rate_point['avg_rate_bpf'])
                    accuracies.append(rate_point['accuracy'])
            
            if rates:
                ax1.plot(rates, accuracies, marker='o', label=dataset_name, linewidth=2)
        
        ax1.set_xlabel('Rate (bits/frame)', fontsize=12)
        ax1.set_ylabel('Accuracy', fontsize=12)
        ax1.set_title('Method 1: Rate Sweep', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 右图：层数方法
        for dataset_name, results in layer_results.items():
            layers = []
            accuracies = []
            
            for key, layer_point in results['layer_points'].items():
                if 'num_layers' in layer_point and 'accuracy' in layer_point:
                    layers.append(layer_point['num_layers'])
                    accuracies.append(layer_point['accuracy'])
            
            if layers:
                sorted_indices = np.argsort(layers)
                layers = [layers[i] for i in sorted_indices]
                accuracies = [accuracies[i] for i in sorted_indices]
                
                ax2.plot(layers, accuracies, marker='s', label=dataset_name, linewidth=2)
        
        ax2.set_xlabel('Number of Layers', fontsize=12)
        ax2.set_ylabel('Accuracy', fontsize=12)
        ax2.set_title('Method 2: Layer Sweep', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        save_path = self.output_dir / save_name
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        
        logger.info(f"✅ 保存图表: {save_path}")
    
    def generate_summary_report(self, rate_results: Dict, layer_results: Dict, 
                               save_name: str = 'summary_report.txt'):
        """
        生成文本摘要报告
        
        Args:
            rate_results: 码率扫描结果
            layer_results: 层数扫描结果
            save_name: 保存文件名
        """
        report_lines = []
        report_lines.append("="*80)
        report_lines.append("情感RVQ信息瓶颈实验 - 评估摘要报告")
        report_lines.append("="*80)
        report_lines.append("")
        
        # 码率扫描结果
        report_lines.append("方法1: 码率扫描")
        report_lines.append("-"*80)
        
        for dataset_name, results in rate_results.items():
            report_lines.append(f"\n数据集: {dataset_name}")
            report_lines.append(f"  情感映射: {results['emotion_mapping']}")
            report_lines.append(f"\n  码率点 | 准确率 | 置信度")
            report_lines.append(f"  " + "-"*40)
            
            for key, rate_point in sorted(results['rate_points'].items()):
                if 'accuracy' in rate_point:
                    rate_bpf = rate_point.get('avg_rate_bpf', 0)
                    acc = rate_point['accuracy']
                    conf = rate_point.get('avg_confidence', 0)
                    report_lines.append(f"  {rate_bpf:7.2f} | {acc:6.2%} | {conf:6.4f}")
        
        # 层数扫描结果
        report_lines.append("\n\n方法2: 层数扫描")
        report_lines.append("-"*80)
        
        for dataset_name, results in layer_results.items():
            report_lines.append(f"\n数据集: {dataset_name}")
            report_lines.append(f"\n  层数 | 准确率 | 置信度")
            report_lines.append(f"  " + "-"*30)
            
            # 按层数排序
            layer_points_sorted = sorted(results['layer_points'].items(), 
                                        key=lambda x: x[1].get('num_layers', 0))
            
            for key, layer_point in layer_points_sorted:
                if 'accuracy' in layer_point:
                    num_layers = layer_point['num_layers']
                    acc = layer_point['accuracy']
                    conf = layer_point.get('avg_confidence', 0)
                    report_lines.append(f"  {num_layers:4d} | {acc:6.2%} | {conf:6.4f}")
        
        report_lines.append("\n" + "="*80)
        
        # 保存报告
        report_text = "\n".join(report_lines)
        save_path = self.output_dir / save_name
        
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        logger.info(f"✅ 保存摘要报告: {save_path}")
        
        # 同时打印到日志
        logger.info(f"\n{report_text}")
    
    def plot_all(self, rate_results: Dict, layer_results: Dict):
        """
        生成所有图表和报告
        
        Args:
            rate_results: 码率扫描结果
            layer_results: 层数扫描结果
        """
        logger.info("开始生成分析图表和报告...")
        
        # 码率方法图表
        if rate_results:
            self.plot_rate_vs_accuracy(rate_results)
            self.plot_rate_vs_confidence(rate_results)
        
        # 层数方法图表
        if layer_results:
            self.plot_layer_vs_accuracy(layer_results)
        
        # 对比图
        if rate_results and layer_results:
            self.plot_combined_comparison(rate_results, layer_results)
        
        # 摘要报告
        self.generate_summary_report(rate_results, layer_results)
        
        logger.info(f"✅ 所有分析结果已保存到: {self.output_dir}")

