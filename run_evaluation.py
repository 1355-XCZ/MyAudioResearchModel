"""
统一评估脚本 - 支持所有数据集
用法:
    python run_evaluation.py --dataset iemocap
    python run_evaluation.py --dataset ravdess --samples 50
    python run_evaluation.py --dataset esd --rates "10,30,50,100"
"""

import torch
import logging
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
# 项目现在是自包含的，不需要引用外层 Amphion 代码

from config import get_default_config
from grouped_rvq import GroupedResidualVQ
from entropy_model import create_entropy_model

eval_module_path = Path(__file__).parent / 'evaluation'
if str(eval_module_path) not in sys.path:
    sys.path.insert(0, str(eval_module_path))

from emotion_classifier import EmotionClassifierV2
from method_rate_sweep import rate_sweep_evaluation
from analyzer import ResultAnalyzer

dataset_module_path = Path(__file__).parent / 'datasets'
if str(dataset_module_path) not in sys.path:
    sys.path.insert(0, str(dataset_module_path))

from iemocap_dataset import IEMOCAPDataset
from ravdess_dataset import RAVDESSDataset
from esd_dataset import ESDDataset

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

DATASETS = {
    'iemocap': IEMOCAPDataset,
    'ravdess': RAVDESSDataset,
    'esd': ESDDataset,
}


def parse_args():
    parser = argparse.ArgumentParser(description='评估情感识别')
    parser.add_argument('--dataset', type=str, required=True, choices=DATASETS.keys(),
                       help='数据集名称 (iemocap/ravdess/esd)')
    parser.add_argument('--samples', type=int, default=100,
                       help='每个情感的样本数 (默认: 100)')
    parser.add_argument('--rates', type=str, default=None,
                       help='码率点 (逗号分隔, 如 "10,30,50,100")')
    parser.add_argument('--output-dir', type=str, default='evaluation_results',
                       help='输出目录 (默认: evaluation_results)')
    parser.add_argument('--rvq-checkpoint', type=str, default='checkpoints/grouped_rvq_best.pt',
                       help='RVQ模型路径')
    parser.add_argument('--entropy-checkpoint', type=str, default='checkpoints/entropy_model_best.pt',
                       help='熵模型路径')
    return parser.parse_args()


def main():
    args = parse_args()
    
    logger.info("="*80)
    logger.info(f"{args.dataset.upper()}评估（{args.samples}样本/情感）")
    logger.info("="*80)
    
    config = get_default_config()
    
    # 覆盖配置
    if args.rates:
        rate_list = [float(r.strip()) for r in args.rates.split(',')]
        config['evaluation'].rate_sweep_rates_bpf = rate_list
    elif config['evaluation'].rate_sweep_rates_bpf is None:
        config['evaluation'].rate_sweep_rates_bpf = config['entropy_model'].target_bpf_grid
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"使用设备: {device}")
    
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # 加载模型
    logger.info("\n" + "="*80)
    logger.info("加载模型")
    logger.info("="*80)
    
    rvq_model = GroupedResidualVQ(config['grouped_rvq'])
    rvq_checkpoint = torch.load(args.rvq_checkpoint, map_location=device)
    rvq_model.load_state_dict(rvq_checkpoint['model_state_dict'])
    rvq_model = rvq_model.to(device)
    rvq_model.eval()
    logger.info(f"✓ RVQ模型已加载: {args.rvq_checkpoint}")
    
    entropy_model = create_entropy_model(config['entropy_model'], config['grouped_rvq'])
    entropy_checkpoint = torch.load(args.entropy_checkpoint, map_location=device)
    entropy_model.load_state_dict(entropy_checkpoint['model_state_dict'])
    entropy_model = entropy_model.to(device)
    entropy_model.eval()
    logger.info(f"✓ 熵模型已加载: {args.entropy_checkpoint}")
    
    classifier = EmotionClassifierV2(
        model_name="iic/emotion2vec_plus_base",  # 使用plus_base(768维)匹配提取的特征
        hub="modelscope",
        device=device
    )
    logger.info(f"✓ 分类器已加载: emotion2vec_plus_base (768维)")
    
    # 加载数据集
    logger.info("\n" + "="*80)
    logger.info("加载数据集")
    logger.info("="*80)
    
    DatasetClass = DATASETS[args.dataset]
    # 数据集路径：data_root/DATASET_NAME/
    dataset_data_root = Path(config['data'].data_root) / args.dataset.upper()
    dataset = DatasetClass(
        data_root=str(dataset_data_root),
        samples_per_emotion=args.samples
    )
    
    logger.info(f"✓ {args.dataset.upper()}数据集已加载")
    logger.info(f"  数据集: {dataset.name}")
    logger.info(f"  总样本数: {len(dataset)}")
    
    # 运行评估
    logger.info("\n" + "="*80)
    logger.info("开始码率扫描评估")
    logger.info("="*80)
    
    results = rate_sweep_evaluation(
        rvq_model=rvq_model,
        entropy_model=entropy_model,
        dataset=dataset,
        target_rates_bpf=config['evaluation'].rate_sweep_rates_bpf,
        classifier=classifier,
        output_dir=str(Path(args.output_dir)),
        device=device
    )
    
    # 分析结果
    if results:
        logger.info("\n" + "="*80)
        logger.info("分析结果")
        logger.info("="*80)
        
        analyzer = ResultAnalyzer(output_dir=Path(args.output_dir))
        analyzer.analyze_rate_sweep(results, dataset.name)
    
    logger.info("\n" + "="*80)
    logger.info("评估完成！")
    logger.info("="*80)


if __name__ == '__main__':
    main()

