"""
ESD快速评估（20样本/情感）- 临时测试
"""

import torch
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from config import get_default_config
from grouped_rvq import GroupedResidualVQ
from entropy_model import create_entropy_model

eval_module_path = Path(__file__).parent / 'evaluation'
if str(eval_module_path) not in sys.path:
    sys.path.insert(0, str(eval_module_path))

from emotion_classifier import EmotionClassifierV2
from method_rate_sweep import rate_sweep_evaluation
from method_layer_sweep import layer_sweep_evaluation
from analyzer import ResultAnalyzer

dataset_module_path = Path(__file__).parent / 'datasets'
if str(dataset_module_path) not in sys.path:
    sys.path.insert(0, str(dataset_module_path))

from esd_dataset import ESDDataset

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    logger.info("="*80)
    logger.info("ESD快速评估（20样本/情感）- 临时测试")
    logger.info("="*80)
    
    config = get_default_config()
    
    # 快速测试配置
    quick_rates = [10, 20, 30, 40, 60, 80, 100, 200, 300, float('inf')]  # 10个码率点
    quick_layers = [12*i for i in range(2, 17, 2)]  # 每2层递增：24,48,72,...,192
    
    logger.info(f"快速码率点: {quick_rates}")
    logger.info(f"快速层数点: {quick_layers}")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"使用设备: {device}")
    
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        logger.info("✅ TF32加速已启用")
    
    logger.info("\n加载模型...")
    rvq_checkpoint = torch.load('checkpoints/grouped_rvq_best.pt', map_location=device)
    rvq_model = GroupedResidualVQ(config['grouped_rvq']).to(device)
    rvq_model.load_state_dict(rvq_checkpoint['model_state_dict'])
    rvq_model.eval()
    
    entropy_checkpoint = torch.load('checkpoints/entropy_model_best.pt', map_location=device)
    entropy_model = create_entropy_model(config['entropy_model'], config['grouped_rvq']).to(device)
    entropy_model.load_state_dict(entropy_checkpoint['model_state_dict'])
    entropy_model.eval()
    
    if not hasattr(entropy_model, '_mask_cache'):
        entropy_model._mask_cache = {}
    
    logger.info("✅ 模型已加载")
    
    logger.info("\n初始化分类器...")
    classifier = EmotionClassifierV2(
        model_name=config['evaluation'].emotion2vec_model,
        hub=config['evaluation'].emotion2vec_hub,
        device=device
    )
    classifier.load_model()
    logger.info("✅ 分类器已加载")
    
    logger.info("\n准备数据集...")
    dataset = ESDDataset(
        "/data/gpfs/projects/punim2341/haoguangzhou/data/evaluation_features/ESD", 
        languages=['english', 'chinese'], 
        samples_per_emotion=20  # 20样本/情感
    )
    
    # 使用独立的输出目录（不污染正式评估）
    output_dir = Path('evaluation_quick_results')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("\n方法1：码率扫描...")
    rate_result = rate_sweep_evaluation(
        rvq_model=rvq_model,
        entropy_model=entropy_model,
        dataset=dataset,
        target_rates_bpf=quick_rates,
        classifier=classifier,
        output_dir=str(output_dir / 'rate_sweep'),
        device=device
    )
    logger.info("✅ 码率扫描完成")
    
    logger.info("\n方法2：层数扫描...")
    layer_result = layer_sweep_evaluation(
        rvq_model=rvq_model,
        dataset=dataset,
        num_layers_list=quick_layers,
        classifier=classifier,
        output_dir=str(output_dir / 'layer_sweep'),
        device=device
    )
    logger.info("✅ 层数扫描完成")
    
    logger.info("\n生成快速分析...")
    analyzer = ResultAnalyzer(str(output_dir / 'analysis'))
    analyzer.plot_all({'ESD': rate_result}, {'ESD': layer_result})
    
    logger.info("\n✅ ESD快速评估完成！")
    logger.info(f"结果保存在: {output_dir}")


if __name__ == "__main__":
    main()

