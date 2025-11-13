"""
评估RAVDESS数据集
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

dataset_module_path = Path(__file__).parent / 'datasets'
if str(dataset_module_path) not in sys.path:
    sys.path.insert(0, str(dataset_module_path))

from ravdess_dataset import RAVDESSDataset

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    logger.info("="*80)
    logger.info("RAVDESS评估（100样本/情感）")
    logger.info("="*80)
    
    config = get_default_config()
    
    if config['evaluation'].rate_sweep_rates_bpf is None:
        config['evaluation'].rate_sweep_rates_bpf = config['entropy_model'].target_bpf_grid
    
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
    dataset = RAVDESSDataset("/data/gpfs/projects/punim2341/haoguangzhou/data/evaluation_features/RAVDESS", samples_per_emotion=100)
    
    output_dir = Path('evaluation_results')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("\n开始评估...")
    result = rate_sweep_evaluation(
        rvq_model=rvq_model,
        entropy_model=entropy_model,
        dataset=dataset,
        target_rates_bpf=config['evaluation'].rate_sweep_rates_bpf,
        classifier=classifier,
        output_dir=str(output_dir / 'rate_sweep'),
        device=device
    )
    
    logger.info("\n✅ RAVDESS评估完成！")


if __name__ == "__main__":
    main()

