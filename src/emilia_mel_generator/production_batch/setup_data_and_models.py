#!/usr/bin/env python3
"""
数据和模型设置脚本
用于在服务器上下载和设置Emilia数据集和模型权重
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import shutil
import json
from typing import Optional

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataAndModelSetup:
    """数据和模型设置管理器"""
    
    def __init__(self, base_path: str, cache_path: str):
        self.base_path = Path(base_path)
        self.cache_path = Path(cache_path)
        
        # 创建必要的目录
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.cache_path.mkdir(parents=True, exist_ok=True)
        
        # 设置环境变量
        self.setup_cache_environment()
        
        logger.info(f"数据管理器初始化完成")
        logger.info(f"  基础路径: {self.base_path}")
        logger.info(f"  缓存路径: {self.cache_path}")
    
    def setup_cache_environment(self):
        """设置缓存环境变量"""
        # HuggingFace缓存
        os.environ['HF_HOME'] = str(self.cache_path / "huggingface")
        os.environ['TRANSFORMERS_CACHE'] = str(self.cache_path / "transformers")
        os.environ['HF_DATASETS_CACHE'] = str(self.cache_path / "datasets")
        
        # Torch缓存
        os.environ['TORCH_HOME'] = str(self.cache_path / "torch")
        
        # 创建缓存目录
        for env_var in ['HF_HOME', 'TRANSFORMERS_CACHE', 'HF_DATASETS_CACHE', 'TORCH_HOME']:
            Path(os.environ[env_var]).mkdir(parents=True, exist_ok=True)
        
        logger.info("✅ 缓存环境变量设置完成")
    
    def download_emilia_dataset(self, subset: Optional[str] = None, max_samples: Optional[int] = None):
        """下载Emilia数据集"""
        logger.info("📥 开始下载Emilia数据集...")
        
        dataset_path = self.base_path / "emilia_dataset"
        
        # 检查是否已存在
        if dataset_path.exists() and any(dataset_path.iterdir()):
            logger.info(f"✅ Emilia数据集已存在: {dataset_path}")
            return dataset_path
        
        try:
            from datasets import load_dataset
            
            logger.info("⬇️ 从HuggingFace下载Emilia数据集...")
            
            # 下载配置
            download_kwargs = {
                "cache_dir": str(self.cache_path / "datasets"),
            }
            
            if subset:
                download_kwargs["name"] = subset
            
            # 下载数据集
            dataset = load_dataset("amphion/Emilia-Dataset", **download_kwargs)
            
            # 如果指定了最大样本数，进行采样
            if max_samples:
                logger.info(f"📊 限制样本数量: {max_samples}")
                for split in dataset.keys():
                    if len(dataset[split]) > max_samples:
                        dataset[split] = dataset[split].select(range(max_samples))
            
            # 保存到指定目录
            dataset.save_to_disk(str(dataset_path))
            
            logger.info(f"✅ Emilia数据集下载完成: {dataset_path}")
            
            # 生成数据集信息
            self._generate_dataset_info(dataset, dataset_path)
            
            return dataset_path
            
        except Exception as e:
            logger.error(f"❌ Emilia数据集下载失败: {e}")
            raise
    
    def download_model_weights(self):
        """下载所有必需的模型权重"""
        logger.info("🤖 开始下载模型权重...")
        
        models_path = self.base_path / "models"
        models_path.mkdir(parents=True, exist_ok=True)
        
        # 1. 下载BigVGAN模型
        self._download_bigvgan_model(models_path)
        
        # 2. 下载Vevo TTS模型
        self._download_vevo_models(models_path)
        
        # 3. 下载Emotion2Vec模型
        self._download_emotion2vec_model(models_path)
        
        logger.info("✅ 所有模型权重下载完成")
    
    def _download_bigvgan_model(self, models_path: Path):
        """下载BigVGAN模型"""
        logger.info("⬇️ 下载BigVGAN模型...")
        
        try:
            import bigvgan
            
            # 这会自动下载到torch cache
            model = bigvgan.BigVGAN.from_pretrained("nvidia/bigvgan_v2_24khz_100band_256x")
            logger.info("✅ BigVGAN模型下载完成")
            
        except Exception as e:
            logger.error(f"❌ BigVGAN模型下载失败: {e}")
            raise
    
    def _download_vevo_models(self, models_path: Path):
        """下载Vevo TTS模型"""
        logger.info("⬇️ 下载Vevo TTS模型...")
        
        vevo_cache_path = models_path / "vevo"
        vevo_cache_path.mkdir(parents=True, exist_ok=True)
        
        try:
            from huggingface_hub import snapshot_download
            
            components = [
                ("tokenizer/vq8192/*", "tokenizer"),
                ("contentstyle_modeling/PhoneToVq8192/*", "AR模型"),
                ("acoustic_modeling/Vq8192ToMels/*", "FMT模型"),
                ("acoustic_modeling/Vocoder/*", "Vocoder模型")
            ]
            
            for pattern, name in components:
                logger.info(f"  下载{name}...")
                snapshot_download(\n                    repo_id=\"amphion/Vevo\",\n                    repo_type=\"model\",\n                    cache_dir=str(vevo_cache_path),\n                    allow_patterns=[pattern],\n                )\n            \n            logger.info(\"✅ Vevo TTS模型下载完成\")\n            \n        except Exception as e:\n            logger.error(f\"❌ Vevo TTS模型下载失败: {e}\")\n            raise\n    \n    def _download_emotion2vec_model(self, models_path: Path):\n        \"\"\"下载Emotion2Vec模型\"\"\"\n        logger.info(\"⬇️ 下载Emotion2Vec模型...\")\n        \n        try:\n            # Emotion2Vec模型通常在transformers缓存中自动下载\n            logger.info(\"✅ Emotion2Vec模型将在使用时自动下载\")\n            \n        except Exception as e:\n            logger.error(f\"❌ Emotion2Vec模型设置失败: {e}\")\n            # 不抛出异常，因为这个模型可以在运行时下载\n    \n    def _generate_dataset_info(self, dataset, dataset_path: Path):\n        \"\"\"生成数据集信息文件\"\"\"\n        info = {\n            \"dataset_name\": \"Emilia-Dataset\",\n            \"download_time\": str(pd.Timestamp.now()),\n            \"splits\": {},\n            \"total_samples\": 0\n        }\n        \n        for split_name in dataset.keys():\n            split_size = len(dataset[split_name])\n            info[\"splits\"][split_name] = split_size\n            info[\"total_samples\"] += split_size\n        \n        # 保存信息文件\n        with open(dataset_path / \"dataset_info.json\", 'w', encoding='utf-8') as f:\n            json.dump(info, f, indent=2, ensure_ascii=False)\n        \n        logger.info(f\"📊 数据集信息: {info['total_samples']} 个样本，{len(info['splits'])} 个分割\")\n    \n    def clean_cache(self, keep_models: bool = True):\n        \"\"\"清理缓存文件\"\"\"\n        logger.info(\"🧹 清理缓存文件...\")\n        \n        if not keep_models:\n            # 完全清理\n            if self.cache_path.exists():\n                shutil.rmtree(self.cache_path)\n                logger.info(\"✅ 缓存完全清理\")\n        else:\n            # 只清理临时文件\n            temp_patterns = [\"temp_*\", \"tmp_*\", \"*.tmp\", \"*.temp\"]\n            for pattern in temp_patterns:\n                for temp_file in self.cache_path.rglob(pattern):\n                    if temp_file.is_file():\n                        temp_file.unlink()\n                    elif temp_file.is_dir():\n                        shutil.rmtree(temp_file)\n            \n            logger.info(\"✅ 临时文件清理完成\")\n    \n    def check_disk_space(self) -> dict:\n        \"\"\"检查磁盘空间\"\"\"\n        import shutil as sh\n        \n        base_usage = sh.disk_usage(self.base_path)\n        cache_usage = sh.disk_usage(self.cache_path)\n        \n        info = {\n            \"base_path\": {\n                \"path\": str(self.base_path),\n                \"total_gb\": base_usage.total / (1024**3),\n                \"used_gb\": base_usage.used / (1024**3),\n                \"free_gb\": base_usage.free / (1024**3)\n            },\n            \"cache_path\": {\n                \"path\": str(self.cache_path),\n                \"total_gb\": cache_usage.total / (1024**3),\n                \"used_gb\": cache_usage.used / (1024**3),\n                \"free_gb\": cache_usage.free / (1024**3)\n            }\n        }\n        \n        logger.info(\"💾 磁盘空间信息:\")\n        logger.info(f\"  基础路径: {info['base_path']['free_gb']:.1f}GB 可用\")\n        logger.info(f\"  缓存路径: {info['cache_path']['free_gb']:.1f}GB 可用\")\n        \n        return info\n\n\ndef parse_arguments():\n    \"\"\"解析命令行参数\"\"\"\n    parser = argparse.ArgumentParser(description=\"数据和模型设置脚本\")\n    \n    parser.add_argument(\"--base_path\", type=str, required=True,\n                       help=\"基础路径（数据和模型存储位置）\")\n    parser.add_argument(\"--cache_path\", type=str, required=True,\n                       help=\"缓存路径（临时文件和模型缓存）\")\n    parser.add_argument(\"--download_dataset\", action=\"store_true\",\n                       help=\"下载Emilia数据集\")\n    parser.add_argument(\"--download_models\", action=\"store_true\",\n                       help=\"下载模型权重\")\n    parser.add_argument(\"--max_samples\", type=int, default=None,\n                       help=\"最大样本数量（用于测试）\")\n    parser.add_argument(\"--clean_cache\", action=\"store_true\",\n                       help=\"清理缓存文件\")\n    parser.add_argument(\"--check_space\", action=\"store_true\",\n                       help=\"检查磁盘空间\")\n    \n    return parser.parse_args()\n\n\ndef main():\n    \"\"\"主函数\"\"\"\n    args = parse_arguments()\n    \n    print(\"=====================================\")\n    print(\"🚀 Emilia数据和模型设置\")\n    print(\"=====================================\")\n    \n    try:\n        # 创建设置管理器\n        setup_manager = DataAndModelSetup(\n            base_path=args.base_path,\n            cache_path=args.cache_path\n        )\n        \n        # 检查磁盘空间\n        if args.check_space:\n            setup_manager.check_disk_space()\n        \n        # 下载数据集\n        if args.download_dataset:\n            setup_manager.download_emilia_dataset(max_samples=args.max_samples)\n        \n        # 下载模型\n        if args.download_models:\n            setup_manager.download_model_weights()\n        \n        # 清理缓存\n        if args.clean_cache:\n            setup_manager.clean_cache()\n        \n        print(\"=====================================\")\n        print(\"✅ 数据和模型设置完成！\")\n        print(\"=====================================\")\n        \n        return 0\n        \n    except Exception as e:\n        logger.error(f\"❌ 设置失败: {e}\")\n        import traceback\n        traceback.print_exc()\n        return 1\n\n\nif __name__ == \"__main__\":\n    import pandas as pd  # 用于时间戳\n    sys.exit(main())
