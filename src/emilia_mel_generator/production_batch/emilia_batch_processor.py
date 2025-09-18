"""
Emilia数据集批处理脚本 - 正式训练数据生成版本
基于test_emilia_correct.py的成功实现，专为训练数据生成优化

核心功能（仅生成训练数据）：
1. 使用Vevo提取原始音频的mel频谱图（兼容Vevo声码器）
2. 使用Vevo TTS生成中性语音的mel频谱图
3. 提取emotion2vec情感特征
4. 支持大规模批处理和进度管理
5. 不生成音频文件，只生成训练所需数据

作者: AI Assistant
日期: 2025-09-18
版本: 2.0 (Training Data Generation)
"""

import os
import sys
import argparse
import torch
import torchaudio
import numpy as np
import soundfile as sf
from pathlib import Path
import logging
import pickle
import json
import time
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import traceback
from tqdm import tqdm  # 进度条

# 设置日志
def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """设置日志配置"""
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    if log_file:
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format=log_format,
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
    else:
        logging.basicConfig(level=getattr(logging, log_level.upper()), format=log_format)

logger = logging.getLogger(__name__)

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent))

from corrected_data_generator import VevoTTSEmulator, Emotion2VecExtractor

# BigVGAN导入
try:
    import bigvgan
    from bigvgan import get_mel_spectrogram
    BIGVGAN_AVAILABLE = True
    logger.info("✅ BigVGAN和get_mel_spectrogram可用")
except ImportError:
    BIGVGAN_AVAILABLE = False
    logger.error("❌ BigVGAN不可用")

# Vevo TTS导入逻辑（从test_emilia_correct.py复制）
amphion_path = Path(__file__).parent.parent / "Amphion"
vevo_path = Path(__file__).parent.parent / "VQ-VAE" / "vevo-code" / "vevo"

# 添加路径到sys.path（优先Amphion路径，避免被顶层src/models遮蔽）
sys.path.insert(0, str(amphion_path))
sys.path.insert(0, str(vevo_path))

# 尝试导入Vevo TTS
REAL_VEVO_AVAILABLE = False
VEVO_SOURCE = None
VevoInferencePipeline = None
save_audio = None
g2p_ = None

def _try_import_vevo_from_amphion() -> bool:
    original_cwd = None
    original_path = None
    try:
        # 清理所有相关模块缓存
        to_delete = [k for k in list(sys.modules.keys()) 
                    if k in ['models', 'utils'] or k.startswith(('models.', 'utils.'))]
        for k in to_delete:
            del sys.modules[k]
        
        # 保存原始环境
        original_cwd = os.getcwd()
        original_path = sys.path.copy()
        
        # 切换到Amphion目录并调整sys.path
        os.chdir(str(amphion_path))
        # 清理sys.path，确保Amphion目录优先
        sys.path.clear()
        sys.path.extend([str(amphion_path)] + original_path)
        
        # 确保utils包可用
        utils_path = amphion_path / "utils"
        if not (utils_path / "__init__.py").exists():
            with open(utils_path / "__init__.py", 'w') as f:
                f.write("# Amphion utils package\n")
        
        # 先测试utils导入
        import utils
        from utils.util import load_config
        logger.info("✅ utils.util导入成功")
        
        # 再测试Vevo TTS导入
        from models.vc.vevo.vevo_utils import VevoInferencePipeline as _VIP, save_audio as _SA, g2p_ as _G2P
        globals()['VevoInferencePipeline'] = _VIP
        globals()['save_audio'] = _SA
        globals()['g2p_'] = _G2P
        logger.info("✅ 真正的Vevo TTS可用 (Amphion)")
        return True
    except Exception as e:
        logger.error(f"Amphion导入失败: {e}")
        return False
    finally:
        # 恢复原始环境
        if original_cwd is not None:
            os.chdir(original_cwd)
        if original_path is not None:
            sys.path.clear()
            sys.path.extend(original_path)

def _try_import_vevo_from_vqvae() -> bool:
    try:
        os.chdir(str(vevo_path.parent))
        from vevo.vevo_utils import VevoInferencePipeline as _VIP, save_audio as _SA, g2p_ as _G2P
        globals()['VevoInferencePipeline'] = _VIP
        globals()['save_audio'] = _SA
        globals()['g2p_'] = _G2P
        logger.info("✅ 真正的Vevo TTS可用 (VQ-VAE)")
        return True
    except Exception as e:
        logger.debug(f"VQ-VAE导入失败: {e}")
        return False
    finally:
        os.chdir(str(Path(__file__).parent))

def _try_import_vevo_via_file() -> bool:
    try:
        import importlib.util
        # 优先Amphion文件路径
        amphion_file = amphion_path / "models" / "vc" / "vevo" / "vevo_utils.py"
        vqvae_file = vevo_path / "vevo_utils.py"
        target_file = amphion_file if amphion_file.exists() else vqvae_file if vqvae_file.exists() else None
        if target_file is None:
            return False
        spec = importlib.util.spec_from_file_location("vevo_utils_dynamic", str(target_file))
        if spec is None or spec.loader is None:
            return False
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        globals()['VevoInferencePipeline'] = getattr(mod, 'VevoInferencePipeline', None)
        globals()['save_audio'] = getattr(mod, 'save_audio', None)
        globals()['g2p_'] = getattr(mod, 'g2p_', None)
        ok = all([VevoInferencePipeline, save_audio, g2p_])
        if ok:
            logger.info("✅ 真正的Vevo TTS可用 (文件路径导入)")
        return ok
    except Exception as e:
        logger.debug(f"文件路径导入失败: {e}")
        return False

# 依次尝试三种导入方式
if _try_import_vevo_from_amphion():
    REAL_VEVO_AVAILABLE = True
    VEVO_SOURCE = "amphion"
elif _try_import_vevo_from_vqvae():
    REAL_VEVO_AVAILABLE = True
    VEVO_SOURCE = "vqvae"
elif _try_import_vevo_via_file():
    REAL_VEVO_AVAILABLE = True
    VEVO_SOURCE = "file"
else:
    REAL_VEVO_AVAILABLE = False
    VEVO_SOURCE = None
    logger.warning("⚠️ 真正的Vevo TTS不可用：多种导入方式均失败")


class EmiliaBatchProcessor:
    """
    Emilia数据集批处理器 - 正式生产版本
    支持大规模数据处理、进度管理、错误恢复
    """
    
    def __init__(self, 
                 input_data_path: str,
                 output_base_path: str,
                 batch_size: int = 100,
                 device: str = 'cuda',
                 resume_from_checkpoint: bool = True,
                 checkpoint_interval: int = 50):
        """
        初始化批处理器
        
        Args:
            input_data_path: 输入数据路径（Emilia数据集路径）
            output_base_path: 输出基础路径
            batch_size: 批处理大小
            device: 计算设备
            resume_from_checkpoint: 是否从检查点恢复
            checkpoint_interval: 检查点保存间隔
        """
        self.input_data_path = Path(input_data_path)
        self.output_base_path = Path(output_base_path)
        self.batch_size = batch_size
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.resume_from_checkpoint = resume_from_checkpoint
        self.checkpoint_interval = checkpoint_interval
        
        # 创建输出目录结构
        self.setup_output_directories()
        
        # 初始化处理组件
        self.vevo_emulator = None
        self.emotion2vec_extractor = None
        self.bigvgan_model = None
        self.real_vevo_pipeline = None
        
        # 进度管理
        self.processed_count = 0
        self.total_count = 0
        self.failed_samples = []
        self.start_time = None
        
        # 简化初始化日志
        print(f"✅ 批处理器初始化完成 (批大小: {self.batch_size}, 设备: {self.device})")

    def setup_output_directories(self):
        """创建输出目录结构"""
        self.output_dirs = {
            'mels': self.output_base_path / "mels",
            'emotion_features': self.output_base_path / "emotion_features", 
            # 正式版本不生成验证音频，只生成训练数据
            # 'verification_audio': self.output_base_path / "verification_audio",
            'checkpoints': self.output_base_path / "checkpoints",
            'logs': self.output_base_path / "logs",
            'reports': self.output_base_path / "reports"
        }
        
        for dir_path in self.output_dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # logger.info("📁 输出目录结构创建完成")

    def initialize_models(self):
        """初始化所有模型组件"""
        print("🚀 初始化模型组件...")
        
        # 初始化组件（正式版本不需要VevoTTSEmulator）
        # self.vevo_emulator = VevoTTSEmulator(device=self.device)
        self.emotion2vec_extractor = Emotion2VecExtractor(device=self.device)
        
        # 正式版本不需要BigVGAN，只用Vevo
        # self._init_bigvgan()
        
        # 初始化Vevo TTS
        self._init_vevo_tts()
        
        print("✅ 所有模型组件初始化完成")

    # 正式版本不需要BigVGAN，只使用Vevo系统
    # def _init_bigvgan(self): ...

    def _init_vevo_tts(self):
        """初始化Vevo TTS"""
        if not REAL_VEVO_AVAILABLE:
            raise RuntimeError("Vevo TTS模块导入失败，无法继续")
            
        try:
            logger.info("🚀 初始化Vevo TTS...")
            
            # 使用已经下载的Vevo模型
            if VEVO_SOURCE == "amphion":
                base_cache_dir = str(amphion_path / "ckpts/Vevo")
                config_base = str(amphion_path / "models/vc/vevo/config")
            else:
                base_cache_dir = str(vevo_path.parent / "ckpt/Vevo")
                config_base = str(vevo_path / "config")
            
            # 检查模型是否已下载
            import glob
            vq8192_paths = glob.glob(f"{base_cache_dir}/models--amphion--Vevo/snapshots/*/tokenizer/vq8192")
            if not vq8192_paths:
                vq8192_paths = glob.glob(f"{base_cache_dir}/tokenizer/vq8192")
                
            ar_paths = glob.glob(f"{base_cache_dir}/models--amphion--Vevo/snapshots/*/contentstyle_modeling/PhoneToVq8192")
            if not ar_paths:
                ar_paths = glob.glob(f"{base_cache_dir}/contentstyle_modeling/PhoneToVq8192")
                
            fmt_paths = glob.glob(f"{base_cache_dir}/models--amphion--Vevo/snapshots/*/acoustic_modeling/Vq8192ToMels")
            if not fmt_paths:
                fmt_paths = glob.glob(f"{base_cache_dir}/acoustic_modeling/Vq8192ToMels")
                
            vocoder_paths = glob.glob(f"{base_cache_dir}/models--amphion--Vevo/snapshots/*/acoustic_modeling/Vocoder")
            if not vocoder_paths:
                vocoder_paths = glob.glob(f"{base_cache_dir}/acoustic_modeling/Vocoder")
            
            # 如缺失，自动下载权重
            if not all([vq8192_paths, ar_paths, fmt_paths, vocoder_paths]):
                logger.info("⬇️ 未找到完整Vevo权重，尝试从HuggingFace自动下载...")
                self._download_vevo_models(base_cache_dir)
                
                # 重新搜集路径
                vq8192_paths = glob.glob(f"{base_cache_dir}/models--amphion--Vevo/snapshots/*/tokenizer/vq8192") or glob.glob(f"{base_cache_dir}/tokenizer/vq8192")
                ar_paths = glob.glob(f"{base_cache_dir}/models--amphion--Vevo/snapshots/*/contentstyle_modeling/PhoneToVq8192") or glob.glob(f"{base_cache_dir}/contentstyle_modeling/PhoneToVq8192")
                fmt_paths = glob.glob(f"{base_cache_dir}/models--amphion--Vevo/snapshots/*/acoustic_modeling/Vq8192ToMels") or glob.glob(f"{base_cache_dir}/acoustic_modeling/Vq8192ToMels")
                vocoder_paths = glob.glob(f"{base_cache_dir}/models--amphion--Vevo/snapshots/*/acoustic_modeling/Vocoder") or glob.glob(f"{base_cache_dir}/acoustic_modeling/Vocoder")

            if not all([vq8192_paths, ar_paths, fmt_paths, vocoder_paths]):
                raise RuntimeError(f"Vevo模型文件不完整，查找路径: {base_cache_dir}")
            
            logger.info(f"找到模型文件:")
            logger.info(f"  tokenizer: {vq8192_paths[0]}")
            logger.info(f"  ar_model: {ar_paths[0]}")
            logger.info(f"  fmt_model: {fmt_paths[0]}")
            logger.info(f"  vocoder: {vocoder_paths[0]}")
            
            # 切换到Amphion目录进行初始化
            original_cwd = os.getcwd()
            try:
                os.chdir(str(amphion_path))
                
                # 初始化Vevo推理流水线
                self.real_vevo_pipeline = VevoInferencePipeline(
                    content_style_tokenizer_ckpt_path=vq8192_paths[0],
                    ar_cfg_path=f"./models/vc/vevo/config/PhoneToVq8192.json",
                    ar_ckpt_path=ar_paths[0],
                    fmt_cfg_path=f"./models/vc/vevo/config/Vq8192ToMels.json",
                    fmt_ckpt_path=fmt_paths[0],
                    vocoder_cfg_path=f"./models/vc/vevo/config/Vocoder.json",
                    vocoder_ckpt_path=vocoder_paths[0],
                    device=self.device,
                )
            finally:
                os.chdir(original_cwd)
            
            logger.info("✅ Vevo TTS初始化成功！")
            
        except Exception as e:
            logger.error(f"❌ Vevo TTS初始化失败: {e}")
            raise

    def _download_vevo_models(self, base_cache_dir: str):
        """自动下载Vevo模型"""
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
                snapshot_download(
                    repo_id="amphion/Vevo",
                    repo_type="model",
                    cache_dir=base_cache_dir,
                    allow_patterns=[pattern],
                )
            
            logger.info("✅ Vevo模型下载完成")
            
        except Exception as e:
            logger.error(f"❌ Vevo模型下载失败: {e}")
            raise

    def download_and_setup_data(self):
        """在服务器上下载和设置数据"""
        print("🚀 在服务器上设置数据和模型...")
        
        # 1. 下载Emilia数据集
        self._download_emilia_dataset()
        
        # 2. 下载和设置模型权重
        self._setup_model_weights()
        
            print("✅ 数据和模型设置完成")
    
    def _download_emilia_dataset(self):
        """下载Emilia数据集到服务器暂存目录"""
        logger.info("📦 下载Emilia数据集...")
        
        # 检查数据集是否已存在
        if self.input_data_path.exists() and any(self.input_data_path.iterdir()):
            logger.info(f"✅ Emilia数据集已存在: {self.input_data_path}")
            return
        
        # 创建数据集目录
        self.input_data_path.mkdir(parents=True, exist_ok=True)
        
        try:
            # 使用HuggingFace datasets下载Emilia数据集
            logger.info("⬇️ 从 HuggingFace 下载 Emilia 数据集...")
            
            from datasets import load_dataset
            
            # 下载数据集到指定目录
            dataset = load_dataset(
                "amphion/Emilia-Dataset",
                cache_dir=str(self.input_data_path.parent / "cache"),
                streaming=False  # 下载完整数据集
            )
            
            # 保存数据集到指定目录
            dataset.save_to_disk(str(self.input_data_path))
            
            logger.info(f"✅ Emilia数据集下载完成: {self.input_data_path}")
            
        except Exception as e:
            logger.error(f"❌ Emilia数据集下载失败: {e}")
            # 尝试使用其他方式或预存在的数据
            logger.warning("⚠️ 将尝试使用本地或预存在的数据")
    
    def _setup_model_weights(self):
        """设置模型权重到暂存目录"""
        logger.info("🎯 设置模型权重...")
        
        # 设置缓存目录环境变量
        cache_base = self.input_data_path.parent / "cache"
        cache_base.mkdir(parents=True, exist_ok=True)
        
        # 设置HuggingFace缓存目录
        os.environ['HF_HOME'] = str(cache_base / "huggingface")
        os.environ['TRANSFORMERS_CACHE'] = str(cache_base / "transformers")
        os.environ['HF_DATASETS_CACHE'] = str(cache_base / "datasets")
        
        # 设置Torch Hub缓存目录
        os.environ['TORCH_HOME'] = str(cache_base / "torch")
        
        logger.info(f"✅ 模型缓存目录设置完成: {cache_base}")
        logger.info(f"  HuggingFace: {os.environ['HF_HOME']}")
        logger.info(f"  Torch Hub: {os.environ['TORCH_HOME']}")
    
    def load_dataset_metadata(self) -> List[Dict]:
        """加载数据集元数据"""
        print("📊 加载数据集元数据...")
        
        try:
            # 尝试从保存的数据集加载
            if self.input_data_path.exists():
                from datasets import load_from_disk
                dataset = load_from_disk(str(self.input_data_path))
                
                # 转换为列表格式
                metadata = []
                for split_name in dataset.keys():
                    split_data = dataset[split_name]
                    for i, item in enumerate(split_data):
                        sample = {
                            'id': f"{split_name}_{i:06d}",
                            'audio': item['audio'],
                            'text': item.get('text', ''),
                            'speaker': item.get('speaker', ''),
                            'language': item.get('language', 'en'),
                            'split': split_name
                        }
                        metadata.append(sample)
                
                print(f"✅ 从本地加载了 {len(metadata)} 个样本")
                return metadata
            
        except Exception as e:
            logger.warning(f"⚠️ 本地数据集加载失败: {e}")
        
        # 如果本地加载失败，尝试直接从 HuggingFace 加载
        try:
            logger.info("⬇️ 直接从 HuggingFace 加载 Emilia 数据集...")
            from datasets import load_dataset
            
            dataset = load_dataset(
                "amphion/Emilia-Dataset",
                streaming=True,  # 使用流式加载节省内存
                cache_dir=str(self.input_data_path.parent / "cache")
            )
            
            # 转换为列表格式（只加载需要的数量）
            metadata = []
            max_samples = getattr(self, 'max_samples', 10000)  # 限制最大样本数
            
            for split_name in dataset.keys():
                split_data = dataset[split_name]
                for i, item in enumerate(split_data):
                    if len(metadata) >= max_samples:
                        break
                        
                    sample = {
                        'id': f"{split_name}_{i:06d}",
                        'audio': item['audio'],
                        'text': item.get('text', ''),
                        'speaker': item.get('speaker', ''),
                        'language': item.get('language', 'en'),
                        'split': split_name
                    }
                    metadata.append(sample)
            
            print(f"✅ 从 HuggingFace 加载了 {len(metadata)} 个样本")
            return metadata
            
        except Exception as e:
            logger.error(f"❌ 数据集加载失败: {e}")
            raise RuntimeError(f"无法加载 Emilia 数据集: {e}")

    def process_sample(self, sample_data: Dict) -> Optional[Dict]:
        """
        处理单个音频样本
        
        Args:
            sample_data: 样本数据字典，包含audio、text、speaker等信息
            
        Returns:
            处理结果字典，如果失败返回None
        """
        sample_id = sample_data.get('id', 'unknown')
        
        try:
            # 减少单个样本的日志输出
            # logger.info(f"处理样本: {sample_id}")
            
            # 1. 准备音频数据
            audio_array = sample_data['audio']['array']
            sample_rate = sample_data['audio']['sampling_rate']
            
            # 确保采样率为24kHz
            if sample_rate != 24000:
                import librosa
                audio_array = librosa.resample(audio_array, orig_sr=sample_rate, target_sr=24000)
            
            # 2. 直接使用Vevo提取原始mel（确保兼容性）
            wav_tensor = torch.FloatTensor(audio_array).unsqueeze(0).to(self.device)
            # 不再使用BigVGAN提取，直接使用Vevo提取器
            mel_original_vevo = self.real_vevo_pipeline.extract_mel_feature(wav_tensor)
            
            # 3. 使用Vevo TTS生成中性mel
            mel_neutral_vevo = self._generate_neutral_mel_vevo(sample_data)
            
            # 4. 提取emotion2vec特征
            audio_tensor = torch.from_numpy(audio_array).float().to(self.device)
            if audio_tensor.dim() == 1:
                audio_tensor = audio_tensor.unsqueeze(0)
            ev2_features = self.emotion2vec_extractor.extract_features(audio_tensor)
            
            # 5. 保存训练数据文件
            self._save_training_data(sample_id, mel_original_vevo, mel_neutral_vevo, ev2_features)
            
            # 6. 正式版本不生成验证音频，只生成训练数据
            # self._generate_verification_audio_vevo(sample_id, audio_array, mel_original, mel_neutral)
            
            # 7. 返回处理结果
            return {
                "sample_id": sample_id,
                "language": sample_data.get('language', 'unknown'),
                "mel_original_shape": list(mel_original_vevo.shape),
                "mel_neutral_shape": list(mel_neutral_vevo.shape),
                "text": sample_data.get('text', ''),
                "speaker": sample_data.get('speaker', ''),
                "duration": len(audio_array) / 24000,
                "processing_time": time.time(),
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"处理样本 {sample_id} 失败: {e}")
            traceback.print_exc()
            
            # 记录失败样本
            self.failed_samples.append({
                "sample_id": sample_id,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })
            
            return None

    def _generate_neutral_mel_vevo(self, sample_data: Dict) -> torch.Tensor:
        """使用Vevo TTS生成中性mel"""
        # 从test_emilia_correct.py复制的核心逻辑
        
        s1_text = sample_data.get('text', 'Hello, this is a neutral speech.')
        s1_language = sample_data.get('language', 'en')
        
        # 创建临时文件
        temp_dir = Path("temp_vevo_call")
        temp_dir.mkdir(exist_ok=True)
        
        s1_temp_path = temp_dir / "s1_timbre.wav"
        neutral_temp_path = temp_dir / "neutral_style.wav"
        output_path = temp_dir / "vevo_output.wav"
        
        try:
            # 保存S1音频文件（作为音色参考）
            sf.write(s1_temp_path, sample_data['audio']['array'], 24000)
            
            # 创建neutral风格参考音频
            duration = min(3.0, len(sample_data['audio']['array']) / 24000)
            t = np.linspace(0, duration, int(duration * 24000))
            neutral_audio = 0.3 * np.sin(2 * np.pi * 150 * t) + 0.05 * np.random.randn(int(duration * 24000))
            sf.write(neutral_temp_path, neutral_audio.astype(np.float32), 24000)
            
            # 文本处理
            clean_text = s1_text.strip()
            if len(clean_text) > 200:
                clean_text = clean_text[:200].rsplit(' ', 1)[0] + '.'
            
            style_text = "This is neutral speech." if s1_language == 'en' else "这是中性语音。"
            
            # 使用Vevo TTS
            gen_audio = self.real_vevo_pipeline.inference_ar_and_fm(
                src_wav_path=None,
                src_text=clean_text,
                style_ref_wav_path=str(neutral_temp_path),
                style_ref_wav_text=style_text,
                timbre_ref_wav_path=str(s1_temp_path),
                src_text_language=s1_language,
                style_ref_wav_text_language=s1_language,
                flow_matching_steps=16,
                use_global_guided_inference=False
            )
            
            # 保存生成的音频
            save_audio(gen_audio, output_path=str(output_path))
            
            # 使用Vevo提取mel特征
            import librosa
            vevo_audio, _ = librosa.load(output_path, sr=24000)
            vevo_audio_tensor = torch.from_numpy(vevo_audio).float().unsqueeze(0).to(self.device)
            
            vevo_mel = self.real_vevo_pipeline.extract_mel_feature(vevo_audio_tensor)
            
            return vevo_mel
            
        finally:
            # 清理临时文件
            import shutil
            if temp_dir.exists():
                shutil.rmtree(temp_dir)

    def _save_training_data(self, sample_id: str, 
                           mel_original_vevo: torch.Tensor, 
                           mel_neutral_vevo: torch.Tensor, 
                           ev2_features: Dict):
        """保存训练数据 - 正式版本只生成训练所需数据"""
        
        # 1. 保存Vevo兼容的源频谱图
        np.save(self.output_dirs['mels'] / f"{sample_id}_mel_original_vevo.npy", 
                mel_original_vevo.squeeze().cpu().numpy())
        
        # 2. 保存中性变换后频谱图
        np.save(self.output_dirs['mels'] / f"{sample_id}_mel_neutral_vevo.npy", 
                mel_neutral_vevo.squeeze().cpu().numpy())
        
        # 3. 保存源音频的情感表征
        np.savez_compressed(
            self.output_dirs['emotion_features'] / f"{sample_id}_ev2.npz",
            utterance=ev2_features['utterance'].cpu().numpy(),
            frame=ev2_features['frame'].cpu().numpy()
        )
        
        # 减少详细日志输出，只在需要时显示
        # logger.info(f"  ✅ 保存训练数据: {sample_id}")

    # 正式版本不生成验证音频，只生成训练数据
    # def _generate_verification_audio_vevo(...): ...
    # def _minimal_post_process(...): ...

    def save_checkpoint(self, batch_idx: int, processed_results: List[Dict]):
        """保存检查点"""
        checkpoint_data = {
            "batch_idx": batch_idx,
            "processed_count": self.processed_count,
            "total_count": self.total_count,
            "failed_samples": self.failed_samples,
            "start_time": self.start_time,
            "checkpoint_time": datetime.now().isoformat(),
            "processed_results": processed_results
        }
        
        checkpoint_file = self.output_dirs['checkpoints'] / f"checkpoint_batch_{batch_idx:06d}.json"
        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 检查点已保存: {checkpoint_file}")

    def load_checkpoint(self) -> Optional[Dict]:
        """加载最新的检查点"""
        if not self.resume_from_checkpoint:
            return None
            
        checkpoint_files = list(self.output_dirs['checkpoints'].glob("checkpoint_batch_*.json"))
        if not checkpoint_files:
            return None
        
        # 找到最新的检查点
        latest_checkpoint = max(checkpoint_files, key=lambda x: x.stat().st_mtime)
        
        try:
            with open(latest_checkpoint, 'r', encoding='utf-8') as f:
                checkpoint_data = json.load(f)
            
            logger.info(f"🔄 从检查点恢复: {latest_checkpoint}")
            logger.info(f"  已处理: {checkpoint_data['processed_count']}/{checkpoint_data['total_count']}")
            
            return checkpoint_data
            
        except Exception as e:
            logger.error(f"❌ 检查点加载失败: {e}")
            return None

    def generate_final_report(self, all_results: List[Dict]):
        """生成最终处理报告"""
        success_count = len([r for r in all_results if r is not None])
        failed_count = len(self.failed_samples)
        
        report = {
            "processing_config": {
                "input_data_path": str(self.input_data_path),
                "output_base_path": str(self.output_base_path),
                "batch_size": self.batch_size,
                "device": self.device,
                "vevo_source": VEVO_SOURCE
            },
            "processing_summary": {
                "total_samples": self.total_count,
                "successful_samples": success_count,
                "failed_samples": failed_count,
                "success_rate": success_count / self.total_count if self.total_count > 0 else 0,
                "start_time": self.start_time,
                "end_time": datetime.now().isoformat(),
                "total_duration_seconds": time.time() - time.mktime(datetime.fromisoformat(self.start_time).timetuple()) if self.start_time else 0
            },
            "failed_samples": self.failed_samples,
            "successful_results": [r for r in all_results if r is not None]
        }
        
        # 保存详细报告
        report_file = self.output_dirs['reports'] / f"final_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        # 保存简化报告
        summary_file = self.output_dirs['reports'] / "processing_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(report["processing_summary"], f, indent=2, ensure_ascii=False)
        
        logger.info(f"📊 最终报告已保存: {report_file}")
        return report

    def run_batch_processing(self):
        """运行批处理任务"""
        logger.info("🚀 开始Emilia数据集批处理...")
        
        self.start_time = datetime.now().isoformat()
        
        try:
            # 1. 在服务器上下载和设置数据
            self.download_and_setup_data()
            
            # 2. 初始化模型
            self.initialize_models()
            
            # 3. 加载数据集元数据
            dataset_metadata = self.load_dataset_metadata()
            self.total_count = len(dataset_metadata)
            
            # 3. 检查是否从检查点恢复
            checkpoint_data = self.load_checkpoint()
            start_idx = 0
            all_results = []
            
            if checkpoint_data:
                start_idx = checkpoint_data['processed_count']
                self.processed_count = checkpoint_data['processed_count']
                self.failed_samples = checkpoint_data['failed_samples']
                all_results = checkpoint_data.get('processed_results', [])
                logger.info(f"🔄 从第 {start_idx} 个样本开始恢复处理")
            
            # 4. 批处理循环带进度条
            logger.info(f"🚀 开始处理 {self.total_count} 个样本...")
            
            # 创建总体进度条
            with tqdm(total=self.total_count, desc="处理样本", 
                     unit="样本", ncols=100, 
                     bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]") as pbar:
                
                # 设置初始进度
                if start_idx > 0:
                    pbar.update(start_idx)
                    self.processed_count = start_idx
                
                for batch_start in range(start_idx, self.total_count, self.batch_size):
                    batch_end = min(batch_start + self.batch_size, self.total_count)
                    batch_data = dataset_metadata[batch_start:batch_end]
                    
                    # 批次进度条
                    batch_desc = f"批次 {batch_start//self.batch_size + 1}"
                    
                    batch_results = []
                    for sample_data in tqdm(batch_data, desc=batch_desc, leave=False, ncols=80):
                        result = self.process_sample(sample_data)
                        if result:
                            batch_results.append(result)
                        
                        self.processed_count += 1
                        pbar.update(1)
                        
                        # 更新进度条描述
                        success_rate = len([r for r in all_results + batch_results if r]) / self.processed_count * 100
                        pbar.set_postfix({
                            '成功率': f'{success_rate:.1f}%',
                            '失败': len(self.failed_samples)
                        })
                
                    all_results.extend(batch_results)
                    
                    # 保存检查点
                    if batch_start % (self.checkpoint_interval * self.batch_size) == 0:
                        self.save_checkpoint(batch_start // self.batch_size, all_results)
                        pbar.set_description(f"处理样本 (检查点已保存)")
            
            # 5. 生成最终报告
            final_report = self.generate_final_report(all_results)
            
            logger.info("✅ 批处理任务完成！")
            logger.info(f"📊 处理结果: {final_report['processing_summary']['successful_samples']}/{final_report['processing_summary']['total_samples']} 成功")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ 批处理任务失败: {e}")
            traceback.print_exc()
            return False


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Emilia数据集批处理脚本")
    
    parser.add_argument("--input_data_path", type=str, required=True,
                       help="输入数据路径（Emilia数据集路径）")
    parser.add_argument("--output_base_path", type=str, required=True,
                       help="输出基础路径")
    parser.add_argument("--batch_size", type=int, default=100,
                       help="批处理大小 (默认: 100)")
    parser.add_argument("--device", type=str, default="cuda",
                       help="计算设备 (默认: cuda)")
    parser.add_argument("--resume", action="store_true",
                       help="从检查点恢复处理")
    parser.add_argument("--checkpoint_interval", type=int, default=50,
                       help="检查点保存间隔（批次数） (默认: 50)")
    parser.add_argument("--log_level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="日志级别 (默认: INFO)")
    parser.add_argument("--log_file", type=str, default=None,
                       help="日志文件路径（可选）")
    
    return parser.parse_args()


def main():
    """主函数"""
    # 解析命令行参数
    args = parse_arguments()
    
    # 设置日志
    if args.log_file:
        log_file = Path(args.output_base_path) / "logs" / args.log_file
        log_file.parent.mkdir(parents=True, exist_ok=True)
    else:
        log_file = None
    
    setup_logging(args.log_level, str(log_file) if log_file else None)
    
    # 打印启动信息
    logger.info("=" * 80)
    logger.info("🚀 Emilia数据集批处理任务启动")
    logger.info("=" * 80)
    logger.info(f"📁 输入数据路径: {args.input_data_path}")
    logger.info(f"📁 输出基础路径: {args.output_base_path}")
    logger.info(f"📦 批处理大小: {args.batch_size}")
    logger.info(f"🖥️  计算设备: {args.device}")
    logger.info(f"🔄 恢复模式: {'启用' if args.resume else '禁用'}")
    logger.info(f"💾 检查点间隔: {args.checkpoint_interval} 批次")
    logger.info("=" * 80)
    
    try:
        # 创建批处理器
        processor = EmiliaBatchProcessor(
            input_data_path=args.input_data_path,
            output_base_path=args.output_base_path,
            batch_size=args.batch_size,
            device=args.device,
            resume_from_checkpoint=args.resume,
            checkpoint_interval=args.checkpoint_interval
        )
        
        # 运行批处理
        success = processor.run_batch_processing()
        
        if success:
            logger.info("🎉 批处理任务成功完成！")
            print("\n" + "=" * 80)
            print("🎉 Emilia数据集批处理任务成功完成！")
            print("=" * 80)
            print(f"📁 输出文件位置: {args.output_base_path}")
            print("📊 生成的训练数据:")
            print("  - mels/*_mel_original_vevo.npy (源音频Vevo兼容mel频谱)")
            print("  - mels/*_mel_neutral_vevo.npy (中性变换mel频谱)")
            print("  - emotion_features/*_ev2.npz (源音频情感表征)")
            print("  - reports/*.json (处理报告)")
            print("  - checkpoints/*.json (检查点文件)")
            print("\n🎯 正式版本专为训练优化，不生成音频文件")
            print("=" * 80)
            return 0
        else:
            logger.error("❌ 批处理任务失败")
            return 1
            
    except KeyboardInterrupt:
        logger.info("⚠️ 用户中断任务")
        return 130
    except Exception as e:
        logger.error(f"❌ 批处理任务异常: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
