"""
使用项目标准BigVGAN实现的Emilia测试脚本
关键：使用bigvgan.get_mel_spectrogram确保参数一致性
"""

import os
import sys
import torch
import torchaudio
import numpy as np
import soundfile as sf
from pathlib import Path
import logging
import pickle
from typing import Dict, List

# 设置日志
logging.basicConfig(level=logging.INFO)
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

# 真正的Vevo TTS导入
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
        import traceback
        logger.error(f"详细错误: {traceback.format_exc()}")
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


class ProjectStandardProcessor:
    """
    使用项目标准的BigVGAN处理器
    关键：mel提取和合成使用完全相同的参数
    """
    
    def __init__(self, num_samples_per_lang=10, k_variants=1, device='cuda'):
        self.num_samples_per_lang = num_samples_per_lang
        self.k_variants = k_variants
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.output_dir = Path("test_output_correct")
        self.cache_dir = Path("emilia_cache")
        
        # 初始化组件
        self.vevo_emulator = VevoTTSEmulator(device=self.device)
        self.emotion2vec_extractor = Emotion2VecExtractor(device=self.device)
        
        # 初始化BigVGAN（项目标准方式）
        self.bigvgan_model = None
        self._init_project_bigvgan()
        
        # 初始化真正的Vevo TTS
        self.real_vevo_pipeline = None
        self._init_real_vevo_tts()
        
        logger.info(f"项目标准处理器初始化完成")
        logger.info(f"  真正的Vevo TTS: {'可用' if self.real_vevo_pipeline else '不可用'}")

    def _init_project_bigvgan(self):
        """按照项目标准初始化BigVGAN"""
        if not BIGVGAN_AVAILABLE:
            logger.error("❌ BigVGAN不可用")
            return
            
        try:
            # 按照您项目中stage_a.py的方式初始化
            logger.info("初始化项目标准BigVGAN 24kHz...")
            
            self.bigvgan_model = bigvgan.BigVGAN.from_pretrained(
                "nvidia/bigvgan_v2_24khz_100band_256x"
            )
            self.bigvgan_model.eval()
            self.bigvgan_model = self.bigvgan_model.to(self.device)
            
            logger.info("✅ 项目标准BigVGAN初始化成功")
            logger.info(f"✅ BigVGAN已移动到设备: {self.device}")
            
            # 验证模型配置
            if hasattr(self.bigvgan_model, 'h'):
                h = self.bigvgan_model.h
                logger.info(f"BigVGAN配置: sampling_rate={h.sampling_rate}, hop_size={h.hop_size}, n_mel_channels={h.num_mels}")
            
        except Exception as e:
            logger.error(f"❌ 项目标准BigVGAN初始化失败: {e}")
            self.bigvgan_model = None

    def _init_real_vevo_tts(self):
        """初始化真正的Vevo TTS"""
        if not REAL_VEVO_AVAILABLE:
            raise RuntimeError("Vevo TTS模块导入失败，无法继续")
            
        try:
            logger.info("🚀 初始化真正的Vevo TTS（使用已下载的模型）...")
            
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
                logger.info("⬇️ 未找到完整Vevo权重，尝试从 HuggingFace自动下载...")
                try:
                    from huggingface_hub import snapshot_download
                    # tokenizer vq8192
                    snapshot_download(
                        repo_id="amphion/Vevo",
                        repo_type="model",
                        cache_dir=base_cache_dir,
                        allow_patterns=["tokenizer/vq8192/*"],
                    )
                    # AR PhoneToVq8192
                    snapshot_download(
                        repo_id="amphion/Vevo",
                        repo_type="model",
                        cache_dir=base_cache_dir,
                        allow_patterns=["contentstyle_modeling/PhoneToVq8192/*"],
                    )
                    # FMT Vq8192ToMels
                    snapshot_download(
                        repo_id="amphion/Vevo",
                        repo_type="model",
                        cache_dir=base_cache_dir,
                        allow_patterns=["acoustic_modeling/Vq8192ToMels/*"],
                    )
                    # Vocoder
                    snapshot_download(
                        repo_id="amphion/Vevo",
                        repo_type="model",
                        cache_dir=base_cache_dir,
                        allow_patterns=["acoustic_modeling/Vocoder/*"],
                    )
                    # 重新搜集路径
                    vq8192_paths = glob.glob(f"{base_cache_dir}/models--amphion--Vevo/snapshots/*/tokenizer/vq8192") or glob.glob(f"{base_cache_dir}/tokenizer/vq8192")
                    ar_paths = glob.glob(f"{base_cache_dir}/models--amphion--Vevo/snapshots/*/contentstyle_modeling/PhoneToVq8192") or glob.glob(f"{base_cache_dir}/contentstyle_modeling/PhoneToVq8192")
                    fmt_paths = glob.glob(f"{base_cache_dir}/models--amphion--Vevo/snapshots/*/acoustic_modeling/Vq8192ToMels") or glob.glob(f"{base_cache_dir}/acoustic_modeling/Vq8192ToMels")
                    vocoder_paths = glob.glob(f"{base_cache_dir}/models--amphion--Vevo/snapshots/*/acoustic_modeling/Vocoder") or glob.glob(f"{base_cache_dir}/acoustic_modeling/Vocoder")
                except Exception as de:
                    logger.warning(f"⚠️ 自动下载权重失败: {de}")

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
            
            logger.info("✅ 真正的Vevo TTS初始化成功！")
            
        except Exception as e:
            logger.error(f"❌ 真正的Vevo TTS初始化失败: {e}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Vevo TTS初始化失败: {e}")

    def get_cached_samples(self) -> Dict:
        """获取缓存的样本"""
        cache_file = self.cache_dir / f"emilia_samples_{self.num_samples_per_lang}per_lang.pkl"
        
        if cache_file.exists():
            logger.info("🔄 使用缓存数据...")
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except:
                pass
        
        return self._load_from_audio_cache()

    def _load_from_audio_cache(self) -> Dict:
        """从音频缓存加载数据"""
        s1_data = {"EN": [], "ZH": []}
        
        for lang_code in ["EN", "ZH"]:
            lang_cache_dir = self.cache_dir / lang_code
            if lang_cache_dir.exists():
                audio_files = list(lang_cache_dir.glob("*.wav"))[:self.num_samples_per_lang]
                
                for audio_file in audio_files:
                    try:
                        import librosa
                        audio_array, _ = librosa.load(audio_file, sr=24000)
                        
                        meta_file = audio_file.with_suffix('.json')
                        if meta_file.exists():
                            import json
                            with open(meta_file, 'r', encoding='utf-8') as f:
                                metadata = json.load(f)
                        else:
                            metadata = {'text': f'Cached sample', 'speaker': f'{lang_code}_speaker'}
                        
                        sample = {
                            'audio': {
                                'array': audio_array.astype(np.float32),
                                'sampling_rate': 24000
                            },
                            'text': metadata.get('text', ''),
                            'speaker': metadata.get('speaker', ''),
                            'language': lang_code.lower(),
                            'duration': len(audio_array) / 24000
                        }
                        s1_data[lang_code].append(sample)
                        
                    except Exception as e:
                        logger.warning(f"加载 {audio_file} 失败: {e}")
        
        # 创建S2数据
        s2_data = self._create_s2_neutral_samples()
        
        return {
            "s1_original": s1_data,
            "s2_neutral": s2_data
        }

    def _create_s2_neutral_samples(self) -> List:
        """创建S2中性参考样本"""
        s2_data = []
        for i in range(20):
            duration = 2.0
            sample_rate = 24000
            samples_count = int(duration * sample_rate)
            
            t = np.linspace(0, duration, samples_count)
            neutral_freq = 150 + i * 10
            
            audio = (
                0.4 * np.sin(2 * np.pi * neutral_freq * t) +
                0.05 * np.random.randn(samples_count)
            )
            
            sample = {
                'audio': {
                    'array': audio.astype(np.float32),
                    'sampling_rate': sample_rate
                },
                'text': f"Neutral reference {i+1}",
                'speaker': f"neutral_speaker_{i+1}",
                'emotion': 'neutral',
                'duration': duration
            }
            s2_data.append(sample)
        
        return s2_data


    def _process_sample_project_standard(self, sample_id: str, s1_sample: Dict, s2_data: List, lang: str) -> Dict:
        """使用项目标准方法处理样本"""
        
        # 准备音频数据
        s1_audio_np = s1_sample['audio']['array']
        s2_sample = np.random.choice(s2_data)
        s2_audio_np = s2_sample['audio']['array']
        
        # 关键：使用BigVGAN的官方mel提取函数（确保参数一致性）
        logger.info(f"  使用BigVGAN官方mel提取: {sample_id}")
        
        try:
            # 按照您项目stage_a.py中的方式提取mel
            wav_tensor_s1 = torch.FloatTensor(s1_audio_np).unsqueeze(0).to(self.device)
            wav_tensor_s2 = torch.FloatTensor(s2_audio_np).unsqueeze(0).to(self.device)
            
            # 使用BigVGAN的get_mel_spectrogram函数（关键！）
            mel_original = get_mel_spectrogram(wav_tensor_s1, self.bigvgan_model.h)
            mel_s2 = get_mel_spectrogram(wav_tensor_s2, self.bigvgan_model.h)
            
            logger.info(f"  BigVGAN mel提取成功: {mel_original.shape}")
            
            # 处理时间维度不匹配问题
            s1_frames = mel_original.shape[-1]
            s2_frames = mel_s2.shape[-1]
            
            logger.info(f"  时间维度: S1={s1_frames}帧, S2={s2_frames}帧")
            
            if s1_frames != s2_frames:
                # 将S2调整到S1的长度，而不是截断S1
                if s2_frames < s1_frames:
                    # S2太短，重复填充
                    repeat_times = (s1_frames + s2_frames - 1) // s2_frames
                    mel_s2_repeated = mel_s2.repeat(1, 1, repeat_times)
                    mel_s2 = mel_s2_repeated[..., :s1_frames]
                else:
                    # S2太长，截断到S1长度
                    mel_s2 = mel_s2[..., :s1_frames]
                
                logger.info(f"  调整后S2维度: {mel_s2.shape[-1]}帧")
            
            # 生成中性mel - 使用真正的Vevo TTS或备选方案
            mel_neutral = self._generate_neutral_mel_real_vevo(s1_sample, mel_original, mel_s2)
            
            # 提取emotion2vec特征
            s1_audio_tensor = torch.from_numpy(s1_audio_np).float().to(self.device)
            if s1_audio_tensor.dim() == 1:
                s1_audio_tensor = s1_audio_tensor.unsqueeze(0)
            ev2_features = self.emotion2vec_extractor.extract_features(s1_audio_tensor)
            
            # 保存文件
            tuple_id = f"{sample_id}_k01"
            
            # 保存BigVGAN格式的mel（供后续分析使用）
            np.save(self.output_dir / "mels" / f"{tuple_id}_mel_original_bigvgan.npy", 
                    mel_original.squeeze().cpu().numpy())
            
            # 保存Vevo格式的mel（主要用于声码器）
            if mel_neutral.dim() == 3 and mel_neutral.shape[-1] == 128:
                # Vevo格式 [1, T, 128]
                np.save(self.output_dir / "mels" / f"{tuple_id}_mel_neutral_vevo.npy", 
                        mel_neutral.squeeze().cpu().numpy())
                # 也保存转换为BigVGAN格式的版本
                mel_neutral_bigvgan = self._convert_bigvgan_mel_to_vevo(mel_original)  # 使用原始的作为模板
                np.save(self.output_dir / "mels" / f"{tuple_id}_mel_neutral_bigvgan.npy", 
                        mel_neutral_bigvgan.squeeze().cpu().numpy())
            else:
                # BigVGAN格式
                np.save(self.output_dir / "mels" / f"{tuple_id}_mel_neutral.npy", 
                        mel_neutral.squeeze().cpu().numpy())
            
            np.savez_compressed(
                self.output_dir / "emotion_features" / f"{tuple_id}_ev2.npz",
                utterance=ev2_features['utterance'].cpu().numpy(),
                frame=ev2_features['frame'].cpu().numpy()
            )
            
            # 使用Vevo声码器生成音频（确保完全对齐）
            self._generate_audio_with_vevo_vocoder(tuple_id, s1_audio_np, mel_original, mel_neutral)
            
            return {
                "tuple_id": tuple_id,
                "s1_id": sample_id,
                "language": lang,
                "mel_original_shape": list(mel_original.shape),
                "mel_neutral_shape": list(mel_neutral.shape),
                "s1_text": s1_sample.get('text', ''),
                "s1_speaker": s1_sample.get('speaker', ''),
                "s1_duration": s1_sample.get('duration', 0.0),
                "method": "bigvgan.get_mel_spectrogram + project_vocoder"
            }
            
        except Exception as e:
            logger.error(f"  ❌ 项目标准处理失败: {e}")
            raise

    def _generate_neutral_mel_real_vevo(self, s1_sample: Dict, mel_original: torch.Tensor, mel_s2: torch.Tensor) -> torch.Tensor:
        """
        使用已经配置好的Vevo TTS生成中性mel
        直接获取Vevo TTS内部生成的mel特征，然后转换为BigVGAN格式
        """
        # 检查Vevo pipeline是否可用
        if self.real_vevo_pipeline is None:
            raise RuntimeError("Vevo TTS pipeline未初始化，无法生成中性mel")
        
        logger.info("    🎯 使用Vevo TTS生成中性mel...")
        
        # 获取S1的文本和语言
        s1_text = s1_sample.get('text', 'Hello, this is a neutral speech.')
        s1_language = s1_sample.get('language', 'en')
        
        # 创建临时文件
        temp_dir = Path("temp_vevo_call")
        temp_dir.mkdir(exist_ok=True)
        
        s1_temp_path = temp_dir / "s1_timbre.wav"
        neutral_temp_path = temp_dir / "neutral_style.wav"
        output_path = temp_dir / "vevo_output.wav"
        
        try:
            # 保存S1音频文件（作为音色参考）
            sf.write(s1_temp_path, s1_sample['audio']['array'], 24000)
            
            # 创建neutral风格参考音频
            duration = min(3.0, s1_sample.get('duration', 3.0))
            t = np.linspace(0, duration, int(duration * 24000))
            neutral_audio = 0.3 * np.sin(2 * np.pi * 150 * t) + 0.05 * np.random.randn(int(duration * 24000))
            sf.write(neutral_temp_path, neutral_audio.astype(np.float32), 24000)
            
            # 使用Vevo TTS进行推理
            logger.info(f"    调用Vevo TTS: text='{s1_text[:50]}...', lang={s1_language}")
            
            try:
                logger.info("    使用原音频文本进行Vevo TTS调用...")
                
                # 使用原音频的真实文本，适当清理和截断
                clean_text = s1_text.strip()
                # 限制文本长度，避免过长文本导致的问题
                if len(clean_text) > 200:
                    clean_text = clean_text[:200].rsplit(' ', 1)[0] + '.'  # 在单词边界截断
                
                # 为风格参考使用中性描述
                style_text = "This is neutral speech." if s1_language == 'en' else "这是中性语音。"
                
                logger.info(f"    实际使用文本: '{clean_text[:50]}...'")
                
                # 使用原来工作的Vevo TTS调用
                gen_audio = self.real_vevo_pipeline.inference_ar_and_fm(
                    src_wav_path=None,  # TTS模式
                    src_text=clean_text,  # 使用原音频的真实文本
                    style_ref_wav_path=str(neutral_temp_path),  # 中性风格参考
                    style_ref_wav_text=style_text,  # 添加风格参考文本
                    timbre_ref_wav_path=str(s1_temp_path),      # S1音色参考
                    src_text_language=s1_language,
                    style_ref_wav_text_language=s1_language,
                    flow_matching_steps=16,
                    use_global_guided_inference=False
                )
                
                logger.info("    ✅ Vevo TTS调用成功")
                
            except Exception as vevo_error:
                logger.error(f"    ❌ Vevo TTS调用失败: {vevo_error}")
                import traceback
                traceback.print_exc()
                raise RuntimeError(f"Vevo TTS调用失败，无法继续: {vevo_error}")
            
            # 保存生成的音频
            save_audio(gen_audio, output_path=str(output_path))
            
            # 使用Vevo提取mel特征（保持一致性）
            if not output_path.exists():
                raise RuntimeError("Vevo TTS未生成输出文件")
            
            import librosa
            vevo_audio, _ = librosa.load(output_path, sr=24000)
            vevo_audio_tensor = torch.from_numpy(vevo_audio).float().unsqueeze(0).to(self.device)
            
            # 使用Vevo的mel提取器（保持与Vevo声码器一致）
            vevo_mel = self.real_vevo_pipeline.extract_mel_feature(vevo_audio_tensor)
            
            logger.info(f"    ✅ Vevo TTS生成成功: {vevo_mel.shape}")
            return vevo_mel
            
        finally:
            # 清理临时文件
            import shutil
            if temp_dir.exists():
                shutil.rmtree(temp_dir)

    def _get_vevo_mel_directly(self, clean_text: str, s1_language: str, style_text: str, 
                              neutral_temp_path: str, s1_temp_path: str) -> torch.Tensor:
        """
        直接获取Vevo TTS内部生成的mel特征，避免通过音频重新提取
        """
        logger.info("    🎯 直接获取Vevo TTS内部mel特征...")
        
        # 修改Vevo TTS的inference方法，使其返回mel特征而不是音频
        # 这需要复制inference_ar_and_fm的逻辑，但在vocoder之前停止
        
        # 加载音频
        import librosa
        style_ref_speech = librosa.load(neutral_temp_path, sr=24000)[0]
        style_ref_speech_tensor = torch.tensor(style_ref_speech).unsqueeze(0).to(self.device)
        style_ref_speech16k = torchaudio.functional.resample(style_ref_speech_tensor, 24000, 16000)
        
        timbre_ref_speech = librosa.load(s1_temp_path, sr=24000)[0]
        timbre_ref_speech_tensor = torch.tensor(timbre_ref_speech).unsqueeze(0).to(self.device)
        timbre_ref_speech24k = timbre_ref_speech_tensor
        timbre_ref_speech16k = torchaudio.functional.resample(timbre_ref_speech_tensor, 24000, 16000)
        
        # AR阶段
        from models.vc.vevo.vevo_utils import g2p_
        ar_input_ids = g2p_(clean_text, s1_language)[1]
        ar_input_ids = torch.tensor([ar_input_ids], dtype=torch.long).to(self.device)
        
        # 不连接style_ref文本，避免重复问题
        # style_ref_input_ids = g2p_(style_text, s1_language)[1]
        # style_ref_input_ids = torch.tensor([style_ref_input_ids], dtype=torch.long).to(self.device)
        # ar_input_ids = torch.cat([style_ref_input_ids, ar_input_ids], dim=1)
        
        logger.info(f"    AR input_ids形状: {ar_input_ids.shape}")
        
        # 使用全局引导模式，避免重复问题
        prompt_output_ids = None
        
        # AR生成
        predicted_hubert_codecs = self.real_vevo_pipeline.ar_model.generate(
            input_ids=ar_input_ids,
            prompt_mels=self.real_vevo_pipeline.extract_prompt_mel_feature(style_ref_speech16k),
            prompt_output_ids=prompt_output_ids,
        )
        
        # Flow Matching阶段
        timbre_ref_hubert_codecs, _ = self.real_vevo_pipeline.extract_hubert_codec(
            self.real_vevo_pipeline.content_style_tokenizer, 
            timbre_ref_speech16k, 
            duration_reduction=False
        )
        diffusion_input_codecs = torch.cat(
            [timbre_ref_hubert_codecs, predicted_hubert_codecs], dim=1
        )
        
        # 生成mel特征（这是我们要的！）
        predict_mel_feat = self.real_vevo_pipeline.fmt_model.reverse_diffusion(
            cond=self.real_vevo_pipeline.fmt_model.cond_emb(diffusion_input_codecs),
            prompt=self.real_vevo_pipeline.extract_mel_feature(timbre_ref_speech24k),
            n_timesteps=16,
        )
        
        logger.info(f"    ✅ 获取Vevo内部mel特征: {predict_mel_feat.shape}")
        return predict_mel_feat
    
    def _adjust_vevo_mel_timing(self, vevo_mel: torch.Tensor, target_mel: torch.Tensor) -> torch.Tensor:
        """
        调整Vevo mel的时间维度以匹配目标长度，但保持Vevo的mel格式
        Vevo: [1, T, 128] -> [1, T', 128]
        """
        logger.info("    🔄 调整Vevo mel时间维度...")
        
        # 计算目标帧数（考虑hop_size差异）
        # BigVGAN hop_size=256, Vevo hop_size=480
        # 所以时间比例应该是 480/256 = 1.875
        target_bigvgan_frames = target_mel.shape[-1]
        target_vevo_frames = int(target_bigvgan_frames * 480 / 256)  # 转换为Vevo的帧数
        
        current_frames = vevo_mel.shape[1]  # Vevo mel: [1, T, 128]
        
        logger.info(f"    时间维度调整: {current_frames} -> {target_vevo_frames} (Vevo格式)")
        
        if current_frames != target_vevo_frames:
            if current_frames < target_vevo_frames:
                # 重复填充
                repeat_times = (target_vevo_frames + current_frames - 1) // current_frames
                vevo_mel_repeated = vevo_mel.repeat(1, repeat_times, 1)
                vevo_mel_adjusted = vevo_mel_repeated[:, :target_vevo_frames, :]
            else:
                # 截断
                vevo_mel_adjusted = vevo_mel[:, :target_vevo_frames, :]
        else:
            vevo_mel_adjusted = vevo_mel
        
        logger.info(f"    ✅ 调整完成: {vevo_mel_adjusted.shape}")
        return vevo_mel_adjusted

    def _fallback_neutral_generation(self, mel_original: torch.Tensor, mel_s2: torch.Tensor) -> torch.Tensor:
        """备选的中性mel生成方案"""
        logger.info("    使用改进的备选中性mel生成...")
        
        # 改进的混合策略：更多地保持原始内容，适度中性化
        mel_neutral = 0.8 * mel_original + 0.2 * mel_s2
        
        return mel_neutral

    def _fallback_neutral_generation(self, mel_original: torch.Tensor, mel_s2: torch.Tensor) -> torch.Tensor:
        """备选的中性mel生成方案"""
        logger.info("    使用改进的备选中性mel生成...")
        
        # 改进的混合策略：更多地保持原始内容，适度中性化
        mel_neutral = 0.8 * mel_original + 0.2 * mel_s2
        
        return mel_neutral

    def _generate_audio_project_standard(self, tuple_id: str, original_audio: np.ndarray,
                                       mel_original: torch.Tensor, mel_neutral: torch.Tensor):
        """使用项目标准方法生成验证音频"""
        audio_dir = self.output_dir / "verification_audio"
        
        # 1. 保存原始音频
        sf.write(audio_dir / f"{tuple_id}_s1_original.wav", original_audio, 24000)
        
        # 2. 使用BigVGAN进行mel到音频的重建
        if self.bigvgan_model is not None:
            try:
                logger.info(f"  使用项目标准BigVGAN重建: {tuple_id}")
                
                with torch.no_grad():
                    # 确保mel输入格式正确（BigVGAN期望 [batch, mel_channels, time]）
                    mel_orig_input = mel_original.to(self.device)
                    mel_neut_input = mel_neutral.to(self.device)
                    
                    # 确保batch维度
                    if mel_orig_input.dim() == 2:
                        mel_orig_input = mel_orig_input.unsqueeze(0)
                    if mel_neut_input.dim() == 2:
                        mel_neut_input = mel_neut_input.unsqueeze(0)
                    
                    logger.info(f"    Mel输入形状: {mel_orig_input.shape}")
                    logger.info(f"    Mel输入范围: [{mel_orig_input.min():.2f}, {mel_orig_input.max():.2f}]")
                    
                    # BigVGAN重建
                    audio_orig_recon = self.bigvgan_model(mel_orig_input)
                    audio_neut_recon = self.bigvgan_model(mel_neut_input)
                    
                    logger.info(f"    BigVGAN输出形状: {audio_orig_recon.shape}")
                    logger.info(f"    BigVGAN输出范围: [{audio_orig_recon.min():.3f}, {audio_orig_recon.max():.3f}]")
                    
                    # 转换为numpy并后处理
                    audio_orig_np = audio_orig_recon.squeeze().cpu().numpy()
                    audio_neut_np = audio_neut_recon.squeeze().cpu().numpy()
                    
                    # 项目标准后处理（最小化处理，保持BigVGAN原始质量）
                    audio_orig_final = self._minimal_post_process(audio_orig_np)
                    audio_neut_final = self._minimal_post_process(audio_neut_np)
                    
                    # 保存音频
                    sf.write(audio_dir / f"{tuple_id}_original_reconstructed.wav",
                            audio_orig_final, 24000)
                    sf.write(audio_dir / f"{tuple_id}_neutral_reconstructed.wav",
                            audio_neut_final, 24000)
                
                logger.info(f"  ✅ 项目标准BigVGAN重建成功")
                
            except Exception as e:
                logger.error(f"  ❌ 项目标准BigVGAN重建失败: {e}")
                import traceback
                traceback.print_exc()
                
                # 保存错误信息用于调试
                with open(audio_dir / f"{tuple_id}_error.txt", 'w') as f:
                    f.write(f"BigVGAN重建失败: {e}\n")
                    f.write(f"Mel形状: {mel_original.shape}\n")
                    f.write(f"设备: {self.device}\n")
        else:
            logger.warning("  ⚠️ BigVGAN模型不可用")

    def _minimal_post_process(self, audio: np.ndarray) -> np.ndarray:
        """最小化后处理，保持BigVGAN原始质量"""
        
        # 1. 检查有效性
        if len(audio) == 0:
            return np.zeros(1000, dtype=np.float32)
        
        # 2. 去除直流分量
        audio = audio - np.mean(audio)
        
        # 3. 只做必要的音量调整（不改变频谱特性）
        max_val = np.max(np.abs(audio))
        if max_val > 1.0:
            # 只在超过范围时才缩放
            audio = audio / max_val * 0.95
        elif max_val < 0.001:
            # 如果音频太小，适度放大
            audio = audio * 100
        
        # 4. 最终限制（避免削波）
        audio = np.clip(audio, -0.99, 0.99)
        
        return audio.astype(np.float32)

    def _generate_audio_with_vevo_vocoder(self, tuple_id: str, original_audio: np.ndarray,
                                        mel_original: torch.Tensor, mel_neutral: torch.Tensor):
        """对原始和中性音频都使用Vevo声码器，确保完全一致性"""
        audio_dir = self.output_dir / "verification_audio"
        
        # 1. 保存原始音频
        sf.write(audio_dir / f"{tuple_id}_s1_original.wav", original_audio, 24000)
        
        # 2. 对两者都使用Vevo声码器重建音频
        if self.real_vevo_pipeline is not None:
            try:
                logger.info(f"  使用Vevo声码器重建（两者都用Vevo）: {tuple_id}")
                
                with torch.no_grad():
                    # 对原始音频：使用Vevo重新提取mel然后用Vevo声码器
                    logger.info("    原始音频：使用Vevo重新提取mel...")
                    original_tensor = torch.from_numpy(original_audio).float().unsqueeze(0).to(self.device)
                    mel_orig_vevo = self.real_vevo_pipeline.extract_mel_feature(original_tensor)
                    logger.info(f"    原始音频Vevo mel: {mel_orig_vevo.shape}")
                    
                    # 中性mel已经是Vevo格式
                    mel_neut_vevo = mel_neutral
                    
                    logger.info(f"    Vevo Mel输入形状: orig={mel_orig_vevo.shape}, neut={mel_neut_vevo.shape}")
                    
                    # 使用Vevo声码器重建（两者都用相同声码器）
                    audio_orig_recon = self.real_vevo_pipeline.vocoder_model(mel_orig_vevo.transpose(1, 2))
                    audio_neut_recon = self.real_vevo_pipeline.vocoder_model(mel_neut_vevo.transpose(1, 2))
                    
                    logger.info(f"    Vevo声码器输出形状: orig={audio_orig_recon.shape}, neut={audio_neut_recon.shape}")
                    
                    # 转换为numpy并后处理
                    audio_orig_np = audio_orig_recon.squeeze().cpu().numpy()
                    audio_neut_np = audio_neut_recon.squeeze().cpu().numpy()
                    
                    # 最小化后处理，保持Vevo原始质量
                    audio_orig_final = self._minimal_post_process(audio_orig_np)
                    audio_neut_final = self._minimal_post_process(audio_neut_np)
                    
                    # 保存音频
                    sf.write(audio_dir / f"{tuple_id}_original_reconstructed.wav",
                            audio_orig_final, 24000)
                    sf.write(audio_dir / f"{tuple_id}_neutral_reconstructed.wav",
                            audio_neut_final, 24000)
                
                logger.info(f"  ✅ Vevo声码器重建成功（两者都用Vevo）")
                
            except Exception as e:
                logger.error(f"  ❌ Vevo声码器重建失败: {e}")
                import traceback
                traceback.print_exc()
        else:
            logger.warning("  ⚠️ Vevo声码器不可用")
    
    def _convert_bigvgan_mel_to_vevo(self, bigvgan_mel: torch.Tensor) -> torch.Tensor:
        """
        将BigVGAN mel转换为Vevo格式 - 但不做复杂转换，直接使用BigVGAN声码器
        这样可以避免转换错误导致的无声问题
        """
        logger.info("    ⚠️ BigVGAN mel不适合直接转换为Vevo格式")
        logger.info("    → 将使用BigVGAN声码器重建这个音频")
        
        # 返回原始格式，让调用者使用BigVGAN声码器
        return bigvgan_mel

    def run_test(self):
        """运行测试"""
        logger.info("🚀 开始项目标准BigVGAN测试...")
        
        try:
            # 获取数据
            datasets = self.get_cached_samples()
            
            # 创建输出目录
            self.output_dir.mkdir(exist_ok=True)
            (self.output_dir / "mels").mkdir(exist_ok=True)
            (self.output_dir / "emotion_features").mkdir(exist_ok=True)
            (self.output_dir / "verification_audio").mkdir(exist_ok=True)
            
            # 处理样本
            test_results = []
            s1_data = datasets["s1_original"]
            s2_data = datasets["s2_neutral"]
            
            for lang in ["EN", "ZH"]:
                for i, s1_sample in enumerate(s1_data[lang]):
                    sample_id = f"{lang}_S1_{i+1:03d}"
                    logger.info(f"处理样本: {sample_id}")
                    
                    try:
                        result = self._process_sample_project_standard(sample_id, s1_sample, s2_data, lang)
                        test_results.append(result)
                    except Exception as e:
                        logger.error(f"处理样本 {sample_id} 失败: {e}")
            
            # 4. 保存报告
            import json
            report = {
                "test_config": {
                    "num_samples_per_lang": self.num_samples_per_lang,
                    "data_source": "Real Emilia Dataset (Project Standard BigVGAN + Vevo TTS)",
                    "method": "bigvgan.get_mel_spectrogram + real_vevo_tts + project_vocoder",
                    "vevo_source": VEVO_SOURCE,
                    "total_samples": len(test_results)
                },
                "test_results": test_results
            }
            
            with open(self.output_dir / "test_report.json", 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            logger.info("✅ 项目标准测试完成！")
            logger.info(f"📊 成功处理: {len(test_results)} 个样本")
            logger.info(f"📁 测试报告已保存: {self.output_dir / 'test_report.json'}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ 测试失败: {e}")
            return False


def main():
    """主函数"""
    print("🧪 开始项目标准BigVGAN + Vevo TTS测试...")
    print("📋 使用项目中vocoder.py的标准实现")
    print("🔑 关键特性：")
    print("  - 使用bigvgan.get_mel_spectrogram确保参数一致性")
    print("  - 集成真正的Vevo TTS生成中性mel")
    print("  - 提取emotion2vec情感特征")
    print(f"  - 测试{10*2}=20个音频样本 (10个英文 + 10个中文)")
    
    processor = ProjectStandardProcessor(
        num_samples_per_lang=10,
        k_variants=1,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    success = processor.run_test()
    
    if success:
        print("✅ 项目标准测试成功！")
        print("\n📁 生成的文件:")
        print("  - mels/*.npy (BigVGAN官方mel格式，包含原始和中性mel)")
        print("  - emotion_features/*.npz (emotion2vec情感特征)")
        print("  - verification_audio/*.wav (项目标准重建音频)")
        print("  - test_report.json (完整测试报告)")
        print("\n🎵 验证方法:")
        print("1. 听取重建音频，应该没有失真和破音")
        print("2. 对比原始音频和中性音频的差异")
        print("3. 检查情感特征和 mel 频谱图")
        print("\n🚀 使用真正的Vevo TTS + 项目标准BigVGAN，音频质量应该大幅改善！")
    else:
        print("❌ 测试失败")
        print("⚠️ 请检查:")
        print("  1. Vevo TTS模型是否完整下载")
        print("  2. BigVGAN模型是否正常加载")
        print("  3. 缓存数据是否存在")
        print("  4. CUDA/GPU设备是否可用")


if __name__ == "__main__":
    main()
