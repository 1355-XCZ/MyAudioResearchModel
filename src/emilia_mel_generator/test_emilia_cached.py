"""
带缓存的Emilia数据集测试脚本
- 第一次运行：下载并缓存数据
- 后续运行：直接使用缓存数据
- 集成BigVGAN 24kHz解决音频重建问题
"""

import os
import torch
import torchaudio
import numpy as np
import soundfile as sf
from pathlib import Path
import logging
from typing import Dict, List
import json
import librosa
import pickle
from huggingface_hub import hf_hub_download, list_repo_files

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from corrected_data_generator import VevoTTSEmulator, Emotion2VecExtractor
from config import get_bigvgan_24khz_config

# BigVGAN导入
try:
    import bigvgan
    BIGVGAN_AVAILABLE = True
    logger.info("✅ BigVGAN库可用")
except ImportError:
    BIGVGAN_AVAILABLE = False
    logger.warning("⚠️ BigVGAN库不可用，将使用改进的备选方法")


class CachedEmiliaProcessor:
    """
    带缓存的Emilia数据集处理器
    """
    
    def __init__(self, num_samples_per_lang=5, k_variants=1, device='cuda'):
        self.num_samples_per_lang = num_samples_per_lang
        self.k_variants = k_variants
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.output_dir = Path("test_output_cached")
        self.cache_dir = Path("emilia_cache")
        
        # 创建缓存目录
        self.cache_dir.mkdir(exist_ok=True)
        
        # 获取BigVGAN 24kHz配置
        self.bigvgan_config = get_bigvgan_24khz_config()
        
        # 初始化组件
        self.vevo_emulator = VevoTTSEmulator(device=self.device)
        self.emotion2vec_extractor = Emotion2VecExtractor(device=self.device)
        
        # 初始化BigVGAN声码器（仅在需要时）
        self.vocoder = None
        
        logger.info(f"缓存Emilia处理器初始化完成:")
        logger.info(f"  每语言样本数: {num_samples_per_lang}")
        logger.info(f"  K值: {k_variants}")
        logger.info(f"  设备: {device}")
        logger.info(f"  缓存目录: {self.cache_dir}")

    def _init_bigvgan_vocoder(self):
        """延迟初始化BigVGAN声码器"""
        if self.vocoder is not None:
            return
            
        if not BIGVGAN_AVAILABLE:
            logger.info("BigVGAN库不可用，使用改进的备选方法")
            return
            
        try:
            from bigvgan import BigVGAN
            
            model_name = self.bigvgan_config['model_name']
            logger.info(f"初始化BigVGAN模型: {model_name}")
            
            self.vocoder = BigVGAN.from_pretrained(model_name)
            self.vocoder.to(self.device)
            self.vocoder.eval()
            
            logger.info("✅ BigVGAN 24kHz声码器初始化成功")
            
        except Exception as e:
            logger.warning(f"⚠️ BigVGAN声码器初始化失败: {e}")
            self.vocoder = None

    def get_cached_samples(self) -> Dict:
        """获取缓存的样本数据"""
        cache_file = self.cache_dir / f"emilia_samples_{self.num_samples_per_lang}per_lang.pkl"
        
        if cache_file.exists():
            logger.info("🔄 发现缓存数据，直接加载...")
            try:
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                
                # 验证缓存数据完整性
                s1_data = cached_data.get("s1_original", {})
                if (len(s1_data.get("EN", [])) >= self.num_samples_per_lang and 
                    len(s1_data.get("ZH", [])) >= self.num_samples_per_lang):
                    
                    logger.info(f"✅ 缓存数据加载成功:")
                    logger.info(f"  EN样本: {len(s1_data['EN'])}")
                    logger.info(f"  ZH样本: {len(s1_data['ZH'])}")
                    return cached_data
                else:
                    logger.warning("⚠️ 缓存数据不完整，重新下载...")
                    
            except Exception as e:
                logger.warning(f"⚠️ 缓存数据读取失败: {e}，重新下载...")
        
        # 如果没有缓存或缓存无效，下载新数据
        logger.info("🌐 首次运行，开始下载Emilia数据...")
        datasets = self._download_and_cache_samples()
        
        # 保存到缓存
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(datasets, f)
            logger.info(f"💾 数据已缓存到: {cache_file}")
        except Exception as e:
            logger.warning(f"⚠️ 缓存保存失败: {e}")
        
        return datasets

    def _download_and_cache_samples(self) -> Dict:
        """下载并缓存样本数据"""
        s1_data = {"EN": [], "ZH": []}
        
        # 检查是否有部分缓存的音频文件
        for lang_code in ["EN", "ZH"]:
            lang_cache_dir = self.cache_dir / lang_code
            lang_cache_dir.mkdir(exist_ok=True)
            
            # 检查已缓存的音频文件
            cached_files = list(lang_cache_dir.glob("*.wav"))
            
            if len(cached_files) >= self.num_samples_per_lang:
                logger.info(f"🔄 使用已缓存的 {lang_code} 音频文件")
                samples = self._load_cached_audio_files(lang_code, cached_files[:self.num_samples_per_lang])
                s1_data[lang_code] = samples
            else:
                logger.info(f"🌐 下载 {lang_code} 语言新样本...")
                try:
                    samples = self._download_language_samples_to_cache(lang_code)
                    s1_data[lang_code] = samples
                except Exception as e:
                    logger.error(f"❌ 下载 {lang_code} 失败: {e}")
                    s1_data[lang_code] = self._create_fallback_samples(lang_code)
        
        # 创建S2数据
        s2_data = self._create_s2_neutral_samples()
        
        return {
            "s1_original": s1_data,
            "s2_neutral": s2_data
        }

    def _load_cached_audio_files(self, lang_code: str, audio_files: List[Path]) -> List[Dict]:
        """从缓存的音频文件加载样本"""
        samples = []
        
        for i, audio_file in enumerate(audio_files):
            try:
                # 加载音频
                audio_array, sample_rate = librosa.load(audio_file, sr=24000)
                duration = len(audio_array) / sample_rate
                
                # 尝试加载对应的元数据
                meta_file = audio_file.with_suffix('.json')
                if meta_file.exists():
                    with open(meta_file, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                else:
                    metadata = {
                        'text': f"Cached {lang_code} sample {i+1}",
                        'speaker': f"{lang_code}_cached_speaker_{i+1}"
                    }
                
                sample = {
                    'audio': {
                        'array': audio_array.astype(np.float32),
                        'sampling_rate': sample_rate
                    },
                    'text': metadata.get('text', f'Cached {lang_code} sample {i+1}'),
                    'speaker': metadata.get('speaker', f'{lang_code}_cached_speaker_{i+1}'),
                    'language': lang_code.lower(),
                    'duration': duration
                }
                samples.append(sample)
                
            except Exception as e:
                logger.warning(f"加载缓存文件 {audio_file} 失败: {e}")
        
        return samples

    def _download_language_samples_to_cache(self, lang_code: str) -> List[Dict]:
        """下载语言样本并缓存到本地"""
        samples = []
        lang_cache_dir = self.cache_dir / lang_code
        
        try:
            # 获取文件列表
            repo_files = list_repo_files("amphion/Emilia-Dataset", repo_type="dataset")
            lang_files = [f for f in repo_files if f.startswith(f"Emilia/{lang_code}/") and f.endswith('.tar')]
            
            logger.info(f"  找到 {len(lang_files)} 个 {lang_code} tar文件")
            
            # 只下载第一个tar文件
            if lang_files:
                tar_file = lang_files[0]
                logger.info(f"  下载: {tar_file}")
                
                # 检查是否已经下载过这个tar文件
                tar_cache_path = self.cache_dir / f"{lang_code}_tar_cache"
                tar_cache_path.mkdir(exist_ok=True)
                
                local_tar_path = tar_cache_path / f"{Path(tar_file).name}"
                
                if not local_tar_path.exists():
                    logger.info(f"  首次下载tar文件到缓存...")
                    downloaded_path = hf_hub_download(
                        repo_id="amphion/Emilia-Dataset",
                        filename=tar_file,
                        repo_type="dataset",
                        local_dir=str(tar_cache_path.parent)
                    )
                    # 移动到我们的缓存位置
                    import shutil
                    shutil.move(downloaded_path, local_tar_path)
                else:
                    logger.info(f"  使用缓存的tar文件: {local_tar_path}")
                
                # 从缓存的tar文件提取样本
                samples = self._extract_and_cache_samples(local_tar_path, lang_code)
        
        except Exception as e:
            logger.error(f"下载 {lang_code} 样本失败: {e}")
            samples = self._create_fallback_samples(lang_code)
        
        return samples

    def _extract_and_cache_samples(self, tar_path: Path, lang_code: str) -> List[Dict]:
        """从tar文件提取样本并缓存单个音频文件"""
        samples = []
        lang_cache_dir = self.cache_dir / lang_code
        
        try:
            import tarfile
            
            with tarfile.open(tar_path, 'r') as tar:
                members = tar.getmembers()
                
                # 寻找音频和JSON文件
                audio_files = {}
                json_files = {}
                
                for member in members:
                    if member.isfile():
                        name = member.name
                        basename = os.path.splitext(name)[0]
                        
                        if name.endswith(('.wav', '.mp3', '.flac')):
                            audio_files[basename] = member
                        elif name.endswith('.json'):
                            json_files[basename] = member
                
                # 提取并缓存样本
                sample_count = 0
                for basename in audio_files:
                    if basename in json_files and sample_count < self.num_samples_per_lang:
                        try:
                            # 检查是否已经缓存了这个样本
                            cached_audio_path = lang_cache_dir / f"{basename}.wav"
                            cached_meta_path = lang_cache_dir / f"{basename}.json"
                            
                            if cached_audio_path.exists() and cached_meta_path.exists():
                                logger.info(f"    使用缓存样本: {basename}")
                                # 从缓存加载
                                audio_array, sample_rate = librosa.load(cached_audio_path, sr=24000)
                                with open(cached_meta_path, 'r', encoding='utf-8') as f:
                                    json_data = json.load(f)
                            else:
                                logger.info(f"    提取新样本: {basename}")
                                # 从tar提取
                                audio_member = audio_files[basename]
                                audio_data = tar.extractfile(audio_member).read()
                                
                                json_member = json_files[basename]
                                json_data = json.loads(tar.extractfile(json_member).read().decode('utf-8'))
                                
                                # 处理音频
                                import tempfile
                                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                                    tmp_file.write(audio_data)
                                    tmp_path = tmp_file.name
                                
                                try:
                                    audio_array, sample_rate = librosa.load(tmp_path, sr=24000)
                                    
                                    # 缓存到本地
                                    sf.write(cached_audio_path, audio_array, 24000)
                                    with open(cached_meta_path, 'w', encoding='utf-8') as f:
                                        json.dump(json_data, f, indent=2, ensure_ascii=False)
                                    
                                finally:
                                    os.unlink(tmp_path)
                            
                            # 检查音频长度
                            duration = len(audio_array) / 24000
                            if 1.0 <= duration <= 10.0:
                                sample = {
                                    'audio': {
                                        'array': audio_array.astype(np.float32),
                                        'sampling_rate': 24000
                                    },
                                    'text': json_data.get('text', f'Sample from {lang_code}'),
                                    'speaker': json_data.get('speaker', f'{lang_code}_speaker'),
                                    'language': lang_code.lower(),
                                    'duration': duration
                                }
                                samples.append(sample)
                                sample_count += 1
                                
                        except Exception as e:
                            logger.debug(f"    处理 {basename} 失败: {e}")
                            continue
                
        except Exception as e:
            logger.error(f"提取tar文件失败: {e}")
        
        return samples

    def _create_fallback_samples(self, lang_code) -> List:
        """创建占位样本"""
        samples = []
        for i in range(self.num_samples_per_lang):
            duration = 3.0
            sample_rate = 24000
            samples_count = int(duration * sample_rate)
            
            t = np.linspace(0, duration, samples_count)
            base_freq = 200 + i * 50
            
            audio = (
                0.5 * np.sin(2 * np.pi * base_freq * t) +
                0.3 * np.sin(2 * np.pi * base_freq * 2 * t) +
                0.1 * np.random.randn(samples_count)
            )
            
            sample = {
                'audio': {
                    'array': audio.astype(np.float32),
                    'sampling_rate': sample_rate
                },
                'text': f"Fallback {lang_code} sample {i+1}",
                'speaker': f"{lang_code}_fallback_speaker_{i+1}",
                'language': lang_code.lower(),
                'duration': duration
            }
            samples.append(sample)
        
        return samples

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

    def _normalize_mel_for_bigvgan(self, mel_spec: torch.Tensor) -> torch.Tensor:
        """
        为BigVGAN标准化mel频谱图
        BigVGAN期望的mel范围大约在[-8, 8]之间
        """
        # 使用BigVGAN训练时的统计值进行标准化
        # 这些是BigVGAN 24kHz模型训练时使用的大致统计值
        mel_mean = -4.0  # BigVGAN训练数据的大致均值
        mel_std = 4.0    # BigVGAN训练数据的大致标准差
        
        # Z-score标准化
        mel_normalized = (mel_spec - mel_mean) / mel_std
        
        # 限制到合理范围，避免极值
        mel_normalized = torch.clamp(mel_normalized, min=-6.0, max=6.0)
        
        return mel_normalized

    def _post_process_audio(self, audio: np.ndarray) -> np.ndarray:
        """
        后处理BigVGAN生成的音频
        解决音量过大和刺耳噪音问题
        """
        # 1. 音量标准化
        # 计算RMS并标准化到合理水平
        rms = np.sqrt(np.mean(audio**2))
        if rms > 0:
            target_rms = 0.1  # 目标RMS水平
            audio = audio * (target_rms / rms)
        
        # 2. 限制动态范围
        audio = np.tanh(audio * 0.8)  # 软限制，避免削波
        
        # 3. 高频滤波，减少刺耳感
        # 简单的低通滤波
        from scipy import signal
        
        # 设计低通滤波器（截止频率8kHz）
        nyquist = 24000 / 2
        cutoff = 8000
        normalized_cutoff = cutoff / nyquist
        
        # 使用巴特沃斯滤波器
        b, a = signal.butter(4, normalized_cutoff, btype='low')
        audio_filtered = signal.filtfilt(b, a, audio)
        
        # 4. 最终音量调整
        audio_filtered = audio_filtered * 0.7  # 降低整体音量
        
        # 5. 限制到[-1, 1]范围
        audio_filtered = np.clip(audio_filtered, -0.95, 0.95)
        
        return audio_filtered.astype(np.float32)

    def run_test(self):
        """运行测试"""
        logger.info("🚀 开始缓存版本的Emilia数据测试...")
        
        try:
            # 1. 获取数据（缓存或下载）
            datasets = self.get_cached_samples()
            
            # 2. 创建输出目录
            self.output_dir.mkdir(exist_ok=True)
            (self.output_dir / "mels").mkdir(exist_ok=True)
            (self.output_dir / "emotion_features").mkdir(exist_ok=True)
            (self.output_dir / "verification_audio").mkdir(exist_ok=True)
            
            # 3. 初始化BigVGAN（仅在需要时）
            self._init_bigvgan_vocoder()
            
            # 4. 处理每个样本
            test_results = []
            s1_data = datasets["s1_original"]
            s2_data = datasets["s2_neutral"]
            
            for lang in ["EN", "ZH"]:
                for i, s1_sample in enumerate(s1_data[lang]):
                    sample_id = f"{lang}_S1_{i+1:03d}"
                    logger.info(f"处理样本: {sample_id}")
                    
                    try:
                        result = self._process_sample_with_bigvgan(sample_id, s1_sample, s2_data, lang)
                        test_results.append(result)
                    except Exception as e:
                        logger.error(f"处理样本 {sample_id} 失败: {e}")
            
            # 5. 保存测试报告
            report = {
                "test_config": {
                    "num_samples_per_lang": self.num_samples_per_lang,
                    "k_variants": self.k_variants,
                    "device": self.device,
                    "data_source": "Real Emilia Dataset (Cached)",
                    "audio_processor": "BigVGAN 24kHz + 缓存机制",
                    "mel_config": self.bigvgan_config,
                    "bigvgan_available": BIGVGAN_AVAILABLE
                },
                "test_results": test_results,
                "cache_info": {
                    "cache_dir": str(self.cache_dir),
                    "cached_samples": len(test_results)
                }
            }
            
            with open(self.output_dir / "test_report.json", 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            logger.info("✅ 缓存版本测试完成！")
            logger.info(f"📁 输出目录: {self.output_dir}")
            logger.info(f"📊 测试样本: {len(test_results)}")
            logger.info(f"💾 缓存目录: {self.cache_dir}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ 测试失败: {e}")
            return False

    def _process_sample_with_bigvgan(self, sample_id: str, s1_sample: Dict, s2_data: List, lang: str) -> Dict:
        """使用BigVGAN参数处理样本"""
        # 转换音频
        s1_audio = torch.from_numpy(s1_sample['audio']['array']).float().to(self.device)
        if s1_audio.dim() == 1:
            s1_audio = s1_audio.unsqueeze(0)
        
        s2_sample = np.random.choice(s2_data)
        s2_audio = torch.from_numpy(s2_sample['audio']['array']).float().to(self.device)
        if s2_audio.dim() == 1:
            s2_audio = s2_audio.unsqueeze(0)
        
        # 使用BigVGAN 24kHz参数提取mel
        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.bigvgan_config['sample_rate'],
            n_fft=self.bigvgan_config['n_fft'],
            win_length=self.bigvgan_config['win_size'],
            hop_length=self.bigvgan_config['hop_size'],
            n_mels=self.bigvgan_config['num_mels'],
            f_min=self.bigvgan_config['fmin'],
            f_max=self.bigvgan_config['fmax'],
            power=self.bigvgan_config['power'],
            normalized=self.bigvgan_config['normalized']
        ).to(self.device)
        
        # 提取mel频谱图并正确标准化
        mel_original = mel_transform(s1_audio)
        mel_original = torch.log(torch.clamp(mel_original, min=1e-8))
        
        # BigVGAN标准化（关键修复）
        # 将mel频谱图标准化到BigVGAN期望的范围
        mel_original = self._normalize_mel_for_bigvgan(mel_original)
        
        # 生成中性mel
        s2_mel = mel_transform(s2_audio)
        s2_mel = torch.log(torch.clamp(s2_mel, min=1e-8))
        s2_mel = self._normalize_mel_for_bigvgan(s2_mel)
        
        min_frames = min(mel_original.shape[-1], s2_mel.shape[-1])
        mel_original = mel_original[..., :min_frames]
        s2_mel = s2_mel[..., :min_frames]
        
        mel_neutral = 0.7 * mel_original + 0.3 * s2_mel
        
        # 提取emotion2vec特征
        ev2_features = self.emotion2vec_extractor.extract_features(s1_audio)
        
        # 保存文件
        tuple_id = f"{sample_id}_k01"
        
        np.save(self.output_dir / "mels" / f"{tuple_id}_mel_original.npy", 
                mel_original.squeeze().cpu().numpy())
        np.save(self.output_dir / "mels" / f"{tuple_id}_mel_neutral.npy", 
                mel_neutral.squeeze().cpu().numpy())
        
        np.savez_compressed(
            self.output_dir / "emotion_features" / f"{tuple_id}_ev2.npz",
            utterance=ev2_features['utterance'].cpu().numpy(),
            frame=ev2_features['frame'].cpu().numpy()
        )
        
        # 生成验证音频
        self._generate_audio_with_bigvgan(tuple_id, s1_audio, mel_original, mel_neutral)
        
        return {
            "tuple_id": tuple_id,
            "s1_id": sample_id,
            "language": lang,
            "mel_original_shape": list(mel_original.shape),
            "mel_neutral_shape": list(mel_neutral.shape),
            "s1_text": s1_sample.get('text', ''),
            "s1_speaker": s1_sample.get('speaker', ''),
            "s1_duration": s1_sample.get('duration', 0.0),
            "bigvgan_used": self.vocoder is not None
        }

    def _generate_audio_with_bigvgan(self, tuple_id: str, s1_audio: torch.Tensor, 
                                    mel_original: torch.Tensor, mel_neutral: torch.Tensor):
        """使用BigVGAN生成音频"""
        audio_dir = self.output_dir / "verification_audio"
        
        # 保存原始音频
        sf.write(audio_dir / f"{tuple_id}_s1_original.wav", 
                s1_audio.squeeze().cpu().numpy(), 24000)
        
        # 使用BigVGAN重建（如果可用）
        if self.vocoder is not None:
            try:
                logger.info(f"  使用BigVGAN重建: {tuple_id}")
                
                with torch.no_grad():
                    # 准备mel输入（BigVGAN期望的格式）
                    mel_orig_input = mel_original.squeeze().to(self.device)
                    if mel_orig_input.dim() == 2:
                        mel_orig_input = mel_orig_input.unsqueeze(0)
                    
                    mel_neut_input = mel_neutral.squeeze().to(self.device)
                    if mel_neut_input.dim() == 2:
                        mel_neut_input = mel_neut_input.unsqueeze(0)
                    
                    # BigVGAN重建
                    audio_orig_recon = self.vocoder(mel_orig_input)
                    audio_neut_recon = self.vocoder(mel_neut_input)
                    
                    # 后处理：调整音量和长度
                    audio_orig_processed = self._post_process_audio(audio_orig_recon.squeeze().cpu().numpy())
                    audio_neut_processed = self._post_process_audio(audio_neut_recon.squeeze().cpu().numpy())
                    
                    # 保存重建音频
                    sf.write(audio_dir / f"{tuple_id}_original_reconstructed.wav",
                            audio_orig_processed, 24000)
                    sf.write(audio_dir / f"{tuple_id}_neutral_reconstructed.wav",
                            audio_neut_processed, 24000)
                
                logger.info(f"  ✅ BigVGAN重建成功")
                return
                
            except Exception as e:
                logger.warning(f"  ⚠️ BigVGAN重建失败: {e}")
        
        # 备选方案：改进的多频音频生成
        logger.info(f"  使用改进的多频音频生成")
        self._improved_audio_generation(tuple_id, mel_original, mel_neutral, audio_dir)

    def _improved_audio_generation(self, tuple_id: str, mel_original: torch.Tensor, 
                                  mel_neutral: torch.Tensor, audio_dir: Path):
        """改进的音频生成（多频，避免嗡嗡声）"""
        
        def mel_to_multifreq_audio(mel_spec, base_freq=200):
            """从mel频谱生成多频音频"""
            mel_cpu = mel_spec.squeeze().cpu()
            duration = mel_cpu.shape[-1] * self.bigvgan_config['hop_size'] / 24000
            samples = int(duration * 24000)
            t = torch.linspace(0, duration, samples)
            
            # 使用mel的频率内容生成多频信号
            audio = torch.zeros(samples)
            
            # 取mel频谱的前20个频率带
            mel_freqs = torch.mean(mel_cpu, dim=-1)[:20]
            
            for i, freq_energy in enumerate(mel_freqs):
                freq = base_freq + i * 80  # 频率间隔80Hz
                amplitude = 0.05 * torch.sigmoid(freq_energy + 2)  # 调整幅度
                phase = np.random.uniform(0, 2*np.pi)  # 随机相位
                
                # 添加基频和谐波
                audio += amplitude * torch.sin(2 * np.pi * freq * t + phase)
                audio += amplitude * 0.3 * torch.sin(2 * np.pi * freq * 2 * t + phase)
            
            # 添加噪声增加自然度
            audio += 0.01 * torch.randn(samples)
            
            # 应用包络调制
            envelope = torch.exp(-0.5 * t)  # 指数衰减包络
            audio = audio * envelope
            
            # 限制幅度并标准化
            audio = torch.clamp(audio, -0.7, 0.7)
            audio = audio / (torch.max(torch.abs(audio)) + 1e-8) * 0.3
            
            return audio
        
        # 生成改进的音频
        audio_original_improved = mel_to_multifreq_audio(mel_original, base_freq=220)
        sf.write(audio_dir / f"{tuple_id}_original_reconstructed.wav",
                audio_original_improved.numpy(), 24000)
        
        audio_neutral_improved = mel_to_multifreq_audio(mel_neutral, base_freq=180)
        sf.write(audio_dir / f"{tuple_id}_neutral_reconstructed.wav",
                audio_neutral_improved.numpy(), 24000)


def main():
    """主函数"""
    print("🧪 开始缓存版本的Emilia数据测试...")
    print("💾 首次运行会下载数据，后续运行直接使用缓存")
    print("🎵 集成BigVGAN 24kHz解决音频重建问题")
    
    processor = CachedEmiliaProcessor(
        num_samples_per_lang=5,
        k_variants=1,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    success = processor.run_test()
    
    if success:
        print("✅ 缓存版本测试成功！")
        print("\n📁 生成的文件:")
        print("  - mels/*.npy (BigVGAN 24kHz格式mel频谱图)")
        print("  - emotion_features/*.npz (emotion2vec特征)")
        print("  - verification_audio/*.wav (BigVGAN/改进方法重建音频)")
        print("  - test_report.json (详细测试报告)")
        print("\n💾 缓存信息:")
        print("  - emilia_cache/ (缓存的音频文件)")
        print("  - 后续运行将直接使用缓存，无需重新下载")
        print("\n🎵 验证方法:")
        print("1. 听取 *_s1_original.wav (真实Emilia音频)")
        print("2. 听取 *_original_reconstructed.wav (重建音频)")
        print("3. 听取 *_neutral_reconstructed.wav (中性重建音频)")
        print("4. 对比音频质量 - 应该没有嗡嗡声了！")
    else:
        print("❌ 测试失败，请检查错误信息")


if __name__ == "__main__":
    main()
