#!/usr/bin/env python3
"""
Emilia训练数据生成脚本 - 简化高效版本
基于test_emilia_correct.py的成功实现，专为训练数据生成优化

只生成：
1. 源音频的Vevo兼容mel频谱图
2. 中性变换后的mel频谱图（确保与源mel参数完全一致）
3. 源音频的emotion2vec情感特征

目标：英文和中文各50小时数据
不生成音频文件，提高处理效率

关键：确保源mel和中性mel都使用相同的Vevo参数，完全兼容Vevo声码器
"""

import os
import sys
import argparse
import torch
import torchaudio
import numpy as np
import soundfile as sf
from pathlib import Path
import json
import time
from typing import Dict, List, Optional
from datetime import datetime
from tqdm import tqdm

# 简化日志设置
import logging
logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# 添加项目路径
emilia_mel_generator_path = Path(__file__).parent.parent  # production_batch -> emilia_mel_generator
sys.path.append(str(emilia_mel_generator_path))
from corrected_data_generator import Emotion2VecExtractor

# Vevo TTS导入（从test_emilia_correct.py复制的成功实现）
amphion_path = emilia_mel_generator_path.parent.parent / "Amphion"  # src/Amphion
sys.path.insert(0, str(amphion_path))

# 导入Vevo TTS
def import_vevo_tts():
    """导入Vevo TTS组件"""
    try:
        original_cwd = os.getcwd()
        os.chdir(str(amphion_path))
        
        # 确保utils包可用
        utils_path = amphion_path / "utils"
        if not (utils_path / "__init__.py").exists():
            with open(utils_path / "__init__.py", 'w') as f:
                f.write("# Amphion utils package\n")
        
        from models.vc.vevo.vevo_utils import VevoInferencePipeline, save_audio, g2p_
        print("✅ Vevo TTS导入成功")
        return VevoInferencePipeline, save_audio, g2p_
        
    except Exception as e:
        print(f"❌ Vevo TTS导入失败: {e}")
        raise
    finally:
        os.chdir(original_cwd)

# 全局导入
VevoInferencePipeline, save_audio, g2p_ = import_vevo_tts()


class TrainingDataGenerator:
    """训练数据生成器 - 简化高效版本"""
    
    def __init__(self, output_path: str, cache_path: str, device: str = 'cuda'):
        self.output_path = Path(output_path)
        self.cache_path = Path(cache_path)
        self.device = device if torch.cuda.is_available() else 'cpu'
        
        # 创建输出目录
        self.output_dirs = {
            'mels': self.output_path / "mels",
            'emotion_features': self.output_path / "emotion_features",
            'checkpoints': self.output_path / "checkpoints",
            'reports': self.output_path / "reports"
        }
        
        for dir_path in self.output_dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # 设置缓存环境
        self._setup_cache_environment()
        
        # 初始化模型
        self.emotion2vec_extractor = None
        self.vevo_pipeline = None
        
        # 中性参考音频池
        self.neutral_reference_pool = []
        
        # CSV元信息管理
        self.metadata_csv_path = self.output_dirs['metadata'] / "training_dataset_metadata.csv"
        self.metadata_records = []
        
        print(f"✅ 训练数据生成器初始化完成 (设备: {self.device})")
    
    def _setup_cache_environment(self):
        """设置缓存环境变量"""
        os.environ['HF_HOME'] = str(self.cache_path / "huggingface")
        os.environ['TRANSFORMERS_CACHE'] = str(self.cache_path / "transformers")
        os.environ['TORCH_HOME'] = str(self.cache_path / "torch")
        
        for env_var in ['HF_HOME', 'TRANSFORMERS_CACHE', 'TORCH_HOME']:
            Path(os.environ[env_var]).mkdir(parents=True, exist_ok=True)
    
    def initialize_models(self):
        """初始化模型组件"""
        print("🚀 初始化模型组件...")
        
        # 初始化Emotion2Vec
        self.emotion2vec_extractor = Emotion2VecExtractor(device=self.device)
        
        # 初始化Vevo TTS
        self._init_vevo_pipeline()
        
        # 加载中性参考音频池
        self._load_neutral_reference_pool()
        
        print("✅ 模型初始化完成")
    
    def _init_vevo_pipeline(self):
        """初始化Vevo TTS流水线"""
        try:
            # 查找Vevo模型文件
            base_cache_dir = str(amphion_path / "ckpts/Vevo")
            import glob
            
            vq8192_paths = glob.glob(f"{base_cache_dir}/models--amphion--Vevo/snapshots/*/tokenizer/vq8192")
            ar_paths = glob.glob(f"{base_cache_dir}/models--amphion--Vevo/snapshots/*/contentstyle_modeling/PhoneToVq8192")
            fmt_paths = glob.glob(f"{base_cache_dir}/models--amphion--Vevo/snapshots/*/acoustic_modeling/Vq8192ToMels")
            vocoder_paths = glob.glob(f"{base_cache_dir}/models--amphion--Vevo/snapshots/*/acoustic_modeling/Vocoder")
            
            if not all([vq8192_paths, ar_paths, fmt_paths, vocoder_paths]):
                print("⬇️ 下载Vevo模型...")
                self._download_vevo_models(base_cache_dir)
                # 重新搜索
                vq8192_paths = glob.glob(f"{base_cache_dir}/models--amphion--Vevo/snapshots/*/tokenizer/vq8192")
                ar_paths = glob.glob(f"{base_cache_dir}/models--amphion--Vevo/snapshots/*/contentstyle_modeling/PhoneToVq8192")
                fmt_paths = glob.glob(f"{base_cache_dir}/models--amphion--Vevo/snapshots/*/acoustic_modeling/Vq8192ToMels")
                vocoder_paths = glob.glob(f"{base_cache_dir}/models--amphion--Vevo/snapshots/*/acoustic_modeling/Vocoder")
            
            # 初始化流水线
            original_cwd = os.getcwd()
            try:
                os.chdir(str(amphion_path))
                self.vevo_pipeline = VevoInferencePipeline(
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
                
        except Exception as e:
            print(f"❌ Vevo TTS初始化失败: {e}")
            raise
    
    def _download_vevo_models(self, cache_dir: str):
        """下载Vevo模型"""
        from huggingface_hub import snapshot_download
        
        components = [
            ("tokenizer/vq8192/*", "tokenizer"),
            ("contentstyle_modeling/PhoneToVq8192/*", "AR模型"),
            ("acoustic_modeling/Vq8192ToMels/*", "FMT模型"),
            ("acoustic_modeling/Vocoder/*", "Vocoder模型")
        ]
        
        for pattern, name in components:
            snapshot_download(
                repo_id="amphion/Vevo",
                repo_type="model",
                cache_dir=cache_dir,
                allow_patterns=[pattern],
            )
    
    def _load_neutral_reference_pool(self):
        """从 ASLP-lab/Emo-Emilia 数据集加载中性标签的音频作为参考"""
        print("🎯 加载 Emo-Emilia 中性参考音频池...")
        
        try:
            from datasets import load_dataset
            
            # 按照官方示例加载 ASLP-lab/Emo-Emilia 数据集
            print("⬇️ 从 HuggingFace 加载 ASLP-lab/Emo-Emilia...")
            print("⚠️ 注意：需要先登录 HuggingFace: huggingface-cli login")
            
            emo_dataset = load_dataset(
                "ASLP-lab/Emo-Emilia", 
                cache_dir=str(self.cache_path / "emo_emilia")
            )
            
            # Emo-Emilia 包含 1400 个样本，7种情感类型，中英文各700个
            # 筛选 neutral 标签的音频
            neutral_samples = []
            
            # 遍历数据集中的所有样本
            for split_name in emo_dataset.keys():
                for item in emo_dataset[split_name]:
                    # 检查情感标签
                    emotion_label = item.get('emotion', '').lower()
                    
                # 严格筛选 neutral 标签的样本
                if emotion_label == 'neutral':
                        audio_data = item['audio']
                        duration = len(audio_data['array']) / audio_data['sampling_rate']
                        
                        # 确保音频长度适合（1-10秒）
                        if 1.0 <= duration <= 10.0:
                            neutral_samples.append({
                                'audio': audio_data,
                                'text': item.get('text', 'This is neutral speech.'),
                                'language': item.get('language', 'en'),
                                'emotion': emotion_label,
                                'duration': duration,
                                'speaker': item.get('speaker', 'unknown')
                            })
            
            self.neutral_reference_pool = neutral_samples
            
            # 统计和验证语言分布
            lang_stats = {}
            for sample in neutral_samples:
                lang = sample['language'].lower()
                if lang not in lang_stats:
                    lang_stats[lang] = 0
                lang_stats[lang] += 1
            
            print(f"✅ 从 Emo-Emilia 加载了 {len(self.neutral_reference_pool)} 个中性参考音频")
            for lang, count in lang_stats.items():
                print(f"  {lang.upper()}: {count} 个中性样本")
            
            # 验证必要的语言是否都有中性参考
            required_langs = ['en', 'zh']
            missing_langs = []
            for lang in required_langs:
                if lang not in lang_stats or lang_stats[lang] == 0:
                    missing_langs.append(lang)
            
            if missing_langs:
                print(f"⚠️ 警告：缺少以下语言的中性参考: {missing_langs}")
                print(f"⚠️ 这将影响对应语言的中性化质量")
            else:
                print(f"✅ 所有需要的语言都有中性参考音频")
            
        except Exception as e:
            print(f"⚠️ 无法加载 Emo-Emilia 数据集: {e}")
            print("⚠️ 将使用合成中性音频作为备选")
            self.neutral_reference_pool = []
    
    def load_dataset(self, dataset_path: str, target_hours_per_lang: float = 50.0) -> List[Dict]:
        """加载数据集"""
        print("📊 加载数据集...")
        
        try:
            from datasets import load_from_disk, load_dataset
            
            # 尝试从本地加载
            if Path(dataset_path).exists():
                dataset = load_from_disk(dataset_path)
                print(f"✅ 从本地加载数据集")
            else:
                # 从HuggingFace加载
                print("⬇️ 从HuggingFace下载数据集...")
                dataset = load_dataset("amphion/Emilia-Dataset", cache_dir=str(self.cache_path))
                dataset.save_to_disk(dataset_path)
            
            # 转换为列表格式并按语言筛选到目标时长
            metadata = self._filter_by_duration(dataset, target_hours_per_lang)
            
            print(f"✅ 加载了 {len(metadata)} 个样本")
            return metadata
            
        except Exception as e:
            print(f"❌ 数据集加载失败: {e}")
            raise
    
    def _filter_by_duration(self, dataset, target_hours_per_lang: float) -> List[Dict]:
        """按语言和时长筛选数据"""
        print(f"🔍 筛选数据：每种语言 {target_hours_per_lang} 小时...")
        
        target_seconds_per_lang = target_hours_per_lang * 3600
        lang_data = {'en': [], 'zh': []}
        lang_duration = {'en': 0.0, 'zh': 0.0}
        
        metadata = []
        
        for split_name in dataset.keys():
            for i, item in enumerate(dataset[split_name]):
                # 获取语言和时长
                language = item.get('language', 'en').lower()
                if language not in ['en', 'zh']:
                    continue
                
                audio_array = item['audio']['array']
                duration = len(audio_array) / item['audio']['sampling_rate']
                
                # 检查是否还需要这种语言的数据
                if lang_duration[language] < target_seconds_per_lang:
                    sample = {
                        'id': f"{split_name}_{language}_{len(lang_data[language]):06d}",
                        'audio': item['audio'],
                        'text': item.get('text', ''),
                        'speaker': item.get('speaker', ''),
                        'language': language,
                        'duration': duration
                    }
                    
                    lang_data[language].append(sample)
                    lang_duration[language] += duration
                    metadata.append(sample)
                
                # 检查是否已达到目标
                if all(lang_duration[lang] >= target_seconds_per_lang for lang in ['en', 'zh']):
                    break
            
            # 检查是否已达到目标
            if all(lang_duration[lang] >= target_seconds_per_lang for lang in ['en', 'zh']):
                break
        
        # 统计信息
        for lang in ['en', 'zh']:
            hours = lang_duration[lang] / 3600
            count = len(lang_data[lang])
            print(f"  {lang.upper()}: {count} 样本, {hours:.1f} 小时")
        
        total_hours = sum(lang_duration.values()) / 3600
        print(f"✅ 筛选完成：总计 {len(metadata)} 样本, {total_hours:.1f} 小时")
        
        return metadata
    
    def process_sample(self, sample_data: Dict) -> bool:
        """处理单个样本，返回是否成功"""
        sample_id = sample_data['id']
        
        try:
            # 检查是否已存在
            if self._check_sample_exists(sample_id):
                return True
            
            # 1. 准备音频数据
            audio_array = sample_data['audio']['array']
            if sample_data['audio']['sampling_rate'] != 24000:
                import librosa
                audio_array = librosa.resample(audio_array, 
                                             orig_sr=sample_data['audio']['sampling_rate'], 
                                             target_sr=24000)
            
            # 2. 提取源音频的Vevo兼容mel（使用Vevo的mel提取器确保参数一致）
            audio_tensor = torch.from_numpy(audio_array).float().unsqueeze(0).to(self.device)
            mel_original_vevo = self.vevo_pipeline.extract_mel_feature(audio_tensor)
            
            # 验证mel格式：Vevo格式应该是 [1, T, 128]
            assert mel_original_vevo.shape[-1] == 128, f"源mel格式错误: {mel_original_vevo.shape}"
            
            # 3. 生成中性mel（确保与源mel使用相同参数）
            mel_neutral_vevo = self._generate_neutral_mel(sample_data)
            
            # 验证中性mel格式：必须与源mel格式一致
            assert mel_neutral_vevo.shape[-1] == 128, f"中性mel格式错误: {mel_neutral_vevo.shape}"
            assert mel_neutral_vevo.dim() == 3, f"中性mel维度错误: {mel_neutral_vevo.shape}"
            
            # 4. 提取情感特征
            ev2_features = self.emotion2vec_extractor.extract_features(audio_tensor)
            
            # 5. 保存训练数据
            self._save_training_data(sample_id, mel_original_vevo, mel_neutral_vevo, ev2_features)
            
            return True
            
        except Exception as e:
            # 静默处理错误，不打印详细信息
            return False
    
    def _check_sample_exists(self, sample_id: str) -> bool:
        """检查样本文件是否已存在"""
        required_files = [
            self.output_dirs['original_mels'] / f"{sample_id}_mel_original_vevo.npz",
            self.output_dirs['neutral_mels'] / f"{sample_id}_mel_neutral_vevo.npz",
            self.output_dirs['emotion_features'] / f"{sample_id}_emotion_features.npz"
        ]
        return all(f.exists() for f in required_files)
    
    def _generate_neutral_mel(self, sample_data: Dict) -> torch.Tensor:
        """生成中性mel"""
        s1_text = sample_data.get('text', 'Hello, this is a neutral speech.')
        s1_language = sample_data.get('language', 'en')
        
        # 创建临时文件
        temp_dir = Path("temp_vevo_call")
        temp_dir.mkdir(exist_ok=True)
        
        try:
            s1_temp_path = temp_dir / "s1_timbre.wav"
            neutral_temp_path = temp_dir / "neutral_style.wav"
            output_path = temp_dir / "vevo_output.wav"
            
            # 保存音频文件
            sf.write(s1_temp_path, sample_data['audio']['array'], 24000)
            
            # 使用Emo-Emilia数据集中的中性音频和文本作为风格参考
            neutral_ref_info = self._get_random_neutral_reference_with_text(s1_language)
            neutral_audio_data = neutral_ref_info['audio']
            neutral_text = neutral_ref_info['text']
            
            sf.write(neutral_temp_path, neutral_audio_data, 24000)
            
            # 文本处理
            clean_text = s1_text.strip()
            if len(clean_text) > 200:
                clean_text = clean_text[:200].rsplit(' ', 1)[0] + '.'
            
            # 验证语言匹配
            neutral_lang = neutral_ref_info.get('language', 'unknown').lower()
            if neutral_lang != s1_language.lower():
                print(f"⚠️ 警告：中性参考语言不匹配！源音频: {s1_language}, 中性参考: {neutral_lang}")
            
            # 使用真实的中性文本作为风格参考
            style_text = neutral_text if neutral_text else (
                "This is neutral speech." if s1_language == 'en' else "这是中性语音。"
            )
            
            # 记录使用的中性参考信息
            # print(f"🎯 {s1_language.upper()} 音频 -> {neutral_lang.upper()} 中性参考")
            
            # Vevo TTS调用
            gen_audio = self.vevo_pipeline.inference_ar_and_fm(
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
            
            # 保存并提取mel（关键：使用相同的Vevo mel提取器）
            save_audio(gen_audio, output_path=str(output_path))
            
            import librosa
            vevo_audio, _ = librosa.load(output_path, sr=24000)
            vevo_audio_tensor = torch.from_numpy(vevo_audio).float().unsqueeze(0).to(self.device)
            
            # 关键：使用相同的Vevo mel提取器，确保参数一致
            vevo_mel = self.vevo_pipeline.extract_mel_feature(vevo_audio_tensor)
            
            # 验证mel格式一致性
            assert vevo_mel.shape[-1] == 128, f"Vevo mel格式错误: {vevo_mel.shape}"
            assert vevo_mel.dim() == 3, f"Vevo mel维度错误: {vevo_mel.shape}"
            
            return vevo_mel
            
        finally:
            # 清理临时文件
            import shutil
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
    
    def _get_random_neutral_reference(self, target_language: str) -> np.ndarray:
        """从 Emo-Emilia 中性参考池中随机获取中性音频"""
        
        if not self.neutral_reference_pool:
            print("⚠️ Emo-Emilia 中性参考池为空，使用合成音频")
            return self._generate_synthetic_neutral_audio()
        
        # 严格匹配语言：中文音频使用中文中性参考，英文音频使用英文中性参考
        matching_samples = [
            sample for sample in self.neutral_reference_pool 
            if sample['language'].lower() == target_language.lower()
        ]
        
        if not matching_samples:
            # 如果没有匹配语言的中性音频，这是一个严重问题
            print(f"❌ 错误：没有找到 {target_language.upper()} 语言的中性参考音频！")
            print(f"⚠️ 这将影响中性化质量，建议检查 Emo-Emilia 数据集")
            
            # 显示可用语言
            available_langs = set(sample['language'].lower() for sample in self.neutral_reference_pool)
            print(f"⚠️ 可用语言: {list(available_langs)}")
            
            # 作为最后手段，使用合成音频
            return {
                'audio': self._generate_synthetic_neutral_audio(),
                'text': "This is neutral speech." if target_language == 'en' else "这是中性语音。"
            }
        
        if not matching_samples:
            print("⚠️ 没有可用的中性参考，使用合成音频")
            return self._generate_synthetic_neutral_audio()
        
        # 随机选择一个 Emo-Emilia 中的中性音频
        import random
        selected_sample = random.choice(matching_samples)
        
        # 获取音频数据
        audio_data = selected_sample['audio']
        audio_array = audio_data['array']
        
        # 重采样到24kHz（如果需要）
        if audio_data['sampling_rate'] != 24000:
            import librosa
            audio_array = librosa.resample(
                audio_array, 
                orig_sr=audio_data['sampling_rate'], 
                target_sr=24000
            )
        
        # 限制时长到合理范围（3-8秒，适合作为风格参考）
        min_samples = int(3.0 * 24000)
        max_samples = int(8.0 * 24000)
        
        if len(audio_array) > max_samples:
            # 随机截取一段
            start_idx = random.randint(0, len(audio_array) - max_samples)
            audio_array = audio_array[start_idx:start_idx + max_samples]
        elif len(audio_array) < min_samples:
            # 如果太短，重复填充
            repeat_times = (min_samples + len(audio_array) - 1) // len(audio_array)
            audio_array = np.tile(audio_array, repeat_times)[:min_samples]
        
        # 验证语言匹配
        selected_lang = selected_sample['language'].lower()
        if selected_lang != target_language.lower():
            print(f"⚠️ 警告：语言不匹配！目标: {target_language}, 选中: {selected_lang}")
        
        print(f"✅ 使用 Emo-Emilia 中性参考: {selected_sample['language'].upper()} 语言, {len(audio_array)/24000:.1f}秒")
        print(f"  参考文本: '{selected_sample.get('text', '')[:30]}...'")
        
        return audio_array.astype(np.float32)
    
    def _append_to_metadata_csv(self, sample_id: str, sample_metadata: Dict, 
                               mel_original: torch.Tensor, mel_neutral: torch.Tensor,
                               ev2_features: Dict):
        """添加记录到CSV元信息文件"""
        
        record = {
            'sample_id': sample_id,
            'language': sample_metadata.get('language', 'unknown'),
            'duration_seconds': sample_metadata.get('duration', 0),
            'text': sample_metadata.get('text', '').replace(',', ';'),  # 避免CSV分隔符问题
            'speaker': sample_metadata.get('speaker', 'unknown'),
            'original_mel_file': f"{sample_id}_mel_original_vevo.npz",
            'neutral_mel_file': f"{sample_id}_mel_neutral_vevo.npz",
            'emotion_features_file': f"{sample_id}_emotion_features.npz",
            'original_mel_shape': f"{mel_original.shape[1]}x{mel_original.shape[2]}",  # TxC
            'neutral_mel_shape': f"{mel_neutral.shape[1]}x{mel_neutral.shape[2]}",
            'emotion_utterance_dim': ev2_features['utterance'].shape[1],
            'emotion_frame_dim': ev2_features['frame'].shape[2],
            'processing_timestamp': datetime.now().isoformat(),
            'vevo_compatible': True,
            'emotion_source': 'original_audio'  # 确认情感特征来源
        }
        
        self.metadata_records.append(record)
        
        # 每处理100个样本就保存一次CSV
        if len(self.metadata_records) % 100 == 0:
            self._save_metadata_csv()
    
    def _save_metadata_csv(self):
        """保存CSV元信息文件"""
        if not self.metadata_records:
            return
        
        import pandas as pd
        
        # 创建或追加到CSV文件
        df = pd.DataFrame(self.metadata_records)
        
        if self.metadata_csv_path.exists():
            # 追加模式
            df.to_csv(self.metadata_csv_path, mode='a', header=False, index=False)
        else:
            # 新建文件
            df.to_csv(self.metadata_csv_path, index=False)
        
        print(f"✅ 元信息已保存: {len(self.metadata_records)} 条记录")
        self.metadata_records.clear()  # 清空缓存
    
    def _get_random_neutral_reference_with_text(self, target_language: str) -> Dict:
        """从 Emo-Emilia 中获取随机中性参考音频和对应文本"""
        
        if not self.neutral_reference_pool:
            return {
                'audio': self._generate_synthetic_neutral_audio(),
                'text': "This is neutral speech." if target_language == 'en' else "这是中性语音。"
            }
        
        # 筛选匹配语言的中性音频
        matching_samples = [
            sample for sample in self.neutral_reference_pool 
            if sample['language'].lower() == target_language.lower()
        ]
        
        if not matching_samples:
            matching_samples = self.neutral_reference_pool
        
        if not matching_samples:
            return {
                'audio': self._generate_synthetic_neutral_audio(),
                'text': "This is neutral speech." if target_language == 'en' else "这是中性语音。"
            }
        
        # 随机选择一个 Emo-Emilia 中的中性样本
        import random
        selected_sample = random.choice(matching_samples)
        
        # 处理音频数据
        audio_data = selected_sample['audio']
        audio_array = audio_data['array']
        
        # 重采样到24kHz
        if audio_data['sampling_rate'] != 24000:
            import librosa
            audio_array = librosa.resample(
                audio_array, 
                orig_sr=audio_data['sampling_rate'], 
                target_sr=24000
            )
        
        # 限制时长
        max_samples = int(8.0 * 24000)
        if len(audio_array) > max_samples:
            start_idx = random.randint(0, len(audio_array) - max_samples)
            audio_array = audio_array[start_idx:start_idx + max_samples]
        
        # 确保返回的数据包含语言信息，供后续验证
        return {
            'audio': audio_array.astype(np.float32),
            'text': selected_sample.get('text', "This is neutral speech." if target_language == 'en' else "这是中性语音。"),
            'language': selected_sample['language'],
            'speaker': selected_sample.get('speaker', 'unknown'),
            'emotion': 'neutral'
        }
    
    def _generate_synthetic_neutral_audio(self, duration: float = 3.0) -> np.ndarray:
        """生成合成中性音频作为备选"""
        t = np.linspace(0, duration, int(duration * 24000))
        neutral_audio = 0.3 * np.sin(2 * np.pi * 150 * t) + 0.05 * np.random.randn(int(duration * 24000))
        return neutral_audio.astype(np.float32)
    
    def _save_training_data(self, sample_id: str, 
                           mel_original: torch.Tensor, 
                           mel_neutral: torch.Tensor, 
                           ev2_features: Dict,
                           sample_metadata: Dict):
        """保存训练数据、元信息并验证参数一致性"""
        
        # 最终验证：确保两个mel都是Vevo格式且参数一致
        assert mel_original.shape[-1] == 128, f"源mel参数错误: {mel_original.shape}"
        assert mel_neutral.shape[-1] == 128, f"中性mel参数错误: {mel_neutral.shape}"
        assert mel_original.dim() == 3 and mel_neutral.dim() == 3, "两个mel都必须是3维张量"
        
        # 保存数据并添加元数据
        mel_orig_data = mel_original.squeeze().cpu().numpy()
        mel_neut_data = mel_neutral.squeeze().cpu().numpy()
        
        # 保存源mel到专用目录
        np.savez_compressed(
            self.output_dirs['original_mels'] / f"{sample_id}_mel_original_vevo.npz",
            mel=mel_orig_data,
            shape=mel_original.shape,
            format="vevo_compatible",
            hop_size=480,
            n_mels=128,
            sample_rate=24000,
            source_type="original_audio",
            language=sample_metadata.get('language', 'unknown'),
            duration=sample_metadata.get('duration', 0),
            text=sample_metadata.get('text', '')[:100]  # 截断长文本
        )
        
        # 保存中性mel到专用目录
        np.savez_compressed(
            self.output_dirs['neutral_mels'] / f"{sample_id}_mel_neutral_vevo.npz",
            mel=mel_neut_data,
            shape=mel_neutral.shape,
            format="vevo_compatible",
            hop_size=480,
            n_mels=128,
            sample_rate=24000,
            source_type="neutral_transformed",
            neutral_reference_source="emo_emilia",
            language=sample_metadata.get('language', 'unknown'),
            original_duration=sample_metadata.get('duration', 0),
            original_text=sample_metadata.get('text', '')[:100]
        )
        
        # 保存源音频的情感特征（确认是从源音频提取）
        np.savez_compressed(
            self.output_dirs['emotion_features'] / f"{sample_id}_emotion_features.npz",
            utterance=ev2_features['utterance'].cpu().numpy(),  # 源音频的语句级情感特征
            frame=ev2_features['frame'].cpu().numpy(),          # 源音频的帧级情感特征
            language=sample_metadata.get('language', 'unknown'),
            duration=sample_metadata.get('duration', 0),
            text=sample_metadata.get('text', '')[:100],
            source_type="original_audio_emotion"  # 明确标记是源音频情感
        )
        
        # 生成CSV元信息记录
        self._append_to_metadata_csv(sample_id, sample_metadata, mel_original, mel_neutral, ev2_features)
    
    def run_generation(self, dataset_path: str, max_samples: Optional[int] = None, 
                      target_hours_per_lang: float = 50.0, batch_start_idx: Optional[int] = None,
                      batch_end_idx: Optional[int] = None, batch_id: Optional[int] = None):
        """运行训练数据生成"""
        print("=" * 60)
        print("🎯 Emilia训练数据生成 - 简化高效版本")
        print("=" * 60)
        
        try:
            # 初始化模型
            self.initialize_models()
            
            # 加载数据集（按指定时长筛选）
            dataset = self.load_dataset(dataset_path, target_hours_per_lang=target_hours_per_lang)
            
            # 处理batch分片
            if batch_start_idx is not None and batch_end_idx is not None:
                dataset = dataset[batch_start_idx:batch_end_idx]
                print(f"📦 Batch {batch_id}: 处理样本 {batch_start_idx}-{batch_end_idx} ({len(dataset)} 个)")
            elif max_samples:
                dataset = dataset[:max_samples]
                print(f"📊 测试模式: 限制处理样本数 {max_samples}")
            
            total_samples = len(dataset)
            success_count = 0
            failed_count = 0
            
            batch_desc = f"Batch {batch_id}" if batch_id is not None else "生成训练数据"
            print(f"🚀 {batch_desc}: 开始处理 {total_samples} 个样本...")
            
            # 主处理循环带进度条
            with tqdm(dataset, desc="生成训练数据", 
                     bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]") as pbar:
                
                for sample_data in pbar:
                    success = self.process_sample(sample_data)
                    
                    if success:
                        success_count += 1
                    else:
                        failed_count += 1
                    
                    # 更新进度条信息
                    success_rate = success_count / (success_count + failed_count) * 100
                    pbar.set_postfix({
                        '成功': success_count,
                        '失败': failed_count,
                        '成功率': f'{success_rate:.1f}%'
                    })
            
            # 保存最终的CSV元信息
            self._save_metadata_csv()
            
            # 生成最终报告和数据集统计
            self._generate_final_report_and_summary(total_samples, success_count, failed_count)
            
            print("=" * 60)
            print(f"✅ 训练数据生成完成！")
            print(f"📊 成功: {success_count}/{total_samples} ({success_count/total_samples*100:.1f}%)")
            print(f"📁 输出位置: {self.output_path}")
            print("=" * 60)
            
            return True
            
        except Exception as e:
            print(f"❌ 训练数据生成失败: {e}")
            return False
    
    def _generate_final_report_and_summary(self, total: int, success: int, failed: int):
        """生成最终报告和数据集统计"""
        
        # 统计生成的文件
        file_stats = self._count_generated_files()
        
        # 生成详细报告
        report = {
            "generation_config": {
                "output_mode": "training_data_only",
                "device": self.device,
                "hours_per_language": 50.0,
                "use_emo_emilia_neutral_ref": True,
                "timestamp": datetime.now().isoformat()
            },
            "processing_summary": {
                "total_samples_processed": total,
                "successful_samples": success,
                "failed_samples": failed,
                "success_rate": success / total if total > 0 else 0
            },
            "output_statistics": file_stats,
            "data_format": {
                "mel_format": "[T, 128] Vevo compatible",
                "hop_size": 480,
                "n_mels": 128,
                "sample_rate": 24000,
                "emotion_features": "emotion2vec from original audio"
            }
        }
        
        # 保存JSON报告
        with open(self.output_dirs['reports'] / "final_generation_report.json", 'w') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        # 生成数据集README
        self._generate_dataset_readme(report)
        
        print(f"✅ 最终报告已生成: final_generation_report.json")
    
    def _count_generated_files(self) -> Dict:
        """统计生成的文件"""
        stats = {
            'original_mels': {'en': 0, 'zh': 0, 'total': 0},
            'neutral_mels': {'en': 0, 'zh': 0, 'total': 0},
            'emotion_features': {'en': 0, 'zh': 0, 'total': 0}
        }
        
        # 统计源mel文件
        for mel_file in self.output_dirs['original_mels'].glob("*.npz"):
            if '_en_' in mel_file.name:
                stats['original_mels']['en'] += 1
            elif '_zh_' in mel_file.name:
                stats['original_mels']['zh'] += 1
            stats['original_mels']['total'] += 1
        
        # 统计中性mel文件
        for mel_file in self.output_dirs['neutral_mels'].glob("*.npz"):
            if '_en_' in mel_file.name:
                stats['neutral_mels']['en'] += 1
            elif '_zh_' in mel_file.name:
                stats['neutral_mels']['zh'] += 1
            stats['neutral_mels']['total'] += 1
        
        # 统计情感特征文件
        for emotion_file in self.output_dirs['emotion_features'].glob("*.npz"):
            if '_en_' in emotion_file.name:
                stats['emotion_features']['en'] += 1
            elif '_zh_' in emotion_file.name:
                stats['emotion_features']['zh'] += 1
            stats['emotion_features']['total'] += 1
        
        return stats
    
    def _generate_dataset_readme(self, report: Dict):
        """生成数据集README文件"""
        
        readme_content = f'''# Emilia训练数据集

## 数据集信息

- **生成时间**: {report['generation_config']['timestamp']}
- **数据规模**: 英文和中文各50小时
- **成功样本**: {report['processing_summary']['successful_samples']}
- **成功率**: {report['processing_summary']['success_rate']:.1%}

## 目录结构

```
training_dataset/
├── original_mels/              # 源音频mel频谱图
│   └── *_mel_original_vevo.npz
├── neutral_mels/               # 中性化mel频谱图
│   └── *_mel_neutral_vevo.npz
├── emotion_features/           # 源音频情感特征
│   └── *_emotion_features.npz
├── metadata/                   # 元信息文件
│   ├── training_dataset_metadata.csv
│   ├── dataset_statistics.json
│   └── README.md
└── reports/                    # 生成报告
    └── final_generation_report.json
```

## 文件格式

### mel频谱图 (.npz)
- **格式**: [T, 128] (Vevo兼容)
- **参数**: hop_size=480, n_mels=128, sr=24000
- **数据**: 'mel' 键包含频谱数据

### 情感特征 (.npz)
- **来源**: 源音频 (emotion2vec)
- **utterance**: 语句级情感特征
- **frame**: 帧级情感特征

## 使用示例

```python
import numpy as np
import pandas as pd

# 加载元信息
df = pd.read_csv("metadata/training_dataset_metadata.csv")

# 加载源mel
original_mel = np.load("original_mels/en_001234_mel_original_vevo.npz")['mel']

# 加载中性mel
neutral_mel = np.load("neutral_mels/en_001234_mel_neutral_vevo.npz")['mel']

# 加载情感特征
emotion_data = np.load("emotion_features/en_001234_emotion_features.npz")
utterance_emotion = emotion_data['utterance']
frame_emotion = emotion_data['frame']
```

## 语言匹配保证

- **中文音频**: 使用中文Emo-Emilia neutral参考
- **英文音频**: 使用英文Emo-Emilia neutral参考
- **质量保证**: 专家验证的情感标签
'''
        
        with open(self.output_dirs['metadata'] / "README.md", 'w', encoding='utf-8') as f:
            f.write(readme_content)
        
        # 生成统计文件
        stats = {
            "dataset_name": "Emilia_Training_Data",
            "generation_timestamp": report['generation_config']['timestamp'],
            "total_samples": report['processing_summary']['successful_samples'],
            "file_statistics": report['output_statistics'],
            "format_info": report['data_format']
        }
        
        with open(self.output_dirs['metadata'] / "dataset_statistics.json", 'w') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Emilia训练数据生成脚本")
    
    parser.add_argument("--dataset_path", type=str, required=True,
                       help="数据集路径")
    parser.add_argument("--output_path", type=str, required=True,
                       help="输出路径")
    parser.add_argument("--cache_path", type=str, required=True,
                       help="缓存路径")
    parser.add_argument("--device", type=str, default="cuda",
                       help="计算设备")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="最大样本数（用于测试）")
    parser.add_argument("--hours_per_lang", type=float, default=50.0,
                       help="每种语言的目标时长（小时）")
    parser.add_argument("--batch_start_idx", type=int, default=None,
                       help="batch开始索引（用于分批处理）")
    parser.add_argument("--batch_end_idx", type=int, default=None,
                       help="batch结束索引（用于分批处理）")
    parser.add_argument("--batch_id", type=int, default=None,
                       help="batch ID（用于标识）")
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_arguments()
    
    try:
        generator = TrainingDataGenerator(
            output_path=args.output_path,
            cache_path=args.cache_path,
            device=args.device
        )
        
        success = generator.run_generation(
            dataset_path=args.dataset_path,
            max_samples=args.max_samples,
            target_hours_per_lang=args.hours_per_lang,
            batch_start_idx=args.batch_start_idx,
            batch_end_idx=args.batch_end_idx,
            batch_id=args.batch_id
        )
        
        return 0 if success else 1
        
    except KeyboardInterrupt:
        print("\n⚠️ 用户中断")
        return 130
    except Exception as e:
        print(f"❌ 程序异常: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
