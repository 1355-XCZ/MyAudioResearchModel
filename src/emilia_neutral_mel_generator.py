"""
Emilia Dataset Neutral Mel Generator
使用 Vevo TTS 模式生成大量中性频谱图
"""

import os
import json
import torch
import torchaudio
import numpy as np
from datasets import load_dataset
from pathlib import Path
import librosa
import soundfile as sf
from tqdm import tqdm
import random
from typing import Dict, List, Tuple, Optional
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmiliaNeutralMelGenerator:
    """
    基于 Emilia 数据集和 Vevo TTS 模式生成中性频谱图
    """
    
    def __init__(self, 
                 output_dir: str = "data/emilia_neutral_mels",
                 vevo_config: Optional[Dict] = None,
                 balance_languages: bool = True,
                 max_samples_per_lang: int = 10000):
        """
        初始化生成器
        
        Args:
            output_dir: 输出目录
            vevo_config: Vevo 配置，如果为 None 则使用默认配置
            balance_languages: 是否平衡中英文数据
            max_samples_per_lang: 每种语言的最大样本数
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.balance_languages = balance_languages
        self.max_samples_per_lang = max_samples_per_lang
        
        # Vevo 配置 - 根据实际的 Vevo 配置文件设置
        if vevo_config is None:
            self.vevo_config = {
                # Mel 频谱图参数 - 与 Vevo 的 Vocoder.json 保持一致
                "sample_rate": 24000,        # Vevo 使用 24kHz
                "hop_size": 480,             # Vevo 的 hop_size
                "n_fft": 1920,               # Vevo 的 n_fft
                "win_size": 1920,            # Vevo 的 win_size
                "num_mels": 128,             # Vevo 使用 128 mel channels
                "fmin": 0,                   # 最小频率
                "fmax": 12000,               # 最大频率 (24000/2)
                "mel_mean": -4.92,           # Vevo 的 mel 均值
                "mel_var": 8.14,             # Vevo 的 mel 方差
                
                # 数据处理参数
                "max_length": 36000,         # 最大音频长度 (1.5s at 24kHz)
                "min_length": 2400,          # 最小音频长度 (0.1s at 24kHz)
            }
        else:
            self.vevo_config = vevo_config
            
        # 初始化 mel 频谱图提取器
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.vevo_config["sample_rate"],
            n_fft=self.vevo_config["n_fft"],
            win_length=self.vevo_config["win_size"],
            hop_length=self.vevo_config["hop_size"],
            n_mels=self.vevo_config["num_mels"],
            f_min=self.vevo_config["fmin"],
            f_max=self.vevo_config["fmax"],
            power=2.0,
            normalized=False
        )
        
        # 支持的语言
        self.target_languages = ["EN", "ZH"]  # 只使用中英文
        
        logger.info(f"初始化完成，输出目录: {self.output_dir}")
        logger.info(f"Vevo 配置: {self.vevo_config}")
    
    def load_emilia_dataset(self, streaming: bool = True) -> Dict[str, any]:
        """
        加载 Emilia 数据集
        
        Args:
            streaming: 是否使用流式加载
            
        Returns:
            包含中英文数据的字典
        """
        logger.info("加载 Emilia 数据集...")
        
        datasets = {}
        
        for lang in self.target_languages:
            logger.info(f"加载 {lang} 数据...")
            
            # 根据 Emilia 数据集的结构加载特定语言的数据
            path = f"Emilia/{lang}/*.tar"
            
            try:
                dataset = load_dataset(
                    "amphion/Emilia-Dataset", 
                    data_files={lang.lower(): path}, 
                    split=lang.lower(), 
                    streaming=streaming
                )
                datasets[lang] = dataset
                logger.info(f"成功加载 {lang} 数据集")
                
            except Exception as e:
                logger.error(f"加载 {lang} 数据失败: {e}")
                # 创建空数据集作为后备
                datasets[lang] = []
        
        return datasets
    
    def preprocess_audio(self, audio: torch.Tensor, sample_rate: int) -> Optional[torch.Tensor]:
        """
        预处理音频数据
        
        Args:
            audio: 音频张量
            sample_rate: 采样率
            
        Returns:
            预处理后的音频，如果不符合要求则返回 None
        """
        # 重采样到目标采样率
        if sample_rate != self.vevo_config["sample_rate"]:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate, 
                new_freq=self.vevo_config["sample_rate"]
            )
            audio = resampler(audio)
        
        # 确保是单声道
        if audio.dim() > 1:
            audio = torch.mean(audio, dim=0)
        
        # 检查音频长度
        if len(audio) < self.vevo_config["min_length"]:
            return None
            
        # 截断过长的音频
        if len(audio) > self.vevo_config["max_length"]:
            audio = audio[:self.vevo_config["max_length"]]
        
        # 音频标准化
        audio = audio / (torch.max(torch.abs(audio)) + 1e-8)
        
        return audio
    
    def extract_mel_spectrogram(self, audio: torch.Tensor) -> torch.Tensor:
        """
        提取 mel 频谱图
        
        Args:
            audio: 预处理后的音频
            
        Returns:
            mel 频谱图张量 [n_mels, frames]
        """
        # 提取 mel 频谱图
        mel = self.mel_transform(audio)
        
        # 转换到 log scale
        mel = torch.log(torch.clamp(mel, min=1e-8))
        
        # 标准化 (使用 Vevo 的统计值)
        mel = (mel - self.vevo_config["mel_mean"]) / torch.sqrt(self.vevo_config["mel_var"])
        
        return mel
    
    def save_mel_and_metadata(self, 
                             mel: torch.Tensor, 
                             metadata: Dict, 
                             save_path: Path) -> None:
        """
        保存 mel 频谱图和元数据
        
        Args:
            mel: mel 频谱图
            metadata: 元数据
            save_path: 保存路径（不含扩展名）
        """
        # 保存 mel 频谱图
        mel_path = save_path.with_suffix('.npy')
        np.save(mel_path, mel.numpy())
        
        # 保存元数据
        metadata_path = save_path.with_suffix('.json')
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    def process_single_sample(self, 
                             sample: Dict, 
                             lang: str, 
                             sample_id: str) -> bool:
        """
        处理单个样本
        
        Args:
            sample: 样本数据
            lang: 语言标识
            sample_id: 样本ID
            
        Returns:
            是否处理成功
        """
        try:
            # 获取音频数据
            if 'audio' in sample and 'array' in sample['audio']:
                audio = torch.from_numpy(sample['audio']['array']).float()
                sample_rate = sample['audio']['sampling_rate']
            else:
                logger.warning(f"样本 {sample_id} 缺少音频数据")
                return False
            
            # 预处理音频
            processed_audio = self.preprocess_audio(audio, sample_rate)
            if processed_audio is None:
                logger.warning(f"样本 {sample_id} 音频预处理失败")
                return False
            
            # 提取 mel 频谱图
            mel = self.extract_mel_spectrogram(processed_audio)
            
            # 准备元数据
            metadata = {
                "id": sample_id,
                "language": lang,
                "text": sample.get('text', ''),
                "duration": float(len(processed_audio)) / self.vevo_config["sample_rate"],
                "mel_shape": list(mel.shape),
                "original_duration": sample.get('duration', 0),
                "speaker": sample.get('speaker', ''),
                "vevo_config": self.vevo_config
            }
            
            # 保存文件
            lang_dir = self.output_dir / lang
            lang_dir.mkdir(exist_ok=True)
            save_path = lang_dir / sample_id
            
            self.save_mel_and_metadata(mel, metadata, save_path)
            
            return True
            
        except Exception as e:
            logger.error(f"处理样本 {sample_id} 时出错: {e}")
            return False
    
    def generate_balanced_samples(self, datasets: Dict) -> None:
        """
        生成平衡的中英文样本
        
        Args:
            datasets: 包含各语言数据集的字典
        """
        logger.info("开始生成平衡的中英文样本...")
        
        # 统计计数器
        counters = {lang: 0 for lang in self.target_languages}
        success_counters = {lang: 0 for lang in self.target_languages}
        
        # 创建语言列表用于平衡采样
        lang_cycle = []
        for lang in self.target_languages:
            if lang in datasets and datasets[lang]:
                lang_cycle.extend([lang] * self.max_samples_per_lang)
        
        random.shuffle(lang_cycle)
        
        # 创建数据迭代器
        iterators = {}
        for lang in self.target_languages:
            if lang in datasets and datasets[lang]:
                iterators[lang] = iter(datasets[lang])
        
        # 生成样本
        pbar = tqdm(total=len(lang_cycle), desc="生成中性mel")
        
        for target_lang in lang_cycle:
            if counters[target_lang] >= self.max_samples_per_lang:
                pbar.update(1)
                continue
                
            if target_lang not in iterators:
                pbar.update(1)
                continue
            
            try:
                # 获取下一个样本
                sample = next(iterators[target_lang])
                counters[target_lang] += 1
                
                # 生成样本ID
                sample_id = f"{target_lang}_{counters[target_lang]:06d}"
                
                # 处理样本
                if self.process_single_sample(sample, target_lang, sample_id):
                    success_counters[target_lang] += 1
                
                pbar.set_postfix({
                    f"{lang}_success": success_counters[lang] 
                    for lang in self.target_languages
                })
                
            except StopIteration:
                logger.warning(f"{target_lang} 数据集已耗尽")
                # 从循环中移除这个语言
                lang_cycle = [l for l in lang_cycle if l != target_lang]
                if not lang_cycle:
                    break
            except Exception as e:
                logger.error(f"处理 {target_lang} 样本时出错: {e}")
            
            pbar.update(1)
        
        pbar.close()
        
        # 输出统计信息
        logger.info("生成完成！统计信息:")
        for lang in self.target_languages:
            logger.info(f"{lang}: 成功 {success_counters[lang]} / 尝试 {counters[lang]}")
    
    def create_file_lists(self) -> None:
        """
        创建文件列表用于训练
        """
        logger.info("创建文件列表...")
        
        # 创建总的文件列表
        all_files = []
        lang_stats = {}
        
        for lang in self.target_languages:
            lang_dir = self.output_dir / lang
            if not lang_dir.exists():
                continue
                
            lang_files = []
            for mel_file in lang_dir.glob("*.npy"):
                sample_id = mel_file.stem
                
                # 检查对应的元数据文件是否存在
                metadata_file = mel_file.with_suffix('.json')
                if metadata_file.exists():
                    lang_files.append({
                        "id": sample_id,
                        "language": lang,
                        "mel_path": str(mel_file.relative_to(self.output_dir)),
                        "metadata_path": str(metadata_file.relative_to(self.output_dir))
                    })
            
            lang_stats[lang] = len(lang_files)
            all_files.extend(lang_files)
        
        # 随机打乱
        random.shuffle(all_files)
        
        # 划分训练集和验证集 (9:1)
        split_idx = int(0.9 * len(all_files))
        train_files = all_files[:split_idx]
        val_files = all_files[split_idx:]
        
        # 保存文件列表
        lists_dir = self.output_dir / "file_lists"
        lists_dir.mkdir(exist_ok=True)
        
        # 保存训练集列表
        with open(lists_dir / "train.json", 'w', encoding='utf-8') as f:
            json.dump(train_files, f, ensure_ascii=False, indent=2)
        
        # 保存验证集列表
        with open(lists_dir / "val.json", 'w', encoding='utf-8') as f:
            json.dump(val_files, f, ensure_ascii=False, indent=2)
        
        # 保存统计信息
        stats = {
            "total_samples": len(all_files),
            "train_samples": len(train_files),
            "val_samples": len(val_files),
            "language_distribution": lang_stats,
            "vevo_config": self.vevo_config
        }
        
        with open(lists_dir / "stats.json", 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        logger.info(f"文件列表创建完成:")
        logger.info(f"总样本: {len(all_files)}")
        logger.info(f"训练集: {len(train_files)}")
        logger.info(f"验证集: {len(val_files)}")
        logger.info(f"语言分布: {lang_stats}")
    
    def run(self) -> None:
        """
        运行完整的生成流程
        """
        logger.info("开始生成中性频谱图...")
        
        # 1. 加载数据集
        datasets = self.load_emilia_dataset(streaming=True)
        
        # 2. 生成样本
        self.generate_balanced_samples(datasets)
        
        # 3. 创建文件列表
        self.create_file_lists()
        
        logger.info("所有任务完成！")


def main():
    """
    主函数
    """
    # 创建生成器
    generator = EmiliaNeutralMelGenerator(
        output_dir="data/emilia_neutral_mels",
        balance_languages=True,
        max_samples_per_lang=5000  # 每种语言5000个样本
    )
    
    # 运行生成流程
    generator.run()


if __name__ == "__main__":
    main()
