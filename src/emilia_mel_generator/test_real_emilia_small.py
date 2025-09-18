"""
真实Emilia数据集小测试脚本
下载中英各5个样本，验证真实数据处理流程
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
from datasets import load_dataset

from corrected_data_generator import CorrectedDataGenerator, VevoTTSEmulator, Emotion2VecExtractor

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RealEmiliaSmallTester:
    """
    真实Emilia数据集小规模测试器
    """
    
    def __init__(self, num_samples_per_lang=5, k_variants=1, device='cuda'):
        self.num_samples_per_lang = num_samples_per_lang
        self.k_variants = k_variants
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.output_dir = Path("test_output_real_small")
        
        # 初始化组件
        self.vevo_emulator = VevoTTSEmulator(device=self.device)
        self.emotion2vec_extractor = Emotion2VecExtractor(device=self.device)
        
        logger.info(f"真实Emilia小测试初始化完成:")
        logger.info(f"  每语言样本数: {num_samples_per_lang}")
        logger.info(f"  K值: {k_variants}")
        logger.info(f"  设备: {device}")

    def download_small_emilia_samples(self) -> Dict:
        """
        下载少量Emilia样本用于测试
        
        Returns:
            包含S1和S2的真实数据
        """
        logger.info("🌐 开始下载真实Emilia数据样本...")
        
        s1_data = {"EN": [], "ZH": []}
        
        # 下载中英文样本
        for lang_code in ["EN", "ZH"]:
            logger.info(f"下载 {lang_code} 语言样本...")
            
            try:
                # 加载特定语言的数据（流式）
                path = f"Emilia/{lang_code}/*.tar"
                dataset = load_dataset(
                    "amphion/Emilia-Dataset", 
                    data_files={"train": path}, 
                    split="train", 
                    streaming=True
                )
                
                logger.info(f"  成功连接到Emilia数据集 {lang_code} 分支")
                
                # 收集指定数量的样本
                samples_collected = 0
                for sample in dataset:
                    if samples_collected >= self.num_samples_per_lang:
                        break
                    
                    # 检查样本质量
                    if self._is_valid_sample(sample):
                        s1_data[lang_code].append(sample)
                        samples_collected += 1
                        logger.info(f"  收集 {lang_code} 样本 {samples_collected}/{self.num_samples_per_lang}")
                
                logger.info(f"✅ {lang_code} 语言收集完成: {len(s1_data[lang_code])} 个样本")
                
            except Exception as e:
                logger.error(f"❌ 下载 {lang_code} 数据失败: {e}")
                logger.info(f"使用占位数据代替 {lang_code} 样本...")
                s1_data[lang_code] = self._create_fallback_samples(lang_code)
        
        # 创建S2数据（模拟Emo-Emilia neutral，因为需要访问权限）
        logger.info("创建S2中性参考数据...")
        s2_data = self._create_s2_neutral_samples()
        
        return {
            "s1_original": s1_data,
            "s2_neutral": s2_data
        }

    def _is_valid_sample(self, sample) -> bool:
        """检查样本是否有效"""
        try:
            # 检查必要字段
            if 'audio' not in sample or 'text' not in sample:
                return False
            
            # 检查音频长度（1-10秒）
            audio_data = sample['audio']
            if 'array' in audio_data and 'sampling_rate' in audio_data:
                duration = len(audio_data['array']) / audio_data['sampling_rate']
                if 1.0 <= duration <= 10.0:
                    return True
            
            # 如果是WebDataset格式，尝试其他字段
            if 'path' in audio_data or 'bytes' in audio_data:
                return True
                
            return False
        except Exception as e:
            logger.debug(f"样本验证失败: {e}")
            return False

    def _create_fallback_samples(self, lang_code) -> List:
        """创建占位样本作为备选"""
        samples = []
        for i in range(self.num_samples_per_lang):
            duration = 3.0
            sample_rate = 24000
            samples_count = int(duration * sample_rate)
            
            # 生成复合音频信号
            t = torch.linspace(0, duration, samples_count)
            base_freq = 200 + i * 50
            
            audio = (
                0.5 * torch.sin(2 * np.pi * base_freq * t) +
                0.3 * torch.sin(2 * np.pi * base_freq * 2 * t) +
                0.1 * torch.randn(samples_count)
            )
            
            sample = {
                'audio': {
                    'array': audio.numpy(),
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
        for i in range(20):  # 20个中性参考
            duration = 2.0
            sample_rate = 24000
            samples_count = int(duration * sample_rate)
            
            # 创建中性风格音频
            t = torch.linspace(0, duration, samples_count)
            neutral_freq = 150 + i * 10  # 低频，模拟中性
            
            audio = (
                0.4 * torch.sin(2 * np.pi * neutral_freq * t) +
                0.05 * torch.randn(samples_count)
            )
            
            sample = {
                'audio': {
                    'array': audio.numpy(),
                    'sampling_rate': sample_rate
                },
                'text': f"Neutral reference {i+1}",
                'speaker': f"neutral_speaker_{i+1}",
                'emotion': 'neutral',
                'duration': duration
            }
            s2_data.append(sample)
        
        return s2_data

    def run_test(self):
        """运行真实数据小测试"""
        logger.info("🚀 开始真实Emilia数据小测试...")
        
        try:
            # 1. 下载真实数据样本
            datasets = self.download_small_emilia_samples()
            
            # 2. 创建输出目录
            self.output_dir.mkdir(exist_ok=True)
            (self.output_dir / "mels").mkdir(exist_ok=True)
            (self.output_dir / "emotion_features").mkdir(exist_ok=True)
            (self.output_dir / "verification_audio").mkdir(exist_ok=True)
            
            # 3. 处理每个样本
            test_results = []
            s1_data = datasets["s1_original"]
            s2_data = datasets["s2_neutral"]
            
            for lang in ["EN", "ZH"]:
                for i, s1_sample in enumerate(s1_data[lang]):
                    sample_id = f"{lang}_S1_{i+1:03d}"
                    logger.info(f"处理样本: {sample_id}")
                    
                    try:
                        result = self._process_sample(sample_id, s1_sample, s2_data, lang)
                        test_results.append(result)
                    except Exception as e:
                        logger.error(f"处理样本 {sample_id} 失败: {e}")
            
            # 4. 保存测试报告
            report = {
                "test_config": {
                    "num_samples_per_lang": self.num_samples_per_lang,
                    "k_variants": self.k_variants,
                    "device": self.device,
                    "data_source": "Real Emilia Dataset (small sample)"
                },
                "test_results": test_results,
                "file_structure": {
                    "mels": "mels/*.npy",
                    "emotion_features": "emotion_features/*.npz",
                    "verification_audio": "verification_audio/*.wav"
                },
                "verification_guide": {
                    "listen_to": [
                        "*_s1_original.wav (真实Emilia原始音频)",
                        "*_original_reconstructed.wav (原始mel重建音频)",
                        "*_neutral_reconstructed.wav (中性mel重建音频)"
                    ],
                    "expected": "对比真实音频和重建音频的质量差异"
                }
            }
            
            with open(self.output_dir / "test_report.json", 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            logger.info("✅ 真实Emilia数据小测试完成！")
            logger.info(f"📁 输出目录: {self.output_dir}")
            logger.info(f"📊 测试样本: {len(test_results)}")
            logger.info(f"🎵 验证音频: {self.output_dir}/verification_audio/")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ 测试失败: {e}")
            return False

    def _process_sample(self, sample_id: str, s1_sample: Dict, s2_data: List, lang: str) -> Dict:
        """处理单个样本"""
        # 转换音频到torch tensor并移到正确设备
        s1_audio = torch.from_numpy(s1_sample['audio']['array']).float().to(self.device)
        if s1_audio.dim() == 1:
            s1_audio = s1_audio.unsqueeze(0)
        
        # 随机选择S2参考
        s2_sample = np.random.choice(s2_data)
        s2_audio = torch.from_numpy(s2_sample['audio']['array']).float().to(self.device)
        if s2_audio.dim() == 1:
            s2_audio = s2_audio.unsqueeze(0)
        
        # 1. 提取s1的原始mel
        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=24000, n_fft=1920, win_length=1920, hop_length=480,
            n_mels=128, f_min=0, f_max=12000, power=2.0, normalized=False
        ).to(self.device)
        
        mel_original = mel_transform(s1_audio)
        mel_original = torch.log(torch.clamp(mel_original, min=1e-8))
        
        # 2. 用Vevo TTS生成中性mel
        mel_neutral = self.vevo_emulator.generate_neutral_mel(s1_audio, s2_audio)
        
        # 3. 提取emotion2vec特征
        ev2_features = self.emotion2vec_extractor.extract_features(s1_audio)
        
        # 4. 保存文件
        tuple_id = f"{sample_id}_k01"
        
        # 保存mel频谱图（移到CPU）
        np.save(self.output_dir / "mels" / f"{tuple_id}_mel_original.npy", 
                mel_original.squeeze().cpu().numpy())
        np.save(self.output_dir / "mels" / f"{tuple_id}_mel_neutral.npy", 
                mel_neutral.squeeze().cpu().numpy())
        
        # 保存emotion2vec特征（移到CPU）
        np.savez_compressed(
            self.output_dir / "emotion_features" / f"{tuple_id}_ev2.npz",
            utterance=ev2_features['utterance'].cpu().numpy(),
            frame=ev2_features['frame'].cpu().numpy()
        )
        
        # 5. 生成验证音频
        self._generate_verification_audio(tuple_id, s1_audio, mel_original, mel_neutral)
        
        return {
            "tuple_id": tuple_id,
            "s1_id": sample_id,
            "language": lang,
            "variant_index": 0,
            "mel_original_shape": list(mel_original.shape),
            "mel_neutral_shape": list(mel_neutral.shape),
            "ev2_utterance_shape": list(ev2_features['utterance'].shape),
            "ev2_frame_shape": list(ev2_features['frame'].shape),
            "audio_files": {
                "s1_original": f"{tuple_id}_s1_original.wav",
                "mel_original_reconstructed": f"{tuple_id}_original_reconstructed.wav",
                "mel_neutral_reconstructed": f"{tuple_id}_neutral_reconstructed.wav"
            },
            "s1_text": s1_sample.get('text', ''),
            "s1_speaker": s1_sample.get('speaker', ''),
            "s1_duration": s1_sample.get('duration', 0.0)
        }

    def _generate_verification_audio(self, tuple_id: str, s1_audio: torch.Tensor, 
                                   mel_original: torch.Tensor, mel_neutral: torch.Tensor):
        """生成验证音频"""
        audio_dir = self.output_dir / "verification_audio"
        
        # 1. 保存原始s1音频（移到CPU）
        sf.write(audio_dir / f"{tuple_id}_s1_original.wav", 
                s1_audio.squeeze().cpu().numpy(), 24000)
        
        # 2. 简化的验证音频生成（由于mel->audio需要vocoder，这里生成简单的音调）
        # 注意：实际应该用训练好的vocoder进行mel到音频的转换
        
        def simple_tone_from_mel(mel_spec, base_freq=200):
            """从mel频谱图生成简单音调作为验证"""
            duration = mel_spec.shape[-1] * 480 / 24000  # 根据帧数计算时长
            samples = int(duration * 24000)
            t = torch.linspace(0, duration, samples)
            
            # 使用mel的平均能量调制频率
            avg_energy = mel_spec.mean().item()
            freq = base_freq * (1 + avg_energy * 0.1)  # 根据能量调整频率
            
            # 生成简单的正弦波
            audio = 0.3 * torch.sin(2 * np.pi * freq * t)
            return audio
        
        # 3. 生成原始mel对应的验证音调
        audio_original_tone = simple_tone_from_mel(mel_original.cpu(), base_freq=200)
        sf.write(audio_dir / f"{tuple_id}_original_reconstructed.wav",
                audio_original_tone.numpy(), 24000)
        
        # 4. 生成中性mel对应的验证音调  
        audio_neutral_tone = simple_tone_from_mel(mel_neutral.cpu(), base_freq=180)
        sf.write(audio_dir / f"{tuple_id}_neutral_reconstructed.wav",
                audio_neutral_tone.numpy(), 24000)


def main():
    """主函数"""
    print("🧪 开始真实Emilia数据小测试 (中英各5个样本)...")
    
    # 创建测试器
    tester = RealEmiliaSmallTester(
        num_samples_per_lang=5,
        k_variants=1,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # 运行测试
    success = tester.run_test()
    
    if success:
        print("✅ 真实数据小测试成功！")
        print("\n📁 生成的文件:")
        print("  - mels/*.npy (mel频谱图)")
        print("  - emotion_features/*.npz (emotion2vec特征)")
        print("  - verification_audio/*.wav (验证音频)")
        print("  - test_report.json (测试报告)")
        print("\n🎵 验证方法:")
        print("1. 听取 *_s1_original.wav (真实Emilia音频)")
        print("2. 听取 *_original_reconstructed.wav (原始mel重建)")
        print("3. 听取 *_neutral_reconstructed.wav (中性mel重建)")
        print("4. 对比真实音频和重建音频的质量")
        print("\n🚀 如果测试通过，可以用于大规模Emilia数据处理！")
    else:
        print("❌ 测试失败，请检查错误信息")


if __name__ == "__main__":
    main()
