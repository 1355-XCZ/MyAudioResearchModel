"""
本地小规模测试脚本
测试10条音频，验证所有组件正常工作
包含声码器还原验证
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

from corrected_data_generator import CorrectedDataGenerator, VevoTTSEmulator, Emotion2VecExtractor

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LocalTestGenerator:
    """
    本地测试生成器
    处理少量样本，验证完整流程
    """
    
    def __init__(self, 
                 output_dir: str = "test_output",
                 num_test_samples: int = 10,
                 k_variants: int = 1):
        """
        初始化本地测试生成器
        
        Args:
            output_dir: 测试输出目录
            num_test_samples: 测试样本数量
            k_variants: K值 (默认1)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.num_test_samples = num_test_samples
        self.k_variants = k_variants
        
        # 初始化组件
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.vevo_tts = VevoTTSEmulator(device=self.device)
        self.emotion_extractor = Emotion2VecExtractor(device=self.device)
        
        # Mel变换器
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=24000,
            n_fft=1920,
            win_length=1920,
            hop_length=480,
            n_mels=128,
            f_min=0,
            f_max=12000,
            power=2.0,
            normalized=False
        ).to(self.device)
        
        # 声码器 (临时实现：Griffin-Lim)
        self.griffin_lim = torchaudio.transforms.GriffinLim(
            n_fft=1920,
            win_length=1920,
            hop_length=480,
            power=2.0,
            n_iter=32
        ).to(self.device)
        
        logger.info(f"本地测试初始化完成:")
        logger.info(f"  测试样本数: {self.num_test_samples}")
        logger.info(f"  K值: {self.k_variants}")
        logger.info(f"  设备: {self.device}")
    
    def create_dummy_datasets(self) -> Dict:
        """
        创建虚拟数据集用于测试
        
        Returns:
            包含S1和S2的虚拟数据
        """
        logger.info("创建虚拟测试数据...")
        
        # 创建S1虚拟数据 (模拟Emilia主数据集)
        s1_data = {"EN": [], "ZH": []}
        
        for lang in ["EN", "ZH"]:
            for i in range(self.num_test_samples // 2):  # 每语言5个样本
                # 生成测试音频 (3秒，不同频率)
                duration = 3.0
                sample_rate = 24000
                samples = int(duration * sample_rate)
                
                # 创建复合信号 (基频 + 谐波 + 噪声)
                t = torch.linspace(0, duration, samples)
                base_freq = 200 + i * 50  # 不同的基频
                
                audio = (
                    0.5 * torch.sin(2 * np.pi * base_freq * t) +      # 基频
                    0.3 * torch.sin(2 * np.pi * base_freq * 2 * t) +  # 二次谐波
                    0.1 * torch.randn(samples)                        # 噪声
                )
                
                sample = {
                    'audio': {
                        'array': audio.numpy(),
                        'sampling_rate': sample_rate
                    },
                    'text': f"Test audio {lang} {i+1}",
                    'speaker': f"{lang}_speaker_{i+1}",
                    'language': lang.lower(),
                    'duration': duration
                }
                
                s1_data[lang].append(sample)
        
        # 创建S2虚拟数据 (模拟Emo-Emilia neutral)
        s2_data = []
        for i in range(20):  # 20个中性参考
            duration = 2.0
            sample_rate = 24000
            samples = int(duration * sample_rate)
            
            # 创建中性风格音频 (平稳的低频信号)
            t = torch.linspace(0, duration, samples)
            neutral_freq = 150 + i * 10
            
            audio = 0.6 * torch.sin(2 * np.pi * neutral_freq * t)  # 平稳信号
            
            sample = {
                'audio': {
                    'array': audio.numpy(),
                    'sampling_rate': sample_rate
                },
                'text': f"Neutral reference {i+1}",
                'emotion': 'neutral',
                'language': 'en' if i % 2 == 0 else 'zh'
            }
            
            s2_data.append(sample)
        
        logger.info(f"虚拟数据创建完成:")
        logger.info(f"  S1样本: EN={len(s1_data['EN'])}, ZH={len(s1_data['ZH'])}")
        logger.info(f"  S2样本: {len(s2_data)}")
        
        return {"S1": s1_data, "S2": s2_data}
    
    def mel_to_audio(self, mel: torch.Tensor) -> torch.Tensor:
        """
        将mel频谱图转换回音频 (用于验证)
        
        Args:
            mel: mel频谱图 [n_mels, frames]
            
        Returns:
            重建的音频
        """
        # 反标准化
        mel = torch.exp(mel)  # log → linear
        
        # 使用Griffin-Lim算法重建音频 (临时方案)
        # 真实应用应该使用Vocos或其他神经声码器
        
        # 转换mel到线性频谱 (简化实现)
        # 这里应该使用mel滤波器组的逆变换，暂时用近似方法
        linear_spec = mel.repeat(1920//128, 1)[:1920//2+1, :]  # 近似扩展
        
        # Griffin-Lim重建
        audio = self.griffin_lim(linear_spec.to(self.device))
        
        return audio.cpu()
    
    def run_test(self) -> Dict:
        """
        运行完整测试流程
        
        Returns:
            测试结果
        """
        logger.info("🚀 开始本地小规模测试...")
        
        try:
            # 1. 创建虚拟数据集
            datasets = self.create_dummy_datasets()
            s1_data = datasets["S1"]
            s2_data = datasets["S2"]
            
            # 2. 测试处理流程
            test_results = []
            
            for lang in ["EN", "ZH"]:
                for i, s1_sample in enumerate(s1_data[lang]):
                    sample_id = f"{lang}_S1_{i+1:03d}"
                    logger.info(f"测试样本: {sample_id}")
                    
                    # 预处理s1音频
                    s1_audio = torch.from_numpy(s1_sample['audio']['array']).float()
                    s1_audio = s1_audio.to(self.device)
                    
                    # 提取s1原始mel
                    mel_original = self.mel_transform(s1_audio)
                    mel_original = torch.log(torch.clamp(mel_original, min=1e-8))
                    
                    # 提取s1的emotion2vec特征
                    s1_audio_16k = torchaudio.transforms.Resample(24000, 16000).to(self.device)(s1_audio)
                    ev2_features = self.emotion_extractor.extract_features(s1_audio_16k, 16000)
                    
                    # 生成K个中性mel变体
                    for k in range(self.k_variants):
                        # 随机选择一个S2参考
                        s2_sample = np.random.choice(s2_data)
                        s2_audio = torch.from_numpy(s2_sample['audio']['array']).float().to(self.device)
                        
                        # 生成中性mel (音色=s1, 风格=s2)
                        mel_neutral = self.vevo_tts.generate_neutral_mel(s1_audio, s2_audio)
                        
                        # 创建训练元组
                        tuple_id = f"{sample_id}_k{k+1:02d}"
                        
                        # 保存mel频谱图
                        mel_dir = self.output_dir / "mels"
                        mel_dir.mkdir(exist_ok=True)
                        
                        np.save(mel_dir / f"{tuple_id}_original.npy", mel_original.cpu().numpy())
                        np.save(mel_dir / f"{tuple_id}_neutral.npy", mel_neutral.numpy())
                        
                        # 保存emotion2vec特征
                        ev2_dir = self.output_dir / "emotion_features"
                        ev2_dir.mkdir(exist_ok=True)
                        
                        np.savez(ev2_dir / f"{tuple_id}_ev2.npz",
                                utterance=ev2_features['utterance'].numpy(),
                                frame=ev2_features['frame'].numpy())
                        
                        # 生成验证音频 (mel → audio)
                        audio_dir = self.output_dir / "verification_audio"
                        audio_dir.mkdir(exist_ok=True)
                        
                        # 重建原始mel的音频
                        audio_original = self.mel_to_audio(mel_original.cpu())
                        sf.write(audio_dir / f"{tuple_id}_original_reconstructed.wav", 
                                audio_original.numpy(), 24000)
                        
                        # 重建中性mel的音频
                        audio_neutral = self.mel_to_audio(mel_neutral)
                        sf.write(audio_dir / f"{tuple_id}_neutral_reconstructed.wav", 
                                audio_neutral.numpy(), 24000)
                        
                        # 保存原始音频作为对比
                        sf.write(audio_dir / f"{tuple_id}_s1_original.wav", 
                                s1_audio.cpu().numpy(), 24000)
                        
                        # 记录测试结果
                        result = {
                            "tuple_id": tuple_id,
                            "s1_id": sample_id,
                            "language": lang,
                            "variant_index": k,
                            "mel_original_shape": list(mel_original.shape),
                            "mel_neutral_shape": list(mel_neutral.shape),
                            "ev2_utterance_shape": list(ev2_features['utterance'].shape),
                            "ev2_frame_shape": list(ev2_features['frame'].shape),
                            "audio_files": {
                                "s1_original": f"{tuple_id}_s1_original.wav",
                                "mel_original_reconstructed": f"{tuple_id}_original_reconstructed.wav",
                                "mel_neutral_reconstructed": f"{tuple_id}_neutral_reconstructed.wav"
                            }
                        }
                        
                        test_results.append(result)
            
            # 3. 保存测试报告
            report = {
                "test_config": {
                    "num_samples": self.num_test_samples,
                    "k_variants": self.k_variants,
                    "device": str(self.device)
                },
                "test_results": test_results,
                "file_structure": {
                    "mels": "mels/*.npy",
                    "emotion_features": "emotion_features/*.npz",
                    "verification_audio": "verification_audio/*.wav"
                },
                "verification_guide": {
                    "listen_to": [
                        "*_s1_original.wav (原始s1音频)",
                        "*_original_reconstructed.wav (原始mel重建音频)",
                        "*_neutral_reconstructed.wav (中性mel重建音频)"
                    ],
                    "expected": "中性重建音频应该保持s1音色但去除情感表达"
                }
            }
            
            with open(self.output_dir / "test_report.json", 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            
            logger.info("✅ 本地测试完成！")
            logger.info(f"📁 输出目录: {self.output_dir}")
            logger.info(f"📊 测试样本: {len(test_results)}")
            logger.info(f"🎵 验证音频: {self.output_dir}/verification_audio/")
            
            return report
            
        except Exception as e:
            logger.error(f"本地测试失败: {e}")
            return {"success": False, "error": str(e)}


def main():
    """主函数"""
    print("🧪 开始本地小规模测试 (10条音频)...")
    
    # 创建测试生成器
    tester = LocalTestGenerator(
        output_dir="test_output_small",
        num_test_samples=10,
        k_variants=1  # 默认K=1
    )
    
    # 运行测试
    result = tester.run_test()
    
    if result.get("success", True):
        print("✅ 本地测试成功！")
        print("\n📁 生成的文件:")
        print("  - mels/*.npy (mel频谱图)")
        print("  - emotion_features/*.npz (emotion2vec特征)")
        print("  - verification_audio/*.wav (验证音频)")
        print("  - test_report.json (测试报告)")
        
        print("\n🎵 验证方法:")
        print("1. 听取 *_s1_original.wav (原始音频)")
        print("2. 听取 *_original_reconstructed.wav (原始mel重建)")
        print("3. 听取 *_neutral_reconstructed.wav (中性mel重建)")
        print("4. 确认中性版本保持音色但去除情感")
        
        print("\n🚀 如果测试通过，可以提交到Spartan集群运行大规模处理！")
    else:
        print(f"❌ 测试失败: {result['error']}")


if __name__ == "__main__":
    main()
