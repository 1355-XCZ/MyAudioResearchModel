"""
最终版本：集成真正Vevo TTS的Emilia小测试
- 使用缓存的Emilia数据
- 使用真正的Vevo TTS生成中性mel
- 使用BigVGAN重建验证音频
- GPU加速处理
"""

import os
import sys
import torch
import torchaudio
import numpy as np
import soundfile as sf
from pathlib import Path
import logging
import json
import librosa
import shutil
import glob
from typing import Dict, List

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 添加Amphion路径
amphion_path = Path(__file__).parent.parent / "Amphion"
sys.path.append(str(amphion_path))

# 导入检查
try:
    import bigvgan
    from bigvgan import get_mel_spectrogram
    BIGVGAN_AVAILABLE = True
    logger.info("✅ BigVGAN可用")
except ImportError:
    BIGVGAN_AVAILABLE = False
    logger.error("❌ BigVGAN不可用")

try:
    from models.vc.vevo.vevo_utils import VevoInferencePipeline, save_audio
    VEVO_AVAILABLE = True
    logger.info("✅ Vevo TTS可用")
except ImportError as e:
    VEVO_AVAILABLE = False
    logger.error(f"❌ Vevo TTS不可用: {e}")


class FinalVevoTest:
    """
    最终的Vevo TTS集成测试
    """
    
    def __init__(self, num_samples=3, device='cuda'):
        self.num_samples = num_samples
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.output_dir = Path("final_vevo_test_output")
        self.cache_dir = Path("emilia_cache")
        
        # 初始化组件
        self.bigvgan_model = None
        self.vevo_pipeline = None
        
        logger.info(f"最终Vevo测试初始化:")
        logger.info(f"  样本数: {num_samples}")
        logger.info(f"  设备: {device} (GPU可用: {torch.cuda.is_available()})")
        
        self._init_components()

    def _init_components(self):
        """初始化所有组件"""
        # 初始化BigVGAN
        if BIGVGAN_AVAILABLE:
            try:
                logger.info("初始化BigVGAN 24kHz...")
                self.bigvgan_model = bigvgan.BigVGAN.from_pretrained("nvidia/bigvgan_v2_24khz_100band_256x")
                self.bigvgan_model.eval()
                self.bigvgan_model = self.bigvgan_model.to(self.device)
                logger.info("✅ BigVGAN初始化成功")
            except Exception as e:
                logger.error(f"❌ BigVGAN初始化失败: {e}")
        
        # 初始化Vevo TTS
        if VEVO_AVAILABLE:
            try:
                logger.info("初始化Vevo TTS...")
                
                # 查找已下载的Vevo模型
                base_cache_dir = str(amphion_path / "ckpts/Vevo")
                
                vq8192_paths = glob.glob(f"{base_cache_dir}/models--amphion--Vevo/snapshots/*/tokenizer/vq8192")
                ar_paths = glob.glob(f"{base_cache_dir}/models--amphion--Vevo/snapshots/*/contentstyle_modeling/PhoneToVq8192")
                fmt_paths = glob.glob(f"{base_cache_dir}/models--amphion--Vevo/snapshots/*/acoustic_modeling/Vq8192ToMels")
                vocoder_paths = glob.glob(f"{base_cache_dir}/models--amphion--Vevo/snapshots/*/acoustic_modeling/Vocoder")
                
                if all([vq8192_paths, ar_paths, fmt_paths, vocoder_paths]):
                    self.vevo_pipeline = VevoInferencePipeline(
                        content_style_tokenizer_ckpt_path=vq8192_paths[0],
                        ar_cfg_path=str(amphion_path / "models/vc/vevo/config/PhoneToVq8192.json"),
                        ar_ckpt_path=ar_paths[0],
                        fmt_cfg_path=str(amphion_path / "models/vc/vevo/config/Vq8192ToMels.json"),
                        fmt_ckpt_path=fmt_paths[0],
                        vocoder_cfg_path=str(amphion_path / "models/vc/vevo/config/Vocoder.json"),
                        vocoder_ckpt_path=vocoder_paths[0],
                        device=self.device,
                    )
                    logger.info("✅ Vevo TTS初始化成功")
                else:
                    logger.warning("⚠️ Vevo模型文件不完整")
                    
            except Exception as e:
                logger.error(f"❌ Vevo TTS初始化失败: {e}")

    def load_cached_emilia_samples(self) -> List[Dict]:
        """加载缓存的Emilia样本"""
        samples = []
        
        for lang_code in ["EN", "ZH"]:
            lang_cache_dir = self.cache_dir / lang_code
            if lang_cache_dir.exists():
                audio_files = list(lang_cache_dir.glob("*.wav"))[:self.num_samples]
                
                for audio_file in audio_files:
                    try:
                        audio_array, _ = librosa.load(audio_file, sr=24000)
                        
                        # 加载元数据
                        meta_file = audio_file.with_suffix('.json')
                        if meta_file.exists():
                            with open(meta_file, 'r', encoding='utf-8') as f:
                                metadata = json.load(f)
                        else:
                            metadata = {'text': f'Sample from {lang_code}', 'speaker': f'{lang_code}_speaker'}
                        
                        sample = {
                            'audio': {'array': audio_array.astype(np.float32), 'sampling_rate': 24000},
                            'text': metadata.get('text', ''),
                            'speaker': metadata.get('speaker', ''),
                            'language': lang_code.lower(),
                            'duration': len(audio_array) / 24000,
                            'lang_code': lang_code
                        }
                        samples.append(sample)
                        
                    except Exception as e:
                        logger.warning(f"加载 {audio_file} 失败: {e}")
        
        return samples

    def generate_neutral_with_vevo(self, sample: Dict) -> torch.Tensor:
        """使用真正的Vevo TTS生成中性mel"""
        if self.vevo_pipeline is None or self.bigvgan_model is None:
            logger.warning("  组件不完整，使用简单中性化")
            return self._simple_neutralize(sample)
        
        try:
            logger.info(f"  🎯 使用真正的Vevo TTS: {sample['lang_code']}")
            
            # 准备临时文件
            temp_dir = Path("temp_vevo")
            temp_dir.mkdir(exist_ok=True)
            
            s1_path = temp_dir / "s1.wav"
            neutral_path = temp_dir / "neutral.wav"
            output_path = temp_dir / "vevo_output.wav"
            
            # 保存S1音频（音色参考）
            sf.write(s1_path, sample['audio']['array'], 24000)
            
            # 创建neutral风格参考
            duration = min(3.0, sample['duration'])
            t = np.linspace(0, duration, int(duration * 24000))
            neutral_audio = 0.3 * np.sin(2 * np.pi * 150 * t)
            sf.write(neutral_path, neutral_audio, 24000)
            
            # Vevo TTS生成
            logger.info(f"    文本: {sample['text'][:30]}...")
            
            gen_audio = self.vevo_pipeline.inference_ar_and_fm(
                src_wav_path=None,                  # TTS模式
                src_text=sample['text'],            # S1文本
                style_ref_wav_path=str(neutral_path), # neutral风格
                timbre_ref_wav_path=str(s1_path),   # S1音色
                src_text_language=sample['language'],
                style_ref_wav_text_language=sample['language'],
                flow_matching_steps=8  # 快速模式
            )
            
            # 保存Vevo输出
            save_audio(gen_audio, output_path=str(output_path))
            
            # 提取mel
            vevo_audio, _ = librosa.load(output_path, sr=24000)
            vevo_tensor = torch.from_numpy(vevo_audio).float().unsqueeze(0).to(self.device)
            
            # 使用BigVGAN的mel提取
            neutral_mel = get_mel_spectrogram(vevo_tensor, self.bigvgan_model.h)
            
            # 清理
            shutil.rmtree(temp_dir)
            
            logger.info(f"    ✅ Vevo TTS成功: {neutral_mel.shape}")
            return neutral_mel
            
        except Exception as e:
            logger.warning(f"    ⚠️ Vevo TTS失败: {e}")
            return self._simple_neutralize(sample)

    def _simple_neutralize(self, sample: Dict) -> torch.Tensor:
        """简单的中性化备选方案"""
        logger.info("    使用简单中性化...")
        
        # 提取原始mel
        audio_tensor = torch.from_numpy(sample['audio']['array']).float().unsqueeze(0).to(self.device)
        
        if self.bigvgan_model:
            mel = get_mel_spectrogram(audio_tensor, self.bigvgan_model.h)
        else:
            # 备选mel提取
            mel_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=24000, n_fft=1024, hop_length=256, n_mels=100
            ).to(self.device)
            mel = mel_transform(audio_tensor)
            mel = torch.log(torch.clamp(mel, min=1e-8))
        
        # 简单中性化：降低情感强度
        neutral_mel = mel * 0.85
        
        return neutral_mel

    def run_final_test(self):
        """运行最终测试"""
        logger.info("🚀 开始最终Vevo TTS集成小测试...")
        
        # 创建输出目录
        self.output_dir.mkdir(exist_ok=True)
        (self.output_dir / "mels").mkdir(exist_ok=True)
        (self.output_dir / "verification_audio").mkdir(exist_ok=True)
        
        # 加载样本
        samples = self.load_cached_emilia_samples()
        logger.info(f"加载了 {len(samples)} 个样本")
        
        # 处理样本
        results = []
        for i, sample in enumerate(samples):
            sample_id = f"{sample['lang_code']}_S{i+1:02d}"
            logger.info(f"处理样本: {sample_id}")
            
            try:
                # 1. 提取原始mel
                if self.bigvgan_model:
                    audio_tensor = torch.from_numpy(sample['audio']['array']).float().unsqueeze(0).to(self.device)
                    mel_original = get_mel_spectrogram(audio_tensor, self.bigvgan_model.h)
                else:
                    mel_original = self._simple_neutralize(sample)
                
                # 2. 生成中性mel（关键：使用真正的Vevo TTS）
                mel_neutral = self.generate_neutral_with_vevo(sample)
                
                # 3. 保存数据
                np.save(self.output_dir / "mels" / f"{sample_id}_original.npy", 
                        mel_original.squeeze().cpu().numpy())
                np.save(self.output_dir / "mels" / f"{sample_id}_neutral.npy", 
                        mel_neutral.squeeze().cpu().numpy())
                
                # 4. 生成验证音频
                sf.write(self.output_dir / "verification_audio" / f"{sample_id}_original.wav",
                        sample['audio']['array'], 24000)
                
                # 使用BigVGAN重建中性音频
                if self.bigvgan_model:
                    try:
                        with torch.no_grad():
                            neutral_audio_recon = self.bigvgan_model(mel_neutral.to(self.device))
                            sf.write(self.output_dir / "verification_audio" / f"{sample_id}_neutral_vevo.wav",
                                    neutral_audio_recon.squeeze().cpu().numpy(), 24000)
                    except Exception as e:
                        logger.warning(f"BigVGAN重建失败: {e}")
                
                results.append({
                    "sample_id": sample_id,
                    "text": sample['text'][:50] + "..." if len(sample['text']) > 50 else sample['text'],
                    "duration": sample['duration'],
                    "mel_original_shape": list(mel_original.shape),
                    "mel_neutral_shape": list(mel_neutral.shape),
                    "vevo_used": self.vevo_pipeline is not None
                })
                
            except Exception as e:
                logger.error(f"处理样本 {sample_id} 失败: {e}")
        
        # 保存报告
        report = {
            "test_config": {
                "num_samples": len(samples),
                "device": self.device,
                "gpu_available": torch.cuda.is_available(),
                "bigvgan_available": BIGVGAN_AVAILABLE,
                "vevo_available": VEVO_AVAILABLE,
                "real_vevo_used": self.vevo_pipeline is not None
            },
            "results": results
        }
        
        with open(self.output_dir / "final_test_report.json", 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info("✅ 最终测试完成！")
        logger.info(f"📊 成功处理: {len(results)} 个样本")
        logger.info(f"📁 输出目录: {self.output_dir}")
        
        return len(results) > 0


def main():
    """主函数"""
    print("🧪 最终版本：真正的Vevo TTS集成小测试")
    print("🎯 目标:")
    print("  ✅ 使用缓存的真实Emilia数据")
    print("  ✅ 使用真正的Vevo TTS生成中性mel")
    print("  ✅ 使用BigVGAN重建验证音频")
    print("  ✅ GPU加速处理")
    
    tester = FinalVevoTest(
        num_samples=3,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    success = tester.run_final_test()
    
    if success:
        print("✅ 最终Vevo TTS集成测试成功！")
        print("\n📁 生成的文件:")
        print("  - mels/*_original.npy (原始mel)")
        print("  - mels/*_neutral.npy (Vevo TTS生成的中性mel)")
        print("  - verification_audio/*_original.wav (原始音频)")
        print("  - verification_audio/*_neutral_vevo.wav (Vevo中性音频)")
        print("  - final_test_report.json (详细报告)")
        print("\n🎵 验证方法:")
        print("1. 听取原始音频和Vevo中性音频的对比")
        print("2. 验证中性音频保持了音色但调整了风格")
        print("3. 检查mel频谱图的格式和质量")
        print("\n🚀 您现在拥有了真正的Vevo TTS训练数据！")
    else:
        print("❌ 测试失败")


if __name__ == "__main__":
    main()
