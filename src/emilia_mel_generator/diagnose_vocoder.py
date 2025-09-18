"""
声码器诊断脚本
分别测试mel提取和声码器重建，找出问题根源
"""

import os
import torch
import torchaudio
import numpy as np
import soundfile as sf
from pathlib import Path
import logging
import librosa

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import bigvgan
    BIGVGAN_AVAILABLE = True
except ImportError:
    BIGVGAN_AVAILABLE = False


def test_mel_extraction():
    """测试mel频谱图提取是否正确"""
    logger.info("🔍 测试1: Mel频谱图提取")
    
    # 加载一个真实音频样本
    audio_path = "emilia_cache/EN/EN_B00000_S00000_W000000.wav"
    if not Path(audio_path).exists():
        logger.error("❌ 找不到缓存的音频文件")
        return False
    
    try:
        # 加载音频
        audio, sr = librosa.load(audio_path, sr=24000)
        logger.info(f"原始音频: 时长={len(audio)/sr:.2f}s, 范围=[{audio.min():.3f}, {audio.max():.3f}]")
        
        # 转换为torch tensor
        audio_tensor = torch.from_numpy(audio).float().unsqueeze(0)
        
        # 测试不同的mel提取参数
        configs = [
            {
                'name': 'BigVGAN_24kHz_官方',
                'sample_rate': 24000,
                'n_fft': 1024,
                'hop_length': 256,
                'win_length': 1024,
                'n_mels': 100,
                'f_min': 0,
                'f_max': 12000
            },
            {
                'name': 'Vevo_原始配置',
                'sample_rate': 24000,
                'n_fft': 1920,
                'hop_length': 480,
                'win_length': 1920,
                'n_mels': 128,
                'f_min': 0,
                'f_max': 12000
            }
        ]
        
        for config in configs:
            logger.info(f"\n--- 测试配置: {config['name']} ---")
            
            mel_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=config['sample_rate'],
                n_fft=config['n_fft'],
                win_length=config['win_length'],
                hop_length=config['hop_length'],
                n_mels=config['n_mels'],
                f_min=config['f_min'],
                f_max=config['f_max'],
                power=2.0,
                normalized=False
            )
            
            # 提取mel
            mel_linear = mel_transform(audio_tensor)
            mel_log = torch.log(torch.clamp(mel_linear, min=1e-8))
            
            logger.info(f"Mel形状: {mel_log.shape}")
            logger.info(f"Mel范围: [{mel_log.min():.2f}, {mel_log.max():.2f}]")
            logger.info(f"Mel均值: {mel_log.mean():.2f}, 标准差: {mel_log.std():.2f}")
            
            # 保存mel用于后续测试
            np.save(f"test_mel_{config['name'].lower()}.npy", mel_log.squeeze().numpy())
        
        logger.info("✅ Mel提取测试完成")
        return True
        
    except Exception as e:
        logger.error(f"❌ Mel提取测试失败: {e}")
        return False


def test_bigvgan_vocoder():
    """测试BigVGAN声码器是否正常工作"""
    logger.info("\n🔍 测试2: BigVGAN声码器")
    
    if not BIGVGAN_AVAILABLE:
        logger.error("❌ BigVGAN不可用")
        return False
    
    try:
        from bigvgan import BigVGAN
        
        # 初始化BigVGAN
        logger.info("初始化BigVGAN模型...")
        model = BigVGAN.from_pretrained('nvidia/bigvgan_v2_24khz_100band_256x')
        model.eval()
        
        # 测试1: 使用标准范围的随机mel
        logger.info("\n--- 测试随机mel输入 ---")
        test_mel_random = torch.randn(1, 100, 100) * 2.0  # 标准范围
        
        with torch.no_grad():
            audio_output = model(test_mel_random)
        
        logger.info(f"随机mel输入: {test_mel_random.shape}, 范围=[{test_mel_random.min():.2f}, {test_mel_random.max():.2f}]")
        logger.info(f"BigVGAN输出: {audio_output.shape}, 范围=[{audio_output.min():.3f}, {audio_output.max():.3f}]")
        
        # 保存随机测试音频
        sf.write("test_bigvgan_random.wav", audio_output.squeeze().numpy(), 24000)
        logger.info("保存随机测试音频: test_bigvgan_random.wav")
        
        # 测试2: 使用我们提取的mel
        if Path("test_mel_bigvgan_24khz_官方.npy").exists():
            logger.info("\n--- 测试真实mel输入 ---")
            real_mel = np.load("test_mel_bigvgan_24khz_官方.npy")
            real_mel_tensor = torch.from_numpy(real_mel).unsqueeze(0)
            
            logger.info(f"真实mel: {real_mel_tensor.shape}, 范围=[{real_mel_tensor.min():.2f}, {real_mel_tensor.max():.2f}]")
            
            # 测试不同的标准化方法
            normalization_methods = [
                {"name": "原始", "mel": real_mel_tensor},
                {"name": "限制范围", "mel": torch.clamp(real_mel_tensor, -5, 5)},
                {"name": "标准化", "mel": (real_mel_tensor - real_mel_tensor.mean()) / real_mel_tensor.std()},
                {"name": "LibriTTS风格", "mel": (real_mel_tensor + 4.0) / 3.0}
            ]
            
            for method in normalization_methods:
                try:
                    logger.info(f"\n  测试标准化方法: {method['name']}")
                    mel_input = method['mel']
                    logger.info(f"  输入范围: [{mel_input.min():.2f}, {mel_input.max():.2f}]")
                    
                    with torch.no_grad():
                        audio_out = model(mel_input)
                    
                    logger.info(f"  输出范围: [{audio_out.min():.3f}, {audio_out.max():.3f}]")
                    
                    # 保存测试音频
                    filename = f"test_bigvgan_{method['name']}.wav"
                    sf.write(filename, audio_out.squeeze().numpy(), 24000)
                    logger.info(f"  保存: {filename}")
                    
                    # 简单质量检查
                    audio_np = audio_out.squeeze().numpy()
                    rms = np.sqrt(np.mean(audio_np**2))
                    logger.info(f"  RMS: {rms:.4f}")
                    
                    if rms > 0.001 and rms < 1.0:
                        logger.info(f"  ✅ {method['name']} 方法音频质量正常")
                    else:
                        logger.warning(f"  ⚠️ {method['name']} 方法音频质量异常")
                        
                except Exception as e:
                    logger.error(f"  ❌ {method['name']} 方法失败: {e}")
        
        logger.info("✅ BigVGAN声码器测试完成")
        return True
        
    except Exception as e:
        logger.error(f"❌ BigVGAN声码器测试失败: {e}")
        return False


def test_audio_roundtrip():
    """测试完整的音频往返：音频→mel→音频"""
    logger.info("\n🔍 测试3: 音频往返测试")
    
    audio_path = "emilia_cache/EN/EN_B00000_S00000_W000000.wav"
    if not Path(audio_path).exists():
        logger.error("❌ 找不到测试音频")
        return False
    
    if not BIGVGAN_AVAILABLE:
        logger.error("❌ BigVGAN不可用")
        return False
    
    try:
        from bigvgan import BigVGAN
        
        # 加载音频和模型
        audio, sr = librosa.load(audio_path, sr=24000)
        audio_tensor = torch.from_numpy(audio).float().unsqueeze(0)
        
        model = BigVGAN.from_pretrained('nvidia/bigvgan_v2_24khz_100band_256x')
        model.eval()
        
        logger.info(f"原始音频: 时长={len(audio)/sr:.2f}s, RMS={np.sqrt(np.mean(audio**2)):.4f}")
        
        # 往返测试
        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=24000,
            n_fft=1024,
            hop_length=256,
            win_length=1024,
            n_mels=100,
            f_min=0,
            f_max=12000,
            power=2.0,
            normalized=False
        )
        
        # 步骤1: 音频 → mel
        mel_extracted = mel_transform(audio_tensor)
        mel_log = torch.log(torch.clamp(mel_extracted, min=1e-8))
        
        # 测试不同的标准化强度
        normalizations = [
            {"name": "无标准化", "mel": mel_log},
            {"name": "轻度标准化", "mel": torch.clamp(mel_log, -8, 8)},
            {"name": "中度标准化", "mel": (mel_log + 4.0) / 2.0},
            {"name": "强度标准化", "mel": (mel_log - mel_log.mean()) / mel_log.std()}
        ]
        
        for norm in normalizations:
            try:
                logger.info(f"\n  测试标准化: {norm['name']}")
                mel_input = norm['mel']
                
                if mel_input.dim() == 3:
                    mel_input = mel_input.squeeze(0)
                if mel_input.dim() == 2:
                    mel_input = mel_input.unsqueeze(0)
                
                logger.info(f"  Mel输入: {mel_input.shape}, 范围=[{mel_input.min():.2f}, {mel_input.max():.2f}]")
                
                # 步骤2: mel → 音频
                with torch.no_grad():
                    audio_reconstructed = model(mel_input)
                
                audio_recon_np = audio_reconstructed.squeeze().numpy()
                
                # 简单的音量调整
                if np.max(np.abs(audio_recon_np)) > 0:
                    audio_recon_np = audio_recon_np / np.max(np.abs(audio_recon_np)) * 0.3
                
                logger.info(f"  重建音频: 时长={len(audio_recon_np)/24000:.2f}s, RMS={np.sqrt(np.mean(audio_recon_np**2)):.4f}")
                
                # 保存往返测试结果
                filename = f"roundtrip_{norm['name']}.wav"
                sf.write(filename, audio_recon_np, 24000)
                logger.info(f"  保存往返测试: {filename}")
                
                # 质量评估
                rms = np.sqrt(np.mean(audio_recon_np**2))
                if 0.01 <= rms <= 0.5:
                    logger.info(f"  ✅ {norm['name']} 往返质量正常")
                else:
                    logger.warning(f"  ⚠️ {norm['name']} 往返质量异常 (RMS: {rms:.4f})")
                
            except Exception as e:
                logger.error(f"  ❌ {norm['name']} 往返失败: {e}")
        
        logger.info("✅ 音频往返测试完成")
        return True
        
    except Exception as e:
        logger.error(f"❌ 音频往返测试失败: {e}")
        return False


def main():
    """主诊断函数"""
    print("🔧 BigVGAN声码器诊断开始...")
    print("📋 将分别测试mel提取和声码器重建")
    
    # 测试1: Mel提取
    mel_ok = test_mel_extraction()
    
    # 测试2: BigVGAN声码器
    vocoder_ok = test_bigvgan_vocoder()
    
    # 测试3: 完整往返
    roundtrip_ok = test_audio_roundtrip()
    
    print("\n" + "="*50)
    print("🎯 诊断结果总结:")
    print(f"Mel提取: {'✅ 正常' if mel_ok else '❌ 异常'}")
    print(f"BigVGAN声码器: {'✅ 正常' if vocoder_ok else '❌ 异常'}")
    print(f"音频往返: {'✅ 正常' if roundtrip_ok else '❌ 异常'}")
    
    if mel_ok and vocoder_ok and roundtrip_ok:
        print("\n🎉 所有组件正常工作！")
        print("🎵 请听取生成的测试音频文件:")
        print("  - test_bigvgan_*.wav (不同标准化方法)")
        print("  - roundtrip_*.wav (完整往返测试)")
        print("  - test_mel_*.npy (mel频谱图文件)")
    else:
        print("\n⚠️ 发现问题，请检查具体的错误信息")
    
    print("\n💡 诊断建议:")
    print("1. 听取 test_bigvgan_random.wav - 如果正常说明BigVGAN工作正常")
    print("2. 听取 roundtrip_*.wav - 找出最佳的标准化方法")
    print("3. 对比不同配置的mel频谱图统计信息")


if __name__ == "__main__":
    main()
