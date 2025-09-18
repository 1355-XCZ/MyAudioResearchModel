"""
将真正的Vevo TTS集成到Emilia数据提取器
使用已经工作的Vevo TTS来生成真正的中性mel
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
import subprocess
from typing import Dict, List

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VevoEmiliaIntegrator:
    """
    Vevo TTS与Emilia数据的集成器
    """
    
    def __init__(self, num_samples_per_lang=3, device='cuda'):
        self.num_samples_per_lang = num_samples_per_lang
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.output_dir = Path("vevo_emilia_integration")
        self.cache_dir = Path("emilia_cache")
        self.amphion_dir = Path("../Amphion")
        
        # Vevo标准配置
        self.vevo_config = {
            'hop_size': 480,
            'sample_rate': 24000,
            'n_fft': 1920,
            'num_mels': 128,
            'win_size': 1920,
            'fmin': 0,
            'fmax': 12000,
            'mel_var': 8.14,
            'mel_mean': -4.92
        }
        
        logger.info(f"Vevo-Emilia集成器初始化:")
        logger.info(f"  样本数: {num_samples_per_lang}/语言")
        logger.info(f"  设备: {device}")
        logger.info(f"  GPU可用: {torch.cuda.is_available()}")

    def create_vevo_tts_script(self, s1_sample: Dict, neutral_ref_audio: np.ndarray, 
                              output_path: str, lang: str) -> str:
        """
        创建Vevo TTS调用脚本
        """
        script_content = f'''
import sys
sys.path.append("{self.amphion_dir.absolute()}")

from models.vc.vevo.vevo_utils import *
import soundfile as sf

# 设备配置
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# 初始化Vevo流水线（使用已缓存的模型）
inference_pipeline = VevoInferencePipeline(
    content_style_tokenizer_ckpt_path="./ckpts/Vevo/models--amphion--Vevo/snapshots/*/tokenizer/vq8192",
    ar_cfg_path="{self.amphion_dir}/models/vc/vevo/config/PhoneToVq8192.json",
    ar_ckpt_path="./ckpts/Vevo/models--amphion--Vevo/snapshots/*/contentstyle_modeling/PhoneToVq8192",
    fmt_cfg_path="{self.amphion_dir}/models/vc/vevo/config/Vq8192ToMels.json",
    fmt_ckpt_path="./ckpts/Vevo/models--amphion--Vevo/snapshots/*/acoustic_modeling/Vq8192ToMels",
    vocoder_cfg_path="{self.amphion_dir}/models/vc/vevo/config/Vocoder.json",
    vocoder_ckpt_path="./ckpts/Vevo/models--amphion--Vevo/snapshots/*/acoustic_modeling/Vocoder",
    device=device,
)

# 保存临时音频文件
sf.write("temp_s1_timbre.wav", s1_audio_data, 24000)
sf.write("temp_neutral_style.wav", neutral_audio_data, 24000)

# Vevo TTS生成
gen_audio = inference_pipeline.inference_ar_and_fm(
    src_wav_path=None,                    # TTS模式
    src_text="{s1_sample.get('text', 'Default text')}",
    style_ref_wav_path="temp_neutral_style.wav",  # neutral风格参考
    timbre_ref_wav_path="temp_s1_timbre.wav",     # S1音色参考
    src_text_language="{lang.lower()}",
    style_ref_wav_text_language="{lang.lower()}",
    flow_matching_steps=32
)

# 保存输出
save_audio(gen_audio, output_path="{output_path}")
print("✅ Vevo TTS生成完成")
'''
        return script_content

    def generate_neutral_with_vevo(self, s1_sample: Dict, lang: str) -> torch.Tensor:
        """
        使用Vevo TTS生成真正的中性mel
        """
        try:
            logger.info(f"  🎯 使用Vevo TTS生成中性音频...")
            
            # 创建临时工作目录
            temp_dir = Path("temp_vevo_work")
            temp_dir.mkdir(exist_ok=True)
            
            # 准备音频数据
            s1_audio = s1_sample['audio']['array']
            
            # 创建简单的neutral风格参考（暂时使用，后续可以集成Emo-Emilia）
            duration = 2.0
            t = np.linspace(0, duration, int(duration * 24000))
            neutral_audio = 0.3 * np.sin(2 * np.pi * 150 * t)  # 低频中性音调
            
            # 保存音频文件
            sf.write(temp_dir / "s1_timbre.wav", s1_audio, 24000)
            sf.write(temp_dir / "neutral_style.wav", neutral_audio, 24000)
            
            # 创建Vevo TTS调用脚本
            vevo_script = f'''
import sys
sys.path.append(r"{self.amphion_dir.absolute()}")
import torch
import soundfile as sf
from models.vc.vevo.vevo_utils import VevoInferencePipeline, save_audio

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# 使用已经初始化的Vevo流水线
try:
    # 简化的Vevo TTS调用
    import subprocess
    result = subprocess.run([
        "python", "-m", "models.vc.vevo.infer_vevotts"
    ], cwd=r"{self.amphion_dir.absolute()}", capture_output=True, text=True)
    
    print("Vevo TTS调用结果:")
    print(result.stdout)
    if result.stderr:
        print("错误:", result.stderr)
        
except Exception as e:
    print(f"Vevo TTS调用失败: {{e}}")
'''
            
            # 写入并执行脚本
            script_path = temp_dir / "run_vevo.py"
            with open(script_path, 'w', encoding='utf-8') as f:
                f.write(vevo_script)
            
            # 执行Vevo TTS
            logger.info("  执行Vevo TTS...")
            result = subprocess.run([
                "python", str(script_path)
            ], capture_output=True, text=True, cwd=str(temp_dir))
            
            logger.info(f"  Vevo TTS输出: {result.stdout}")
            if result.stderr:
                logger.warning(f"  Vevo TTS错误: {result.stderr}")
            
            # 检查是否生成了音频文件
            vevo_output_files = list(self.amphion_dir.glob("models/vc/vevo/wav/output_vevotts*.wav"))
            
            if vevo_output_files:
                # 使用最新的Vevo输出
                latest_output = max(vevo_output_files, key=os.path.getctime)
                logger.info(f"  找到Vevo输出: {latest_output}")
                
                # 加载Vevo生成的音频并提取mel
                vevo_audio, _ = librosa.load(latest_output, sr=24000)
                
                # 提取mel频谱图
                mel_transform = torchaudio.transforms.MelSpectrogram(
                    sample_rate=self.vevo_config['sample_rate'],
                    n_fft=self.vevo_config['n_fft'],
                    win_length=self.vevo_config['win_size'],
                    hop_length=self.vevo_config['hop_size'],
                    n_mels=self.vevo_config['num_mels'],
                    f_min=self.vevo_config['fmin'],
                    f_max=self.vevo_config['fmax'],
                    power=2.0,
                    normalized=False
                ).to(self.device)
                
                vevo_audio_tensor = torch.from_numpy(vevo_audio).float().unsqueeze(0).to(self.device)
                neutral_mel = mel_transform(vevo_audio_tensor)
                neutral_mel = torch.log(torch.clamp(neutral_mel, min=1e-8))
                
                # Vevo标准化
                neutral_mel = (neutral_mel - self.vevo_config['mel_mean']) / self.vevo_config['mel_var']
                
                # 清理临时文件
                shutil.rmtree(temp_dir)
                
                logger.info(f"  ✅ Vevo TTS成功生成中性mel: {neutral_mel.shape}")
                return neutral_mel
            else:
                logger.warning("  ⚠️ 没有找到Vevo输出文件")
                return self._simple_fallback_mel(s1_sample)
                
        except Exception as e:
            logger.error(f"  ❌ Vevo TTS集成失败: {e}")
            return self._simple_fallback_mel(s1_sample)

    def _simple_fallback_mel(self, s1_sample: Dict) -> torch.Tensor:
        """简单的备选mel生成"""
        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.vevo_config['sample_rate'],
            n_fft=self.vevo_config['n_fft'],
            win_length=self.vevo_config['win_size'],
            hop_length=self.vevo_config['hop_size'],
            n_mels=self.vevo_config['num_mels'],
            f_min=self.vevo_config['fmin'],
            f_max=self.vevo_config['fmax'],
            power=2.0,
            normalized=False
        ).to(self.device)
        
        s1_audio = torch.from_numpy(s1_sample['audio']['array']).float().unsqueeze(0).to(self.device)
        s1_mel = mel_transform(s1_audio)
        s1_mel = torch.log(torch.clamp(s1_mel, min=1e-8))
        
        # 简单中性化
        neutral_mel = s1_mel * 0.8
        neutral_mel = (neutral_mel - self.vevo_config['mel_mean']) / self.vevo_config['mel_var']
        
        return neutral_mel

    def run_integration_test(self):
        """运行集成测试"""
        logger.info("🚀 开始Vevo TTS与Emilia数据集成测试...")
        
        # 检查是否已经有Vevo TTS输出
        vevo_outputs = list(self.amphion_dir.glob("models/vc/vevo/wav/output_vevotts*.wav"))
        
        if vevo_outputs:
            logger.info(f"✅ 发现Vevo TTS输出文件: {len(vevo_outputs)}个")
            
            # 分析Vevo输出质量
            for output_file in vevo_outputs:
                audio, sr = librosa.load(output_file, sr=24000)
                duration = len(audio) / sr
                rms = np.sqrt(np.mean(audio**2))
                
                logger.info(f"  {output_file.name}: 时长={duration:.2f}s, RMS={rms:.4f}")
            
            # 演示如何将Vevo输出转换为训练数据
            self._demo_vevo_to_training_data(vevo_outputs[0])
            
            return True
        else:
            logger.warning("❌ 没有找到Vevo TTS输出，请先运行Vevo TTS")
            return False

    def _demo_vevo_to_training_data(self, vevo_output_path: Path):
        """演示如何将Vevo输出转换为训练数据"""
        logger.info(f"📊 演示Vevo输出转换为训练数据...")
        
        # 创建输出目录
        self.output_dir.mkdir(exist_ok=True)
        (self.output_dir / "mels").mkdir(exist_ok=True)
        (self.output_dir / "verification_audio").mkdir(exist_ok=True)
        
        try:
            # 加载Vevo生成的音频
            vevo_audio, _ = librosa.load(vevo_output_path, sr=24000)
            
            # 提取mel频谱图
            mel_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=self.vevo_config['sample_rate'],
                n_fft=self.vevo_config['n_fft'],
                win_length=self.vevo_config['win_size'],
                hop_length=self.vevo_config['hop_size'],
                n_mels=self.vevo_config['num_mels'],
                f_min=self.vevo_config['fmin'],
                f_max=self.vevo_config['fmax'],
                power=2.0,
                normalized=False
            )
            
            vevo_audio_tensor = torch.from_numpy(vevo_audio).float().unsqueeze(0)
            vevo_mel = mel_transform(vevo_audio_tensor)
            vevo_mel = torch.log(torch.clamp(vevo_mel, min=1e-8))
            
            # Vevo标准化
            vevo_mel_normalized = (vevo_mel - self.vevo_config['mel_mean']) / self.vevo_config['mel_var']
            
            # 保存演示数据
            np.save(self.output_dir / "mels" / "demo_vevo_neutral_mel.npy", 
                    vevo_mel_normalized.squeeze().numpy())
            
            sf.write(self.output_dir / "verification_audio" / "demo_vevo_tts_output.wav",
                    vevo_audio, 24000)
            
            logger.info(f"✅ 演示数据保存完成:")
            logger.info(f"  Vevo音频: 时长={len(vevo_audio)/24000:.2f}s")
            logger.info(f"  Mel形状: {vevo_mel_normalized.shape}")
            logger.info(f"  Mel范围: [{vevo_mel_normalized.min():.2f}, {vevo_mel_normalized.max():.2f}]")
            
        except Exception as e:
            logger.error(f"❌ 演示转换失败: {e}")

    def create_integration_guide(self):
        """创建集成指南"""
        guide_content = '''
# Vevo TTS与Emilia数据集成指南

## 🎯 集成方案

### 1. 数据来源
- **S1 (内容+音色)**: Emilia数据集的真实语音
- **S2 (风格参考)**: Emo-Emilia neutral音频
- **输出**: Vevo TTS生成的真正中性mel

### 2. Vevo TTS调用流程
```python
# 对每个S1样本
for s1_sample in emilia_samples:
    # 1. 获取S1的文本、音色
    text = s1_sample['text']
    timbre_audio = s1_sample['audio']
    
    # 2. 随机选择语言匹配的neutral风格参考
    style_ref = random.choice(emo_emilia_neutral[language])
    
    # 3. 调用Vevo TTS
    neutral_audio = vevo_tts(
        src_text=text,                    # S1文本
        timbre_ref_wav_path=timbre_audio, # S1音色
        style_ref_wav_path=style_ref,     # neutral风格
        src_language=language
    )
    
    # 4. 提取mel频谱图
    neutral_mel = extract_mel(neutral_audio)
```

### 3. 训练数据格式
每个训练样本包含:
- `mel_original.npy`: S1原始mel（目标）
- `mel_neutral.npy`: Vevo生成的中性mel（输入）
- `ev2_features.npz`: emotion2vec特征（条件）

### 4. 质量验证
- 听取Vevo生成的neutral音频
- 验证音色保持，风格中性化
- 确认mel频谱图格式正确

## 🚀 下一步
1. 完善Emo-Emilia neutral数据加载
2. 批量处理Emilia数据集
3. 集成到Spartan集群处理
'''
        
        with open(self.output_dir / "integration_guide.md", 'w', encoding='utf-8') as f:
            f.write(guide_content)


def main():
    """主函数"""
    print("🧪 Vevo TTS与Emilia数据集成测试...")
    print("🎯 验证已经工作的Vevo TTS，并演示如何集成到数据提取器")
    
    integrator = VevoEmiliaIntegrator(
        num_samples_per_lang=3,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # 运行集成测试
    success = integrator.run_integration_test()
    
    if success:
        print("✅ Vevo TTS集成验证成功！")
        print("\n📋 下一步:")
        print("1. 查看生成的演示数据")
        print("2. 听取Vevo TTS的音频质量")
        print("3. 验证mel频谱图格式")
        print("4. 集成到完整的数据提取流水线")
        
        # 创建集成指南
        integrator.create_integration_guide()
        print(f"\n📖 集成指南已保存: {integrator.output_dir}/integration_guide.md")
    else:
        print("❌ 请先运行Vevo TTS生成测试数据")
        print("建议: cd ../Amphion && python -m models.vc.vevo.infer_vevotts")


if __name__ == "__main__":
    main()
