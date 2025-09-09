"""
完整的训练和推理流水线
支持模块化的实验设计，包含VQ-VAE情感量化实验
"""
import os
import yaml
import torch
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt

from ..core.interfaces import Pipeline, DataProcessor, StageAModel, StageBModel, Vocoder, EmotionQuantizer, AudioData
from ..data_processing.audio_processor import WhisperEmotionProcessor, DatasetLoader
from ..models.stage_a import FastSpeech2StageA
from ..models.stage_b import TwoStageEmotionModel
from ..models.vocoder import VocoderFactory
from ..models.emotion_quantizer import create_emotion_quantizer


class EmotionAudioPipeline(Pipeline):
    """
    完整的情感音频建模流水线
    支持训练、推理和VQ-VAE码本实验
    """
    
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.logger = self._setup_logger()
        
        # 初始化组件
        self.data_processor = self._create_data_processor()
        self.stage_a_model = self._create_stage_a_model()
        self.stage_b_model = self._create_stage_b_model()
        self.vocoder = self._create_vocoder()
        self.emotion_quantizer = self._create_emotion_quantizer()
        
        # 数据加载器
        self.dataset_loader = DatasetLoader(self.config.get('dataset', {}))
        
        # 输出目录
        self.output_dir = self.config.get('experiment', {}).get('output_dir', 'outputs')
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 调用父类构造函数
        super().__init__(
            data_processor=self.data_processor,
            stage_a_model=self.stage_a_model,
            stage_b_model=self.stage_b_model,
            vocoder=self.vocoder,
            emotion_quantizer=self.emotion_quantizer
        )
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """加载配置文件"""
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def _setup_logger(self) -> logging.Logger:
        """设置日志"""
        logger = logging.getLogger('EmotionAudioPipeline')
        logger.setLevel(getattr(logging, self.config.get('experiment', {}).get('log_level', 'INFO')))
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _create_data_processor(self) -> DataProcessor:
        """创建数据处理器"""
        return WhisperEmotionProcessor(self.config.get('data_processing', {}))
    
    def _create_stage_a_model(self) -> StageAModel:
        """创建阶段A模型"""
        stage_a_config = self.config.get('stage_a', {})
        model_type = stage_a_config.get('model_type', 'fastspeech2')
        
        if model_type.lower() == 'fastspeech2':
            return FastSpeech2StageA(stage_a_config)
        else:
            raise ValueError(f"不支持的阶段A模型类型: {model_type}")
    
    def _create_stage_b_model(self) -> StageBModel:
        """创建阶段B模型"""
        stage_b_config = self.config.get('stage_b', {})
        model_type = stage_b_config.get('model_type', 'two_stage_emotion')
        
        if model_type.lower() == 'two_stage_emotion':
            return TwoStageEmotionModel(stage_b_config)
        else:
            raise ValueError(f"不支持的阶段B模型类型: {model_type}")
    
    def _create_vocoder(self) -> Vocoder:
        """创建声码器"""
        vocoder_config = self.config.get('vocoder', {})
        model_type = vocoder_config.get('model_type', 'hifigan')
        
        return VocoderFactory.create_vocoder(model_type, vocoder_config)
    
    def _create_emotion_quantizer(self) -> Optional[EmotionQuantizer]:
        """创建情感量化器"""
        quantizer_config = self.config.get('emotion_quantizer', {})
        
        if not quantizer_config.get('enabled', False):
            return None
        
        quantizer_type = quantizer_config.get('type', 'vqvae')
        return create_emotion_quantizer(quantizer_type, quantizer_config)
    
    def train(self, config: Dict[str, Any]) -> None:
        """训练完整流水线"""
        self.logger.info("开始训练情感音频建模流水线")
        
        # 加载训练数据
        train_data = self.dataset_loader.load_train_data()
        self.logger.info(f"加载了 {len(train_data)} 个训练样本")
        
        if len(train_data) == 0:
            self.logger.warning("没有找到训练数据，跳过训练")
            return
        
        # 预处理数据
        processed_data = self._preprocess_training_data(train_data)
        
        # 阶段A训练
        self._train_stage_a(processed_data)
        
        # 阶段B训练
        self._train_stage_b(processed_data)
        
        self.logger.info("训练完成")
    
    def _preprocess_training_data(self, train_data: List[AudioData]) -> List[Dict[str, Any]]:
        """预处理训练数据"""
        self.logger.info("预处理训练数据...")
        processed_data = []
        
        for audio_data in tqdm(train_data, desc="预处理音频"):
            try:
                # 使用数据处理器提取特征
                processed = self.data_processor.process_audio(audio_data)
                
                # 提取目标Mel频谱
                target_mel = self._extract_mel_spectrogram(audio_data)
                
                processed_data.append({
                    'phonemes': processed.phonemes,
                    'emotion_features': processed.emotion_features,
                    'target_mel': target_mel,
                    'audio_path': processed.audio_path
                })
            except Exception as e:
                self.logger.warning(f"处理音频失败 {audio_data.file_path}: {e}")
                continue
        
        self.logger.info(f"成功预处理 {len(processed_data)} 个样本")
        return processed_data
    
    def _extract_mel_spectrogram(self, audio_data: AudioData) -> np.ndarray:
        """从音频提取Mel频谱（目标）"""
        import librosa
        
        # 提取Mel频谱
        mel_spec = librosa.feature.melspectrogram(
            y=audio_data.waveform,
            sr=audio_data.sample_rate,
            n_mels=80,
            hop_length=256,
            win_length=1024,
            fmin=0,
            fmax=8000
        )
        
        # 转换为对数尺度
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        return mel_spec_db.T  # (time, mel_dim)
    
    def _train_stage_a(self, processed_data: List[Dict[str, Any]]) -> None:
        """训练阶段A"""
        self.logger.info("开始训练阶段A (音素 -> M0)")
        
        stage_a_config = self.config.get('stage_a', {})
        training_config = stage_a_config.get('training', {})
        
        epochs = training_config.get('epochs', 100)
        batch_size = training_config.get('batch_size', 32)
        
        for epoch in range(epochs):
            epoch_losses = []
            
            # 创建批次
            for i in range(0, len(processed_data), batch_size):
                batch = processed_data[i:i+batch_size]
                
                # 准备批次数据
                batch_data = {
                    'phonemes': [item['phonemes'] for item in batch],
                    'mel_spectrograms': torch.stack([
                        torch.tensor(item['target_mel'], dtype=torch.float32) 
                        for item in batch
                    ])
                }
                
                # 训练步骤
                losses = self.stage_a_model.train_step(batch_data)
                epoch_losses.append(losses['loss'])
            
            avg_loss = np.mean(epoch_losses)
            self.logger.info(f"阶段A Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
            
            # 保存检查点
            if (epoch + 1) % training_config.get('checkpoint_interval', 10) == 0:
                checkpoint_path = os.path.join(self.output_dir, f'stage_a_epoch_{epoch+1}.pth')
                self.stage_a_model.save_checkpoint(checkpoint_path)
    
    def _train_stage_b(self, processed_data: List[Dict[str, Any]]) -> None:
        """训练阶段B"""
        self.logger.info("开始训练阶段B")
        
        stage_b_config = self.config.get('stage_b', {})
        training_config = stage_b_config.get('training', {})
        
        epochs = training_config.get('epochs', 50)
        batch_size = training_config.get('batch_size', 16)
        
        # 先训练B1
        self._train_b1(processed_data, epochs, batch_size)
        
        # 如果启用了B2，再训练B2
        if self.stage_b_model.enable_b2:
            self._train_b2(processed_data, epochs // 2, batch_size)
    
    def _train_b1(self, processed_data: List[Dict[str, Any]], epochs: int, batch_size: int) -> None:
        """训练B1阶段"""
        self.logger.info("训练B1阶段 (M0 + 情感 -> M1)")
        
        for epoch in range(epochs):
            epoch_losses = []
            
            for i in range(0, len(processed_data), batch_size):
                batch = processed_data[i:i+batch_size]
                
                # 使用阶段A生成M0
                batch_m0 = []
                for item in batch:
                    m0_output = self.stage_a_model.forward(item['phonemes'])
                    batch_m0.append(m0_output.mel_spectrogram.squeeze(0))
                
                # 准备批次数据
                batch_data = {
                    'mel_inputs': torch.stack(batch_m0),
                    'emotion_features': np.stack([item['emotion_features'] for item in batch]),
                    'target_mels': torch.stack([
                        torch.tensor(item['target_mel'], dtype=torch.float32) 
                        for item in batch
                    ])
                }
                
                # 训练B1
                losses = self.stage_b_model.train_step_b1(batch_data)
                epoch_losses.append(losses['b1_loss'])
            
            avg_loss = np.mean(epoch_losses)
            self.logger.info(f"B1 Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
            
            # 保存检查点
            if (epoch + 1) % 10 == 0:
                checkpoint_path = os.path.join(self.output_dir, f'stage_b1_epoch_{epoch+1}.pth')
                self.stage_b_model.save_checkpoint(checkpoint_path)
    
    def _train_b2(self, processed_data: List[Dict[str, Any]], epochs: int, batch_size: int) -> None:
        """训练B2阶段"""
        self.logger.info("训练B2阶段 (M1 + 情感 -> M2)")
        
        for epoch in range(epochs):
            epoch_losses = []
            
            for i in range(0, len(processed_data), batch_size):
                batch = processed_data[i:i+batch_size]
                
                # 使用阶段A和B1生成M1
                batch_m1 = []
                for item in batch:
                    # A阶段: 音素 -> M0
                    m0_output = self.stage_a_model.forward(item['phonemes'])
                    # B1阶段: M0 + 情感 -> M1
                    m1_output = self.stage_b_model.forward_b1(
                        m0_output.mel_spectrogram, item['emotion_features']
                    )
                    batch_m1.append(m1_output.mel_spectrogram.squeeze(0))
                
                # 准备批次数据
                batch_data = {
                    'mel_inputs': torch.stack(batch_m1),
                    'emotion_features': np.stack([item['emotion_features'] for item in batch]),
                    'target_mels': torch.stack([
                        torch.tensor(item['target_mel'], dtype=torch.float32) 
                        for item in batch
                    ])
                }
                
                # 训练B2
                losses = self.stage_b_model.train_step_b2(batch_data)
                epoch_losses.append(losses['b2_loss'])
            
            avg_loss = np.mean(epoch_losses)
            self.logger.info(f"B2 Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    def inference(self, audio_data: AudioData, use_quantizer: bool = False, 
                 codebook_size: Optional[int] = None, use_b2: bool = False) -> np.ndarray:
        """
        推理生成音频
        Args:
            audio_data: 输入音频数据
            use_quantizer: 是否使用VQ-VAE量化器
            codebook_size: VQ-VAE码本大小
            use_b2: 是否使用B2阶段
        Returns:
            重建的音频波形
        """
        self.logger.info("开始推理...")
        
        # 数据预处理
        processed = self.data_processor.process_audio(audio_data)
        
        # 阶段A: 音素 -> M0
        m0_output = self.stage_a_model.forward(processed.phonemes)
        
        # 情感特征处理
        emotion_features = processed.emotion_features
        
        # 可选的VQ-VAE量化
        if use_quantizer and self.emotion_quantizer is not None:
            if codebook_size is None:
                codebook_size = 256  # 默认码本大小
            
            quantized_emotion, vq_loss = self.emotion_quantizer.quantize(
                emotion_features, codebook_size
            )
            emotion_features = quantized_emotion
            self.logger.info(f"使用VQ-VAE量化，码本大小: {codebook_size}, VQ损失: {vq_loss:.4f}")
        
        # 阶段B: M0 + 情感 -> M1/M2
        mb_output = self.stage_b_model.forward(
            m0_output.mel_spectrogram, emotion_features, use_b2=use_b2
        )
        
        stage_info = "B2" if use_b2 else "B1"
        self.logger.info(f"使用{stage_info}阶段生成Mel频谱")
        
        # 声码器: Mel -> 音频
        synthesized_audio = self.vocoder.synthesize(mb_output.mel_spectrogram)
        
        self.logger.info("推理完成")
        return synthesized_audio
    
    def evaluate(self, test_data: List[AudioData]) -> Dict[str, float]:
        """评估模型性能"""
        self.logger.info("开始模型评估...")
        
        if len(test_data) == 0:
            self.logger.warning("没有测试数据")
            return {}
        
        metrics = {
            'mel_loss': [],
            'emotion_similarity': [],
            'content_preservation': []
        }
        
        for audio_data in tqdm(test_data, desc="评估进度"):
            try:
                # 推理
                reconstructed_audio = self.inference(audio_data)
                
                # 计算指标
                original_mel = self._extract_mel_spectrogram(audio_data)
                reconstructed_mel = self._extract_mel_spectrogram(
                    AudioData(reconstructed_audio, audio_data.sample_rate, 
                             len(reconstructed_audio) / audio_data.sample_rate)
                )
                
                # Mel损失
                mel_loss = np.mean((original_mel - reconstructed_mel) ** 2)
                metrics['mel_loss'].append(mel_loss)
                
            except Exception as e:
                self.logger.warning(f"评估样本失败: {e}")
                continue
        
        # 计算平均指标
        avg_metrics = {}
        for key, values in metrics.items():
            if values:
                avg_metrics[key] = np.mean(values)
            else:
                avg_metrics[key] = 0.0
        
        self.logger.info(f"评估结果: {avg_metrics}")
        return avg_metrics
    
    def run_vq_vae_experiment(self, test_audio: AudioData) -> Dict[str, Dict[str, Any]]:
        """
        运行VQ-VAE码本大小实验
        测试不同码本大小对情感细节保留的影响
        """
        if self.emotion_quantizer is None:
            self.logger.error("情感量化器未启用")
            return {}
        
        self.logger.info("开始VQ-VAE码本大小实验...")
        
        # 获取可用的码本大小
        codebook_sizes = self.emotion_quantizer.get_available_codebook_sizes()
        
        results = {}
        
        # 基线：不使用量化
        baseline_audio = self.inference(test_audio, use_quantizer=False, use_b2=False)
        baseline_b2_audio = self.inference(test_audio, use_quantizer=False, use_b2=True)
        
        results['baseline'] = {
            'audio_b1': baseline_audio,
            'audio_b2': baseline_b2_audio,
            'codebook_size': 'None',
            'vq_loss': 0.0
        }
        
        # 保存基线音频
        self._save_audio(baseline_audio, 
                        os.path.join(self.output_dir, 'baseline_b1.wav'), 
                        test_audio.sample_rate)
        self._save_audio(baseline_b2_audio, 
                        os.path.join(self.output_dir, 'baseline_b2.wav'), 
                        test_audio.sample_rate)
        
        # 测试不同码本大小
        for codebook_size in codebook_sizes:
            self.logger.info(f"测试码本大小: {codebook_size}")
            
            # B1阶段
            audio_b1 = self.inference(
                test_audio, 
                use_quantizer=True, 
                codebook_size=codebook_size, 
                use_b2=False
            )
            
            # B2阶段
            audio_b2 = self.inference(
                test_audio, 
                use_quantizer=True, 
                codebook_size=codebook_size, 
                use_b2=True
            )
            
            # 计算量化损失
            processed = self.data_processor.process_audio(test_audio)
            _, vq_loss = self.emotion_quantizer.quantize(
                processed.emotion_features, codebook_size
            )
            
            results[f'k{codebook_size}'] = {
                'audio_b1': audio_b1,
                'audio_b2': audio_b2,
                'codebook_size': codebook_size,
                'vq_loss': vq_loss
            }
            
            # 保存音频样本
            self._save_audio(audio_b1, 
                            os.path.join(self.output_dir, f'vq_k{codebook_size}_b1.wav'), 
                            test_audio.sample_rate)
            self._save_audio(audio_b2, 
                            os.path.join(self.output_dir, f'vq_k{codebook_size}_b2.wav'), 
                            test_audio.sample_rate)
        
        # 生成实验报告
        self._generate_experiment_report(results)
        
        self.logger.info(f"VQ-VAE实验完成，结果保存在 {self.output_dir}")
        return results
    
    def _generate_experiment_report(self, results: Dict[str, Dict[str, Any]]) -> None:
        """生成实验报告"""
        report_path = os.path.join(self.output_dir, 'vq_vae_experiment_report.txt')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("VQ-VAE码本大小实验报告\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("实验假设:\n")
            f.write("- 大码本 (K=1024): 应保留细腻情感细节\n")
            f.write("- 中等码本 (K=256, K=512): 保留主要情感，丢失部分细节\n")
            f.write("- 小码本 (K=64, K=128): 只保留基础情感特征\n\n")
            
            f.write("实验结果:\n")
            for key, result in results.items():
                f.write(f"\n{key}:\n")
                f.write(f"  码本大小: {result['codebook_size']}\n")
                f.write(f"  VQ损失: {result['vq_loss']:.4f}\n")
                f.write(f"  生成文件: {key}_b1.wav, {key}_b2.wav\n")
            
            f.write("\n音频文件说明:\n")
            f.write("- baseline_b1.wav: 不使用量化的B1阶段结果\n")
            f.write("- baseline_b2.wav: 不使用量化的B2阶段结果\n")
            f.write("- vq_k{size}_b1.wav: 使用码本大小{size}的B1阶段结果\n")
            f.write("- vq_k{size}_b2.wav: 使用码本大小{size}的B2阶段结果\n")
    
    def _save_audio(self, audio: np.ndarray, output_path: str, sample_rate: int):
        """保存音频文件"""
        import soundfile as sf
        sf.write(output_path, audio, sample_rate)


def main():
    """主函数，用于测试流水线"""
    # 创建流水线
    pipeline = EmotionAudioPipeline('config/base_config.yaml')
    
    # 如果有测试数据，运行推理
    test_data = pipeline.dataset_loader.load_test_data()
    if test_data:
        # 选择一个测试样本
        test_sample = test_data[0]
        
        # 运行推理
        reconstructed_audio = pipeline.inference(test_sample)
        
        # 保存结果
        output_path = os.path.join(pipeline.output_dir, 'inference_result.wav')
        pipeline._save_audio(reconstructed_audio, output_path, test_sample.sample_rate)
        
        # 运行VQ-VAE实验
        if pipeline.emotion_quantizer is not None:
            pipeline.run_vq_vae_experiment(test_sample)


if __name__ == "__main__":
    main()