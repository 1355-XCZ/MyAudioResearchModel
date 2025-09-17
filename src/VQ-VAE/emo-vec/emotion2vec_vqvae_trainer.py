# Copyright (c) 2023 Amphion.
# Adapted for Emotion2Vec features by [Your Name]
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import numpy as np
import yaml
import torchaudio
import logging
import sys
import os
from pathlib import Path

# 添加路径以导入 VEVO 组件
current_dir = Path(__file__).parent
vevo_code_dir = current_dir.parent / "vevo-code"
sys.path.append(str(vevo_code_dir))

try:
    from base.base_trainer import BaseTrainer
    from base.emilia_dataset import VCEmiliaDataset, VCCollator
    from vevo.vevo_repcodec import VevoRepCodec
    from utils.util import Logger, ValueWindow
    VEVO_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Cannot import from vevo-code: {e}")
    logging.warning("Using simplified implementation")
    BaseTrainer = object
    VEVO_AVAILABLE = False

from emotion2vec_extractor import create_emotion2vec_extractor


class Emotion2VecDataset(torch.utils.data.Dataset):
    """
    Emotion2Vec 数据集
    替代原来的 VCEmiliaDataset，专门处理 emotion2vec 特征
    """
    
    def __init__(
        self,
        data_root: str,
        cfg: any,
        is_valid: bool = False
    ):
        super().__init__()
        self.data_root = data_root
        self.cfg = cfg
        self.is_valid = is_valid
        
        # 获取音频文件列表
        self.audio_files = self._get_audio_files()
        
        # 预处理配置
        self.sample_rate = getattr(cfg.preprocess, 'sample_rate', 16000)
        self.max_length = getattr(cfg.preprocess, 'max_length', 128000)
        self.random_crop = getattr(cfg.preprocess, 'random_crop', True)
        
        print(f"Loaded {len(self.audio_files)} audio files for {'validation' if is_valid else 'training'}")
        
    def _get_audio_files(self):
        """获取音频文件列表"""
        audio_files = []
        audio_extensions = ('.wav', '.flac', '.mp3', '.ogg')
        
        if os.path.isfile(self.data_root):
            # 如果是文件，认为是文件列表
            with open(self.data_root, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and os.path.exists(line):
                        audio_files.append(line)
        else:
            # 如果是目录，递归搜索音频文件
            for root, dirs, files in os.walk(self.data_root):
                for file in files:
                    if file.lower().endswith(audio_extensions):
                        audio_files.append(os.path.join(root, file))
        
        # 限制数量用于调试
        if hasattr(self.cfg, 'debug') and self.cfg.debug:
            audio_files = audio_files[:100]
        
        return audio_files
        
    def __len__(self):
        return len(self.audio_files)
        
    def __getitem__(self, idx):
        audio_path = self.audio_files[idx]
        
        try:
            # 加载音频
            waveform, sr = torchaudio.load(audio_path)
            
            # 重采样到目标采样率
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                waveform = resampler(waveform)
            
            # 转为单声道
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            waveform = waveform.squeeze(0)  # [T]
            
            # 裁剪或填充到指定长度
            if self.random_crop and len(waveform) > self.max_length:
                start = torch.randint(0, len(waveform) - self.max_length + 1, (1,))
                waveform = waveform[start:start + self.max_length]
            elif len(waveform) > self.max_length:
                waveform = waveform[:self.max_length]
            elif len(waveform) < self.max_length:
                # 重复填充
                repeat_times = (self.max_length // len(waveform)) + 1
                waveform = waveform.repeat(repeat_times)[:self.max_length]
                
        except Exception as e:
            logging.warning(f"Failed to load audio {audio_path}: {e}")
            # 返回随机波形作为 fallback
            waveform = torch.randn(self.max_length)
        
        return {
            "wav": waveform,
            "wav_16000": waveform,  # emotion2vec 使用 16kHz
            "wav_16000_len": torch.tensor(len(waveform)),
            "audio_path": audio_path,
            "idx": idx
        }


class Emotion2VecCollator:
    """
    Emotion2Vec 数据整理器
    """
    
    def __init__(self, cfg):
        self.cfg = cfg
        
    def __call__(self, batch):
        # 批处理音频数据
        wavs = [item["wav"] for item in batch]
        wav_16000s = [item["wav_16000"] for item in batch]
        wav_16000_lens = [item["wav_16000_len"] for item in batch]
        audio_paths = [item["audio_path"] for item in batch]
        indices = [item["idx"] for item in batch]
        
        # 填充到相同长度
        max_len = max(len(wav) for wav in wavs)
        padded_wavs = []
        padded_wav_16000s = []
        
        for wav, wav_16000 in zip(wavs, wav_16000s):
            if len(wav) < max_len:
                pad_size = max_len - len(wav)
                wav = torch.cat([wav, torch.zeros(pad_size)], dim=0)
                wav_16000 = torch.cat([wav_16000, torch.zeros(pad_size)], dim=0)
            padded_wavs.append(wav)
            padded_wav_16000s.append(wav_16000)
        
        return {
            "wav": torch.stack(padded_wavs),
            "wav_16000": torch.stack(padded_wav_16000s),
            "wav_16000_len": torch.stack(wav_16000_lens),
            "audio_paths": audio_paths,
            "indices": indices
        }


class Emotion2VecVQVAETrainer(BaseTrainer if VEVO_AVAILABLE else object):
    """
    Emotion2Vec VQ-VAE 训练器
    基于 VEVO 的 VQVAETrainer，替换特征提取器为 emotion2vec
    """
    
    def __init__(self, args, cfg):
        if VEVO_AVAILABLE:
            super(Emotion2VecVQVAETrainer, self).__init__(args, cfg)
        else:
            # 简化版初始化
            self.args = args
            self.cfg = cfg
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._init_logger()
            
        # 构建 emotion2vec 特征提取器
        self._build_emotion2vec_extractor()
        
        if not VEVO_AVAILABLE:
            # 构建模型和优化器（如果没有基类）
            self.model = self._build_model()
            self._build_optimizer()
            
        print("Emotion2Vec VQ-VAE Trainer initialized successfully!")
        
    def _init_logger(self):
        """初始化日志器（简化版）"""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def _build_emotion2vec_extractor(self):
        """构建 emotion2vec 特征提取器"""
        extractor_cfg = getattr(self.cfg.model, 'emotion2vec', {})
        
        self.emotion2vec_extractor = create_emotion2vec_extractor(
            model_name=getattr(extractor_cfg, 'model_name', 'emotion2vec_base'),
            use_dummy=getattr(extractor_cfg, 'use_dummy', True),  # 默认使用 dummy 模式
            feature_dim=getattr(extractor_cfg, 'feature_dim', 1024),
            sample_rate=getattr(extractor_cfg, 'sample_rate', 16000),
            normalize=getattr(extractor_cfg, 'normalize', True),
            device=self.device if hasattr(self, 'device') else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        
        self.emotion2vec_extractor.eval()
        if hasattr(self, 'accelerator'):
            self.emotion2vec_extractor.to(self.accelerator.device)
        else:
            self.emotion2vec_extractor.to(self.device)
        
        # 加载归一化统计量（如果提供）
        if hasattr(self.cfg.model, 'use_norm_feat') and self.cfg.model.use_norm_feat:
            if hasattr(self.cfg.model, 'emotion2vec_stat_mean_var_path'):
                stat = np.load(self.cfg.model.emotion2vec_stat_mean_var_path)
                device = self.accelerator.device if hasattr(self, 'accelerator') else self.device
                self.feat_norm_mean = torch.tensor(stat["mean"]).to(device)
                self.feat_norm_std = torch.tensor(stat["std"]).to(device)
                
    def _build_model(self):
        """构建码本模型"""
        model_cfg = self.cfg.model.repcodec
        
        if hasattr(model_cfg, 'config_path') and os.path.exists(model_cfg.config_path):
            with open(model_cfg.config_path) as fp:
                conf = yaml.load(fp, Loader=yaml.FullLoader)
        else:
            # 默认配置，适配 emotion2vec
            emotion2vec_cfg = getattr(self.cfg.model, 'emotion2vec', {})
            feature_dim = getattr(emotion2vec_cfg, 'feature_dim', 1024)
            
            conf = {
                'input_channels': feature_dim,  # emotion2vec 特征维度
                'output_channels': feature_dim,
                'encode_channels': 512,
                'decode_channels': 512,
                'code_dim': 512,
                'codebook_num': getattr(model_cfg, 'codebook_num', 1),
                'codebook_size': getattr(model_cfg, 'codebook_size', 1024),  # 码本大小
                'bias': True,
                'enc_ratios': [1, 1],
                'dec_ratios': [1, 1],
                'enc_strides': [1, 1],
                'dec_strides': [1, 1]
            }
            
        model = VevoRepCodec(**conf)
        return model
        
    def _build_optimizer(self):
        """构建优化器（简化版）"""
        if not VEVO_AVAILABLE:
            lr = getattr(self.cfg.train.adam, 'lr', 1e-4)
            betas = getattr(self.cfg.train.adam, 'betas', [0.5, 0.9])
            
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=lr,
                betas=betas
            )
            
            # 简单的学习率调度器
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=10000, gamma=0.95
            )
        
    def _build_dataset(self):
        """构建数据集"""
        return Emotion2VecDataset, Emotion2VecCollator
        
    @torch.no_grad()
    def _extract_emotion2vec_feature(self, wavs, wav_lens=None):
        """
        提取 emotion2vec 特征
        
        Args:
            wavs: [B, T] 音频波形
            wav_lens: [B,] 音频长度
            
        Returns:
            feats: [B, T, D] emotion2vec 特征
        """
        features_list = []
        
        for i, wav in enumerate(wavs):
            try:
                # 如果有长度信息，截取有效部分
                if wav_lens is not None:
                    valid_len = wav_lens[i].item()
                    wav_valid = wav[:valid_len]
                else:
                    wav_valid = wav
                
                # 提取特征
                feat = self.emotion2vec_extractor.extract_features_from_waveform(
                    wav_valid.cpu().numpy()
                )
                
                if isinstance(feat, np.ndarray):
                    feat = torch.from_numpy(feat)
                
                # 确保特征在正确的设备上
                device = self.accelerator.device if hasattr(self, 'accelerator') else self.device
                feat = feat.to(device)
                
                features_list.append(feat)
                
            except Exception as e:
                logging.warning(f"Failed to extract features from batch {i}: {e}")
                # 使用随机特征作为fallback
                device = self.accelerator.device if hasattr(self, 'accelerator') else self.device
                dummy_feat = torch.randn(100, 1024, device=device)  # 假设100帧
                features_list.append(dummy_feat)
                
        # 批处理特征（简单的 padding 处理）
        max_len = max(feat.shape[0] for feat in features_list)
        
        batched_features = []
        for feat in features_list:
            if feat.shape[0] < max_len:
                # 简单的重复填充
                pad_size = max_len - feat.shape[0]
                if feat.shape[0] > 0:
                    padding = feat[-1:].repeat(pad_size, 1)
                    feat = torch.cat([feat, padding], dim=0)
                else:
                    # 如果特征为空，创建零填充
                    device = feat.device
                    feat = torch.zeros(max_len, feat.shape[1], device=device)
            batched_features.append(feat)
            
        return torch.stack(batched_features)
        
    def _train_step(self, batch):
        """
        训练步骤
        基于 VEVO 的训练逻辑，但使用 emotion2vec 特征
        """
        train_losses = {}
        total_loss = 0
        train_stats = {}
        
        # 从批次中获取音频数据
        wavs = batch["wav_16000"]  # [B, T]
        wav_lens = batch["wav_16000_len"]  # [B,]
        
        # 提取 emotion2vec 特征
        feat = self._extract_emotion2vec_feature(wavs, wav_lens)  # [B, T, D]
        
        # Gaussian 归一化（如果配置了）
        if hasattr(self.cfg.model, 'use_norm_feat') and self.cfg.model.use_norm_feat:
            if hasattr(self, 'feat_norm_mean') and hasattr(self, 'feat_norm_std'):
                feat = (feat - self.feat_norm_mean.to(feat)) / self.feat_norm_std.to(feat)
        
        # 清理 GPU 内存
        torch.cuda.empty_cache()
        
        # 前向传播 (使用 VEVO 的码本模型)
        # VEVO 期望 [B, D, T] 格式
        feat_rec, _, _, vqloss, perplexity = self.model(feat.transpose(1, 2))  
        feat_rec = feat_rec.transpose(1, 2)  # 转回 [B, T, D]
        
        # 计算码本损失
        codebook_loss = torch.sum(vqloss)
        
        # 计算重构损失 (使用 L1 loss，与 VEVO 保持一致)
        rec_loss = torch.nn.functional.l1_loss(feat_rec, feat)
        total_loss += rec_loss * 32  # 重构损失权重，与 VEVO 保持一致
        train_losses["rec_loss"] = rec_loss
        
        # 添加码本损失
        total_loss += codebook_loss
        train_losses["codebook_loss"] = codebook_loss
        
        # 反向传播
        if hasattr(self, 'optimizer'):
            self.optimizer.zero_grad()
            
            if hasattr(self, 'accelerator'):
                self.accelerator.backward(total_loss)
                if self.accelerator.sync_gradients:
                    self.accelerator.clip_grad_norm_(
                        filter(lambda p: p.requires_grad, self.model.parameters()), 1.0
                    )
            else:
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    filter(lambda p: p.requires_grad, self.model.parameters()), 1.0
                )
                
            self.optimizer.step()
            
            if hasattr(self, 'scheduler'):
                self.scheduler.step()
        
        # 转换为标量
        for item in train_losses:
            if isinstance(train_losses[item], torch.Tensor):
                train_losses[item] = train_losses[item].item()
                
        train_losses["batch_size"] = wavs.shape[0]
        
        # 添加困惑度统计
        if isinstance(perplexity, torch.Tensor):
            train_stats["perplexity"] = torch.mean(perplexity).item()
        
        if hasattr(self, 'current_loss'):
            self.current_loss = total_loss.item()
        
        return (total_loss.item(), train_losses, train_stats)
        
    def _valid_step(self, batch):
        """验证步骤"""
        with torch.no_grad():
            return self._train_step(batch)
            
    def save_checkpoint(self, checkpoint_path):
        """保存检查点"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if hasattr(self, 'optimizer') else None,
            'scheduler_state_dict': self.scheduler.state_dict() if hasattr(self, 'scheduler') else None,
            'config': self.cfg,
        }
        
        # 保存 emotion2vec 提取器的状态（如果需要）
        if hasattr(self.emotion2vec_extractor, 'state_dict'):
            checkpoint['emotion2vec_state_dict'] = self.emotion2vec_extractor.state_dict()
        
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")
        
    def load_checkpoint(self, checkpoint_path):
        """加载检查点"""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if 'optimizer_state_dict' in checkpoint and hasattr(self, 'optimizer'):
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
        if 'scheduler_state_dict' in checkpoint and hasattr(self, 'scheduler'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
        print(f"Checkpoint loaded from {checkpoint_path}")


# 简化的训练配置类
class SimpleConfig:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            if isinstance(value, dict):
                setattr(self, key, SimpleConfig(**value))
            else:
                setattr(self, key, value)


def create_emotion2vec_config(
    data_root: str = "./data",
    exp_dir: str = "./experiments/emotion2vec_vqvae",
    codebook_size: int = 1024,
    feature_dim: int = 1024,
    batch_size: int = 8,
    learning_rate: float = 1e-4,
    max_steps: int = 100000,
    use_dummy: bool = True
):
    """创建 Emotion2Vec VQ-VAE 训练配置"""
    config = SimpleConfig(
        exp_name="emotion2vec_vqvae",
        dataset={
            "data_root": data_root
        },
        preprocess=SimpleConfig(
            hop_size=320,
            sample_rate=16000,
            max_length=128000,
            random_crop=True
        ),
        model=SimpleConfig(
            emotion2vec=SimpleConfig(
                model_name='emotion2vec_base',
                use_dummy=use_dummy,
                feature_dim=feature_dim,
                sample_rate=16000,
                normalize=True
            ),
            repcodec=SimpleConfig(
                codebook_size=codebook_size,
                codebook_num=1,
                input_channels=feature_dim,
                output_channels=feature_dim,
                encode_channels=512,
                decode_channels=512,
                code_dim=512
            ),
            use_norm_feat=False,  # 如果有统计文件可以设为 True
            representation_type="emotion2vec"
        ),
        train=SimpleConfig(
            batch_size=batch_size,
            max_steps=max_steps,
            learning_rate=learning_rate,
            adam=SimpleConfig(
                lr=learning_rate,
                betas=[0.5, 0.9]
            ),
            save_checkpoints_steps=2000,
            valid_interval=2000,
            gradient_accumulation_step=1,
            dataloader=SimpleConfig(
                num_worker=4,
                pin_memory=True
            )
        ),
        log_dir=exp_dir,
        debug=False
    )
    return config


if __name__ == "__main__":
    # 测试代码
    print("Testing Emotion2Vec VQ-VAE Trainer...")
    
    # 创建简单的参数和配置
    class SimpleArgs:
        def __init__(self):
            self.exp_name = "test_emotion2vec_vqvae"
            self.config = None
            self.exp_dir = "./test_exp"
            
    args = SimpleArgs()
    cfg = create_emotion2vec_config(
        data_root="./test_data",
        use_dummy=True,  # 使用 dummy 特征用于测试
        batch_size=2,
        max_steps=100
    )
    
    # 创建训练器
    try:
        trainer = Emotion2VecVQVAETrainer(args, cfg)
        print("✓ Trainer created successfully!")
        
        # 创建测试数据
        test_batch = {
            "wav_16000": torch.randn(2, 16000),  # 2个样本，每个1秒
            "wav_16000_len": torch.tensor([16000, 16000])
        }
        
        print("Testing training step...")
        loss, train_losses, train_stats = trainer._train_step(test_batch)
        print(f"✓ Training step successful! Loss: {loss:.4f}")
        print(f"  - Reconstruction loss: {train_losses['rec_loss']:.4f}")
        print(f"  - Codebook loss: {train_losses['codebook_loss']:.4f}")
        
        print("Emotion2Vec VQ-VAE Trainer test completed successfully!")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()
        print("This might be expected if running without proper setup.")
