"""
阶段A模型实现: 内容音素 -> M0 (中性Mel频谱)
支持FastSpeech2和其他TTS模型的接口
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any, Optional
import numpy as np
import os

from ..core.interfaces import StageAModel, ModelOutput


class FastSpeech2StageA(StageAModel):
    """
    基于FastSpeech2的阶段A模型实现
    输入: 音素序列
    输出: 中性的Mel频谱 M0
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_config = config.get('model_config', {})
        
        # 模型参数
        self.d_model = self.model_config.get('d_model', 256)
        self.n_layers = self.model_config.get('n_layers', 6)
        self.n_heads = self.model_config.get('n_heads', 8)
        self.dropout = self.model_config.get('dropout', 0.1)
        
        # 构建模型
        self.model = self._build_model()
        self.optimizer = None
        self.criterion = nn.MSELoss()
        
        # 音素到索引的映射
        self.phoneme_to_idx = self._build_phoneme_vocab()
        
    def _build_phoneme_vocab(self) -> Dict[str, int]:
        """构建音素词汇表"""
        # 这里是一个简化的中文音素词汇表，实际应用中需要更完整的音素集
        common_phonemes = [
            '<PAD>', '<UNK>', '<SOS>', '<EOS>',
            # 声母
            'b', 'p', 'm', 'f', 'd', 't', 'n', 'l',
            'g', 'k', 'h', 'j', 'q', 'x', 'zh', 'ch', 'sh', 'r', 'z', 'c', 's', 'y', 'w',
            # 韵母
            'a', 'o', 'e', 'i', 'u', 'v', 'ai', 'ei', 'ui', 'ao', 'ou', 'iu',
            'ie', 've', 'an', 'en', 'in', 'un', 'vn', 'ang', 'eng', 'ing', 'ong',
            # 声调
            '1', '2', '3', '4', '5'
        ]
        return {phoneme: idx for idx, phoneme in enumerate(common_phonemes)}
    
    def _build_model(self) -> nn.Module:
        """构建FastSpeech2模型"""
        return SimpleFastSpeech2(
            vocab_size=len(self.phoneme_to_idx),
            d_model=self.d_model,
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            dropout=self.dropout,
            mel_dim=80  # 标准Mel频谱维度
        )
    
    def _phonemes_to_indices(self, phonemes: List[str]) -> torch.Tensor:
        """将音素序列转换为索引"""
        indices = []
        for phoneme in phonemes:
            if phoneme in self.phoneme_to_idx:
                indices.append(self.phoneme_to_idx[phoneme])
            else:
                indices.append(self.phoneme_to_idx['<UNK>'])
        
        return torch.tensor(indices, dtype=torch.long)
    
    def forward(self, phonemes: List[str]) -> ModelOutput:
        """
        前向传播
        输入: 内容音素
        输出: M0 (中性Mel频谱)
        """
        # 转换音素为索引
        phoneme_indices = self._phonemes_to_indices(phonemes).unsqueeze(0)  # (1, seq_len)
        
        # 前向传播
        self.model.eval()
        with torch.no_grad():
            mel_output, attention_weights = self.model(phoneme_indices)
        
        return ModelOutput(
            mel_spectrogram=mel_output,
            attention_weights=attention_weights,
            metadata={'phonemes': phonemes}
        )
    
    def train_step(self, batch_data: Dict[str, Any]) -> Dict[str, float]:
        """训练步骤"""
        self.model.train()
        
        # 解析批次数据
        phoneme_sequences = batch_data['phonemes']  # List[List[str]]
        target_mels = batch_data['mel_spectrograms']  # torch.Tensor (batch, mel_dim, time)
        
        # 转换音素为索引
        batch_phoneme_indices = []
        for phonemes in phoneme_sequences:
            indices = self._phonemes_to_indices(phonemes)
            batch_phoneme_indices.append(indices)
        
        # 填充序列到相同长度
        batch_phoneme_indices = torch.nn.utils.rnn.pad_sequence(
            batch_phoneme_indices, batch_first=True, padding_value=self.phoneme_to_idx['<PAD>']
        )
        
        # 前向传播
        mel_outputs, attention_weights = self.model(batch_phoneme_indices)
        
        # 计算损失
        # 需要调整维度匹配
        if target_mels.dim() == 3:  # (batch, mel_dim, time)
            target_mels = target_mels.transpose(1, 2)  # (batch, time, mel_dim)
        
        loss = self.criterion(mel_outputs, target_mels)
        
        # 反向传播
        if self.optimizer is None:
            self.optimizer = torch.optim.Adam(self.model.parameters(), 
                                            lr=self.config.get('training', {}).get('learning_rate', 1e-4))
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return {
            'loss': loss.item(),
            'mel_loss': loss.item()
        }
    
    def save_checkpoint(self, path: str) -> None:
        """保存检查点"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'phoneme_to_idx': self.phoneme_to_idx
        }
        if self.optimizer:
            checkpoint['optimizer_state_dict'] = self.optimizer.state_dict()
        
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str) -> None:
        """加载检查点"""
        checkpoint = torch.load(path, map_location='cpu')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if 'optimizer_state_dict' in checkpoint and self.optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'phoneme_to_idx' in checkpoint:
            self.phoneme_to_idx = checkpoint['phoneme_to_idx']


class SimpleFastSpeech2(nn.Module):
    """
    简化的FastSpeech2模型实现
    这是一个基础版本，实际应用中可能需要更复杂的实现
    """
    
    def __init__(self, vocab_size: int, d_model: int, n_layers: int, n_heads: int, dropout: float, mel_dim: int):
        super().__init__()
        
        self.d_model = d_model
        self.mel_dim = mel_dim
        
        # 音素嵌入
        self.phoneme_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # 时长预测器（简化版）
        self.duration_predictor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
            nn.ReLU()
        )
        
        # Mel频谱预测器
        self.mel_predictor = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, mel_dim)
        )
    
    def forward(self, phoneme_indices: torch.Tensor) -> tuple:
        """
        前向传播
        Args:
            phoneme_indices: (batch_size, seq_len)
        Returns:
            mel_outputs: (batch_size, expanded_seq_len, mel_dim)
            attention_weights: None (FastSpeech2不使用注意力)
        """
        # 音素嵌入
        embedded = self.phoneme_embedding(phoneme_indices) * np.sqrt(self.d_model)
        embedded = self.positional_encoding(embedded)
        
        # 编码
        encoded = self.encoder(embedded)
        
        # 预测时长
        durations = self.duration_predictor(encoded).squeeze(-1)  # (batch, seq_len)
        durations = torch.clamp(durations, min=0.1)  # 确保时长为正
        
        # 长度调节器（简化版）
        expanded_encoded = self._length_regulator(encoded, durations)
        
        # 预测Mel频谱
        mel_outputs = self.mel_predictor(expanded_encoded)
        
        return mel_outputs, None
    
    def _length_regulator(self, encoded: torch.Tensor, durations: torch.Tensor) -> torch.Tensor:
        """
        长度调节器（简化实现）
        """
        batch_size, seq_len, d_model = encoded.shape
        
        # 简化处理：假设每个音素对应固定长度
        target_len = int(durations.mean().item() * seq_len)
        target_len = max(target_len, seq_len)  # 至少保持原长度
        
        # 使用插值扩展序列
        expanded = F.interpolate(
            encoded.transpose(1, 2),  # (batch, d_model, seq_len)
            size=target_len,
            mode='linear',
            align_corners=False
        ).transpose(1, 2)  # (batch, target_len, d_model)
        
        return expanded


class PositionalEncoding(nn.Module):
    """位置编码"""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(1), :].transpose(0, 1)
        return self.dropout(x)
