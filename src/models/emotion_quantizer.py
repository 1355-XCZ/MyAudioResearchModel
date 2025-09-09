"""
情感量化器实现: 使用VQ-VAE量化emotion2vec表征
作为整个B阶段的前置处理，支持码本大小实验
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple, List
import numpy as np
import os

from ..core.interfaces import EmotionQuantizer


class VQVAEEmotionQuantizer(EmotionQuantizer):
    """
    VQ-VAE情感量化器
    在B阶段之前对emotion2vec表征进行量化
    支持不同码本大小的实验
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # 基础参数
        self.input_dim = config.get('input_dim', 768)  # emotion2vec维度
        self.embedding_dim = config.get('embedding_dim', 64)
        self.commitment_loss_weight = config.get('commitment_loss_weight', 0.25)
        
        # 支持的码本大小
        self.codebook_sizes = config.get('codebook_sizes', [64, 128, 256, 512, 1024])
        
        # 特征映射器（将emotion2vec映射到embedding空间）
        self.feature_encoder = nn.Sequential(
            nn.Linear(self.input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, self.embedding_dim)
        )
        
        # 特征解码器（将量化特征映射回原始维度）
        self.feature_decoder = nn.Sequential(
            nn.Linear(self.embedding_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, self.input_dim)
        )
        
        # 为不同码本大小创建VQ模块
        self.vq_modules = nn.ModuleDict()
        for codebook_size in self.codebook_sizes:
            self.vq_modules[str(codebook_size)] = VectorQuantizer(
                num_embeddings=codebook_size,
                embedding_dim=self.embedding_dim,
                commitment_cost=self.commitment_loss_weight
            )
    
    def quantize(self, emotion_features: np.ndarray, codebook_size: int) -> Tuple[np.ndarray, float]:
        """
        量化情感特征
        输入: emotion2vec表征, 码本大小
        输出: (量化后的特征, 重建损失)
        """
        if codebook_size not in self.codebook_sizes:
            raise ValueError(f"码本大小 {codebook_size} 不支持。支持的大小: {self.codebook_sizes}")
        
        # 转换为tensor
        if isinstance(emotion_features, np.ndarray):
            emotion_tensor = torch.tensor(emotion_features, dtype=torch.float32)
            if emotion_tensor.dim() == 1:
                emotion_tensor = emotion_tensor.unsqueeze(0)  # (1, input_dim)
        else:
            emotion_tensor = emotion_features
        
        # 编码到embedding空间
        encoded = self.feature_encoder(emotion_tensor)  # (batch, embedding_dim)
        
        # 量化
        vq_module = self.vq_modules[str(codebook_size)]
        # VQ需要(batch, embedding_dim, 1, 1)格式
        encoded_4d = encoded.unsqueeze(-1).unsqueeze(-1)
        quantized_4d, vq_loss, _ = vq_module(encoded_4d)
        quantized = quantized_4d.squeeze(-1).squeeze(-1)  # (batch, embedding_dim)
        
        # 解码回原始维度
        decoded = self.feature_decoder(quantized)  # (batch, input_dim)
        
        # 计算重建损失
        reconstruction_loss = F.mse_loss(decoded, emotion_tensor)
        total_loss = vq_loss + reconstruction_loss
        
        return decoded.detach().cpu().numpy(), total_loss.item()
    
    def experiment_codebook_sizes(self, emotion_features: np.ndarray) -> Dict[int, Tuple[np.ndarray, float]]:
        """
        实验不同码本大小的效果
        输入: emotion2vec表征
        输出: {码本大小: (量化特征, 损失)}
        """
        results = {}
        
        for codebook_size in self.codebook_sizes:
            quantized_features, loss = self.quantize(emotion_features, codebook_size)
            results[codebook_size] = (quantized_features, loss)
        
        return results
    
    def get_available_codebook_sizes(self) -> List[int]:
        """获取可用的码本大小列表"""
        return self.codebook_sizes
    
    def forward_pass_through(self, emotion_features: np.ndarray) -> np.ndarray:
        """
        直通模式：不进行量化，只是通过编码器-解码器
        用于验证当不使用量化时，输出应该接近输入
        """
        if isinstance(emotion_features, np.ndarray):
            emotion_tensor = torch.tensor(emotion_features, dtype=torch.float32)
            if emotion_tensor.dim() == 1:
                emotion_tensor = emotion_tensor.unsqueeze(0)
        else:
            emotion_tensor = emotion_features
        
        # 编码-解码（不量化）
        with torch.no_grad():
            encoded = self.feature_encoder(emotion_tensor)
            decoded = self.feature_decoder(encoded)
        
        return decoded.cpu().numpy()
    
    def save_checkpoint(self, path: str) -> None:
        """保存检查点"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        checkpoint = {
            'feature_encoder_state_dict': self.feature_encoder.state_dict(),
            'feature_decoder_state_dict': self.feature_decoder.state_dict(),
            'vq_modules_state_dict': self.vq_modules.state_dict(),
            'config': self.config
        }
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str) -> None:
        """加载检查点"""
        checkpoint = torch.load(path, map_location='cpu')
        self.feature_encoder.load_state_dict(checkpoint['feature_encoder_state_dict'])
        self.feature_decoder.load_state_dict(checkpoint['feature_decoder_state_dict'])
        self.vq_modules.load_state_dict(checkpoint['vq_modules_state_dict'])


class VectorQuantizer(nn.Module):
    """
    向量量化器 (VQ-VAE)
    """
    
    def __init__(self, num_embeddings: int, embedding_dim: int, commitment_cost: float = 0.25):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        
        # 码本
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)
    
    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播
        Args:
            inputs: (batch, embedding_dim, H, W) 或 (batch, embedding_dim, 1, 1)
        Returns:
            quantized: 量化后的特征
            vq_loss: VQ损失
            perplexity: 困惑度
        """
        # 获取输入形状
        input_shape = inputs.shape
        
        # 展平输入: (batch * H * W, embedding_dim)
        flat_input = inputs.view(-1, self.embedding_dim)
        
        # 计算距离
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self.embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self.embedding.weight.t()))
        
        # 找到最近的嵌入
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # 量化
        quantized = torch.matmul(encodings, self.embedding.weight).view(input_shape)
        
        # 损失计算
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        vq_loss = q_latent_loss + self.commitment_cost * e_latent_loss
        
        # Straight-through estimator
        quantized = inputs + (quantized - inputs).detach()
        
        # 困惑度计算
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        return quantized, vq_loss, perplexity


class IdentityQuantizer(EmotionQuantizer):
    """
    恒等量化器：不进行任何量化，直接返回输入
    用于对比实验中的基线
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.codebook_sizes = config.get('codebook_sizes', [64, 128, 256, 512, 1024])
    
    def quantize(self, emotion_features: np.ndarray, codebook_size: int) -> Tuple[np.ndarray, float]:
        """
        恒等量化：直接返回输入特征
        """
        return emotion_features.copy(), 0.0
    
    def experiment_codebook_sizes(self, emotion_features: np.ndarray) -> Dict[int, Tuple[np.ndarray, float]]:
        """
        恒等实验：所有码本大小都返回相同的输入特征
        """
        results = {}
        for codebook_size in self.codebook_sizes:
            results[codebook_size] = (emotion_features.copy(), 0.0)
        return results
    
    def get_available_codebook_sizes(self) -> List[int]:
        """获取可用的码本大小列表"""
        return self.codebook_sizes


def create_emotion_quantizer(quantizer_type: str, config: Dict[str, Any]) -> EmotionQuantizer:
    """
    工厂函数：创建情感量化器
    Args:
        quantizer_type: 量化器类型 ('vqvae', 'identity')
        config: 配置字典
    Returns:
        情感量化器实例
    """
    if quantizer_type.lower() == 'vqvae':
        return VQVAEEmotionQuantizer(config)
    elif quantizer_type.lower() == 'identity':
        return IdentityQuantizer(config)
    else:
        raise ValueError(f"不支持的量化器类型: {quantizer_type}")
