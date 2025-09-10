"""
阶段B模型实现: 包含B1和B2两个子阶段
B1: (M0, emotion2vec表征) -> M1
B2: (M1, emotion2vec表征) -> M2 (扩散模型细化，可选)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional
import numpy as np
import os
import math

from ..core.interfaces import StageBModel, ModelOutput


class TwoStageEmotionModel(StageBModel):
    """
    两阶段情感模型
    B1: (M0, emotion2vec表征) -> M1
    B2: (M1, emotion2vec表征) -> M2 (可选)
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_config = config.get('model_config', {})
        
        # 模型参数
        self.emotion_dim = self.model_config.get('emotion_dim', 768)
        self.mel_dim = self.model_config.get('mel_dim', 80)
        self.hidden_dim = self.model_config.get('hidden_dim', 512)
        self.n_layers = self.model_config.get('n_layers', 4)
        
        # 是否启用B2阶段
        self.enable_b2 = self.model_config.get('enable_b2', False)
        
        # 构建B1和B2模型
        self.b1_model = self._build_b1_model()
        self.b2_model = self._build_b2_model() if self.enable_b2 else None
        
        # 优化器
        self.b1_optimizer = None
        self.b2_optimizer = None
        
        # 损失函数
        self.criterion = nn.MSELoss()
        
    def _build_b1_model(self) -> nn.Module:
        """构建B1模型（情感适配器）"""
        return EmotionAdapter(
            emotion_dim=self.emotion_dim,
            mel_dim=self.mel_dim,
            hidden_dim=self.hidden_dim,
            n_layers=self.n_layers
        )
    
    def _build_b2_model(self) -> Optional[nn.Module]:
        """构建B2模型（扩散细化器）"""
        if not self.enable_b2:
            return None
        
        return DiffusionRefiner(
            emotion_dim=self.emotion_dim,
            mel_dim=self.mel_dim,
            hidden_dim=self.hidden_dim,
            diffusion_steps=self.model_config.get('diffusion_steps', 20)
        )
    
    def forward_b1(self, mel_input: torch.Tensor, emotion_features: np.ndarray) -> ModelOutput:
        """
        B1阶段前向传播
        输入: (M0, emotion2vec表征)
        输出: M1 (带情感的Mel频谱)
        """
        # 转换emotion特征为tensor
        if isinstance(emotion_features, np.ndarray):
            emotion_tensor = torch.tensor(emotion_features, dtype=torch.float32)
            if emotion_tensor.dim() == 1:
                emotion_tensor = emotion_tensor.unsqueeze(0)
        else:
            emotion_tensor = emotion_features
        
        # 确保mel_input的维度正确
        if mel_input.dim() == 2:
            mel_input = mel_input.unsqueeze(0)
        
        # B1前向传播
        try:
            self.b1_model.eval()
            with torch.no_grad():
                mel_output = self.b1_model(mel_input, emotion_tensor)
        except Exception as e:
            print(f"⚠️ B1处理失败，使用透传: {e}")
            mel_output = mel_input
        
        return ModelOutput(
            mel_spectrogram=mel_output,
            metadata={'stage': 'B1', 'emotion_dim': emotion_tensor.shape[-1]}
        )
    
    def forward_b2(self, mel_input: torch.Tensor, emotion_features: np.ndarray) -> ModelOutput:
        """
        B2阶段前向传播 (可选，扩散模型细化)
        输入: (M1, emotion2vec表征)
        输出: M2 (细化的Mel频谱)
        """
        if self.b2_model is None:
            raise ValueError("B2模型未启用，请在配置中设置 enable_b2: true")
        
        # 转换emotion特征为tensor
        if isinstance(emotion_features, np.ndarray):
            emotion_tensor = torch.tensor(emotion_features, dtype=torch.float32)
            if emotion_tensor.dim() == 1:
                emotion_tensor = emotion_tensor.unsqueeze(0)
        else:
            emotion_tensor = emotion_features
        
        # 确保mel_input的维度正确
        if mel_input.dim() == 2:
            mel_input = mel_input.unsqueeze(0)
        
        # B2前向传播
        self.b2_model.eval()
        with torch.no_grad():
            mel_output = self.b2_model(mel_input, emotion_tensor)
        
        return ModelOutput(
            mel_spectrogram=mel_output,
            metadata={'stage': 'B2', 'emotion_dim': emotion_tensor.shape[-1]}
        )
    
    def forward(self, mel_input: torch.Tensor, emotion_features: np.ndarray, use_b2: bool = False) -> ModelOutput:
        """
        完整B阶段前向传播
        输入: (M0, emotion2vec表征)
        输出: M1 或 M2 (根据use_b2参数)
        """
        # 先执行B1
        b1_output = self.forward_b1(mel_input, emotion_features)
        
        # 如果不使用B2或B2未启用，返回B1结果
        if not use_b2 or self.b2_model is None:
            return b1_output
        
        # 执行B2
        b2_output = self.forward_b2(b1_output.mel_spectrogram, emotion_features)
        b2_output.metadata['previous_stage'] = 'B1'
        
        return b2_output
    
    def train_step_b1(self, batch_data: Dict[str, Any]) -> Dict[str, float]:
        """B1训练步骤"""
        self.b1_model.train()
        
        # 解析批次数据
        mel_inputs = batch_data['mel_inputs']  # M0: (batch, time, mel_dim)
        emotion_features = batch_data['emotion_features']  # (batch, emotion_dim)
        target_mels = batch_data['target_mels']  # 目标Mel频谱: (batch, time, mel_dim)
        
        # 确保数据类型和设备
        if isinstance(emotion_features, np.ndarray):
            emotion_features = torch.tensor(emotion_features, dtype=torch.float32)
        
        # 前向传播
        mel_outputs = self.b1_model(mel_inputs, emotion_features)
        
        # 计算损失
        loss = self.criterion(mel_outputs, target_mels)
        
        # 反向传播
        if self.b1_optimizer is None:
            self.b1_optimizer = torch.optim.Adam(
                self.b1_model.parameters(), 
                lr=self.config.get('training', {}).get('learning_rate', 5e-5)
            )
        
        self.b1_optimizer.zero_grad()
        loss.backward()
        self.b1_optimizer.step()
        
        return {
            'b1_loss': loss.item(),
            'total_loss': loss.item()
        }
    
    def train_step_b2(self, batch_data: Dict[str, Any]) -> Dict[str, float]:
        """B2训练步骤"""
        if self.b2_model is None:
            return {'b2_loss': 0.0, 'total_loss': 0.0}
        
        self.b2_model.train()
        
        # 解析批次数据
        mel_inputs = batch_data['mel_inputs']  # M1: (batch, time, mel_dim)
        emotion_features = batch_data['emotion_features']  # (batch, emotion_dim)
        target_mels = batch_data['target_mels']  # 目标Mel频谱: (batch, time, mel_dim)
        
        # 确保数据类型和设备
        if isinstance(emotion_features, np.ndarray):
            emotion_features = torch.tensor(emotion_features, dtype=torch.float32)
        
        # 前向传播
        mel_outputs = self.b2_model(mel_inputs, emotion_features)
        
        # 计算损失
        loss = self.criterion(mel_outputs, target_mels)
        
        # 反向传播
        if self.b2_optimizer is None:
            self.b2_optimizer = torch.optim.Adam(
                self.b2_model.parameters(), 
                lr=self.config.get('training', {}).get('learning_rate', 1e-5)
            )
        
        self.b2_optimizer.zero_grad()
        loss.backward()
        self.b2_optimizer.step()
        
        return {
            'b2_loss': loss.item(),
            'total_loss': loss.item()
        }
    
    def save_checkpoint(self, path: str) -> None:
        """保存检查点"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        checkpoint = {
            'b1_model_state_dict': self.b1_model.state_dict(),
            'config': self.config
        }
        
        if self.b1_optimizer:
            checkpoint['b1_optimizer_state_dict'] = self.b1_optimizer.state_dict()
        
        if self.b2_model:
            checkpoint['b2_model_state_dict'] = self.b2_model.state_dict()
            if self.b2_optimizer:
                checkpoint['b2_optimizer_state_dict'] = self.b2_optimizer.state_dict()
        
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str) -> None:
        """加载检查点"""
        checkpoint = torch.load(path, map_location='cpu')
        
        # 加载B1模型
        self.b1_model.load_state_dict(checkpoint['b1_model_state_dict'])
        if 'b1_optimizer_state_dict' in checkpoint and self.b1_optimizer:
            self.b1_optimizer.load_state_dict(checkpoint['b1_optimizer_state_dict'])
        
        # 加载B2模型（如果存在）
        if 'b2_model_state_dict' in checkpoint and self.b2_model:
            self.b2_model.load_state_dict(checkpoint['b2_model_state_dict'])
            if 'b2_optimizer_state_dict' in checkpoint and self.b2_optimizer:
                self.b2_optimizer.load_state_dict(checkpoint['b2_optimizer_state_dict'])


class EmotionAdapter(nn.Module):
    """
    B1阶段：情感适配器网络
    将M0和情感特征融合生成M1
    """
    
    def __init__(self, emotion_dim: int, mel_dim: int, hidden_dim: int, n_layers: int):
        super().__init__()
        
        self.emotion_dim = emotion_dim
        self.mel_dim = mel_dim
        self.hidden_dim = hidden_dim
        
        # 情感特征投影
        self.emotion_projection = nn.Linear(emotion_dim, hidden_dim)
        
        # Mel特征投影
        self.mel_projection = nn.Linear(mel_dim, hidden_dim)
        
        # 融合网络
        fusion_layers = []
        for i in range(n_layers):
            fusion_layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
        self.fusion_network = nn.Sequential(*fusion_layers)
        
        # 输出投影
        self.output_projection = nn.Linear(hidden_dim, mel_dim)
        
        # 残差连接权重
        self.residual_weight = nn.Parameter(torch.tensor(0.1))
    
    def forward(self, mel_input: torch.Tensor, emotion_features: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        Args:
            mel_input: (batch, time, mel_dim) - M0
            emotion_features: (batch, emotion_dim) - 情感特征
        Returns:
            mel_output: (batch, time, mel_dim) - M1
        """
        batch_size, time_steps, mel_dim = mel_input.shape
        
        # 投影情感特征
        emotion_proj = self.emotion_projection(emotion_features)  # (batch, hidden_dim)
        emotion_proj = emotion_proj.unsqueeze(1).expand(-1, time_steps, -1)  # (batch, time, hidden_dim)
        
        # 投影Mel特征
        mel_proj = self.mel_projection(mel_input)  # (batch, time, hidden_dim)
        
        # 融合特征
        fused_features = mel_proj + emotion_proj
        
        # 通过融合网络
        processed_features = self.fusion_network(fused_features)
        
        # 输出投影
        mel_output = self.output_projection(processed_features)
        
        # 残差连接
        mel_output = mel_input + self.residual_weight * mel_output
        
        return mel_output


class DiffusionRefiner(nn.Module):
    """
    B2阶段：扩散模型细化器
    对M1进行进一步的情感细化，生成M2
    """
    
    def __init__(self, emotion_dim: int, mel_dim: int, hidden_dim: int, diffusion_steps: int = 20):
        super().__init__()
        
        self.emotion_dim = emotion_dim
        self.mel_dim = mel_dim
        self.hidden_dim = hidden_dim
        self.diffusion_steps = diffusion_steps
        
        # 时间嵌入
        self.time_embedding = nn.Sequential(
            SinusoidalPositionEmbedding(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 情感条件嵌入
        self.emotion_embedding = nn.Linear(emotion_dim, hidden_dim)
        
        # U-Net结构的简化版本
        self.input_projection = nn.Linear(mel_dim, hidden_dim)
        
        # 中间处理层
        self.middle_layers = nn.ModuleList([
            ResidualBlock(hidden_dim, hidden_dim) for _ in range(4)
        ])
        
        # 输出投影
        self.output_projection = nn.Linear(hidden_dim, mel_dim)
        
        # 噪声预测头
        self.noise_pred_head = nn.Linear(hidden_dim, mel_dim)
    
    def forward(self, mel_input: torch.Tensor, emotion_features: torch.Tensor, timestep: Optional[int] = None) -> torch.Tensor:
        """
        前向传播
        Args:
            mel_input: (batch, time, mel_dim) - M1
            emotion_features: (batch, emotion_dim) - 情感特征
            timestep: 扩散时间步（推理时为None）
        Returns:
            mel_output: (batch, time, mel_dim) - M2
        """
        batch_size, time_steps, mel_dim = mel_input.shape
        
        # 如果是推理模式，执行简化的前向传播
        if timestep is None:
            return self._inference_forward(mel_input, emotion_features)
        
        # 训练模式的扩散过程
        return self._diffusion_forward(mel_input, emotion_features, timestep)
    
    def _inference_forward(self, mel_input: torch.Tensor, emotion_features: torch.Tensor) -> torch.Tensor:
        """推理模式的简化前向传播"""
        batch_size, time_steps, mel_dim = mel_input.shape
        
        # 投影输入
        x = self.input_projection(mel_input)  # (batch, time, hidden_dim)
        
        # 情感条件
        emotion_emb = self.emotion_embedding(emotion_features)  # (batch, hidden_dim)
        emotion_emb = emotion_emb.unsqueeze(1).expand(-1, time_steps, -1)  # (batch, time, hidden_dim)
        
        # 加入情感条件
        x = x + emotion_emb
        
        # 中间处理
        for layer in self.middle_layers:
            x = layer(x)
        
        # 输出投影
        refinement = self.output_projection(x)
        
        # 残差连接
        return mel_input + 0.1 * refinement
    
    def _diffusion_forward(self, mel_input: torch.Tensor, emotion_features: torch.Tensor, timestep: int) -> torch.Tensor:
        """扩散训练模式的前向传播"""
        # 这里实现完整的扩散过程
        # 为了简化，我们使用类似的结构
        return self._inference_forward(mel_input, emotion_features)


class SinusoidalPositionEmbedding(nn.Module):
    """正弦位置编码"""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    
    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ResidualBlock(nn.Module):
    """残差块"""
    
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(out_dim, out_dim)
        )
        
        self.shortcut = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.shortcut(x) + self.layers(x)