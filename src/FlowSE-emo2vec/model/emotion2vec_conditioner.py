"""
Emotion2Vec Conditioner for FlowSE
Replaces text conditioning with emotion2vec features
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from .modules import ConvNeXtV2Block, precompute_freqs_cis, get_pos_embed_indices


class Emotion2VecConditioner(nn.Module):
    """
    Emotion2Vec feature conditioner that replaces TextEmbedding
    
    Processes both utterance-level and frame-level emotion2vec features
    and outputs sequence-level conditioning features compatible with FlowSE
    """
    
    def __init__(self, 
                 e2v_utterance_dim=768,  # emotion2vec utterance feature dimension
                 e2v_frame_dim=768,      # emotion2vec frame feature dimension
                 output_dim=512,         # output conditioning dimension
                 conv_layers=0,          # number of additional conv layers
                 conv_mult=2,            # conv expansion multiplier
                 fusion_mode='add'):     # 'add', 'concat', 'gate'
        super().__init__()
        
        self.e2v_utterance_dim = e2v_utterance_dim
        self.e2v_frame_dim = e2v_frame_dim
        self.output_dim = output_dim
        self.fusion_mode = fusion_mode
        
        # Utterance-level feature projection (global conditioning)
        self.utterance_proj = nn.Linear(e2v_utterance_dim, output_dim)
        
        # Frame-level feature projection (sequence conditioning)
        self.frame_proj = nn.Linear(e2v_frame_dim, output_dim)
        
        # Fusion layer for combining utterance and frame features
        if fusion_mode == 'concat':
            self.fusion_proj = nn.Linear(output_dim * 2, output_dim)
        elif fusion_mode == 'gate':
            self.gate = nn.Sequential(
                nn.Linear(output_dim * 2, output_dim),
                nn.Sigmoid()
            )
        
        # Optional extra modeling layers
        if conv_layers > 0:
            self.extra_modeling = True
            self.precompute_max_pos = 4096  # ~44s of 24khz audio
            self.register_buffer("freqs_cis", precompute_freqs_cis(output_dim, self.precompute_max_pos), persistent=False)
            self.conv_blocks = nn.Sequential(
                *[ConvNeXtV2Block(output_dim, output_dim * conv_mult) for _ in range(conv_layers)]
            )
        else:
            self.extra_modeling = False
            
        # Layer normalization for stable training
        self.layer_norm = nn.LayerNorm(output_dim)
        
    def forward(self, emotion_features, seq_len, drop_emotion=False):
        """
        Args:
            emotion_features: dict with keys:
                - 'utterance': [batch, e2v_utterance_dim] - utterance-level features
                - 'frame': [batch, frame_len, e2v_frame_dim] - frame-level features
            seq_len: target sequence length (mel frames)
            drop_emotion: whether to drop emotion conditioning (for CFG)
            
        Returns:
            emotion_embed: [batch, seq_len, output_dim] - conditioning features
        """
        batch = emotion_features["utterance"].shape[0]
        device = emotion_features["utterance"].device
        
        if drop_emotion:  # Classifier-free guidance support
            return torch.zeros(batch, seq_len, self.output_dim, device=device, dtype=emotion_features["utterance"].dtype)
        
        # Process utterance-level features -> broadcast to sequence
        utterance_embed = self.utterance_proj(emotion_features["utterance"])  # [batch, output_dim]
        utterance_embed = utterance_embed.unsqueeze(1).expand(-1, seq_len, -1)  # [batch, seq_len, output_dim]
        
        # Process frame-level features -> interpolate to target length
        frame_embed = self.frame_proj(emotion_features["frame"])  # [batch, frame_len, output_dim]
        
        # Interpolate frame features to match mel sequence length
        if frame_embed.shape[1] != seq_len:
            frame_embed = F.interpolate(
                frame_embed.transpose(1, 2),  # [batch, output_dim, frame_len]
                size=seq_len, 
                mode='linear', 
                align_corners=False
            ).transpose(1, 2)  # [batch, seq_len, output_dim]
        
        # Fuse utterance and frame features
        if self.fusion_mode == 'add':
            emotion_embed = utterance_embed + frame_embed
        elif self.fusion_mode == 'concat':
            emotion_embed = torch.cat([utterance_embed, frame_embed], dim=-1)
            emotion_embed = self.fusion_proj(emotion_embed)
        elif self.fusion_mode == 'gate':
            combined = torch.cat([utterance_embed, frame_embed], dim=-1)
            gate = self.gate(combined)
            emotion_embed = gate * utterance_embed + (1 - gate) * frame_embed
        else:
            raise ValueError(f"Unknown fusion_mode: {self.fusion_mode}")
        
        # Apply layer normalization
        emotion_embed = self.layer_norm(emotion_embed)
        
        # Optional extra modeling
        if self.extra_modeling:
            # Add positional embeddings
            batch_start = torch.zeros((batch,), dtype=torch.long, device=device)
            pos_idx = get_pos_embed_indices(batch_start, seq_len, max_pos=self.precompute_max_pos)
            pos_embed = self.freqs_cis[pos_idx]
            emotion_embed = emotion_embed + pos_embed
            
            # Apply conv blocks
            emotion_embed = self.conv_blocks(emotion_embed)
            
        return emotion_embed


class EmotionProcessor:
    """
    Helper class to process raw emotion2vec features
    Handles feature extraction and preprocessing
    """
    
    def __init__(self, emotion2vec_model=None, device='cuda'):
        self.emotion2vec_model = emotion2vec_model
        self.device = device
        
    def extract_features(self, audio_tensor, sample_rate=16000):
        """
        Extract emotion2vec features from audio
        
        Args:
            audio_tensor: [batch, samples] or [samples] audio tensor
            sample_rate: audio sample rate (should be 16kHz for emotion2vec)
            
        Returns:
            dict with 'utterance' and 'frame' features
        """
        if self.emotion2vec_model is None:
            # Placeholder implementation - return dummy features
            if audio_tensor.ndim == 1:
                audio_tensor = audio_tensor.unsqueeze(0)
            batch = audio_tensor.shape[0]
            audio_len = audio_tensor.shape[1]
            
            # Dummy features matching emotion2vec dimensions
            utterance_features = torch.randn(batch, 768, device=self.device)
            
            # Frame-level features at 50Hz (emotion2vec frame rate)
            frame_len = int(audio_len / sample_rate * 50)  # 50Hz frame rate
            frame_features = torch.randn(batch, frame_len, 768, device=self.device)
            
            return {
                'utterance': utterance_features,
                'frame': frame_features
            }
        else:
            # Real emotion2vec feature extraction
            # This would be implemented with actual emotion2vec model
            with torch.no_grad():
                features = self.emotion2vec_model.extract_features(audio_tensor)
                return features
    
    def process_features_from_dict(self, feature_dict):
        """
        Process pre-extracted features from dictionary
        
        Args:
            feature_dict: dict containing emotion features
            
        Returns:
            processed features ready for conditioning
        """
        return {
            'utterance': torch.tensor(feature_dict['utterance'], device=self.device),
            'frame': torch.tensor(feature_dict['frame'], device=self.device)
        }
