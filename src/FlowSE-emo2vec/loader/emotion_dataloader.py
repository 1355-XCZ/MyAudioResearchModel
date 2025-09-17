"""
Emotion2Vec Data Loader for FlowSE
Handles loading of Vevo mel, source mel, and emotion2vec features
"""

import numpy as np
import torch
import torch.utils.data as tud
from torch.utils.data import DataLoader, Dataset
import json
import os
import soundfile as sf
import librosa
from pathlib import Path
import sys
sys.path.append("../")
from model.modules import MelSpec
from model.emotion2vec_conditioner import EmotionProcessor

EPS = np.finfo(float).eps


def normalize(audio, target_level=-25):
    """Normalize the signal to the target level using PyTorch"""
    rms = torch.sqrt(torch.mean(audio**2))
    scalar = 10 ** (target_level / 20) / (rms + EPS)
    audio = audio * scalar
    return audio


class EmotionDataset(Dataset):
    """
    Dataset for Emotion2Vec + FlowSE training
    
    Loads:
    - Vevo mel spectrograms (neutral + timbre)
    - Source mel spectrograms (with emotion)
    - Emotion2vec features (utterance + frame level)
    """
    
    def __init__(self, 
                 vevo_mel_scp=None,
                 source_mel_scp=None, 
                 source_audio_scp=None,
                 emotion_features_scp=None,
                 mel_spec_kwargs=None,
                 sample_rate=16000,
                 max_samples=64,
                 use_precomputed_emotion=True):
        
        self.sample_rate = sample_rate
        self.max_samples = max_samples
        self.use_precomputed_emotion = use_precomputed_emotion
        
        # Initialize mel spectrogram extractor
        if mel_spec_kwargs is None:
            mel_spec_kwargs = {
                'target_sample_rate': 24000,
                'n_mel_channels': 100,
                'hop_length': 256,
                'win_length': 1024,
                'n_fft': 1024
            }
        self.mel_spec = MelSpec(**mel_spec_kwargs)
        
        # Initialize emotion processor
        self.emotion_processor = EmotionProcessor()
        
        # Load file lists
        self.vevo_mel_files = self._load_scp_file(vevo_mel_scp) if vevo_mel_scp else {}
        self.source_mel_files = self._load_scp_file(source_mel_scp) if source_mel_scp else {}
        self.source_audio_files = self._load_scp_file(source_audio_scp) if source_audio_scp else {}
        
        # Load emotion features
        if emotion_features_scp and os.path.exists(emotion_features_scp):
            with open(emotion_features_scp, 'r') as f:
                self.emotion_features = json.load(f)
        else:
            self.emotion_features = {}
        
        # Get common utterance IDs
        self.utt_ids = list(set(self.vevo_mel_files.keys()) & 
                           set(self.source_mel_files.keys()) &
                           set(self.source_audio_files.keys()))
        
        if len(self.utt_ids) == 0:
            print("Warning: No common utterance IDs found across all data sources")
            # For demo purposes, create some dummy data
            self.utt_ids = ['dummy_utt_1', 'dummy_utt_2', 'dummy_utt_3']
    
    def _load_scp_file(self, scp_path):
        """Load .scp file format: utt_id /path/to/file"""
        files = {}
        if os.path.exists(scp_path):
            with open(scp_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        utt_id = parts[0]
                        file_path = ' '.join(parts[1:])
                        files[utt_id] = file_path
        return files
    
    def _load_mel_spectrogram(self, file_path):
        """Load mel spectrogram from file"""
        if file_path.endswith('.npy'):
            # Load pre-computed mel spectrogram
            mel = np.load(file_path)
            return torch.from_numpy(mel).float()
        else:
            # Load audio and compute mel spectrogram
            audio, sr = sf.read(file_path)
            if sr != self.sample_rate:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
            
            audio = torch.from_numpy(audio).float().unsqueeze(0)  # [1, samples]
            mel = self.mel_spec(audio)  # [1, mel_channels, frames]
            return mel.squeeze(0).transpose(0, 1)  # [frames, mel_channels]
    
    def _load_emotion_features(self, utt_id):
        """Load or extract emotion2vec features"""
        if self.use_precomputed_emotion and utt_id in self.emotion_features:
            # Use pre-computed features
            features = self.emotion_features[utt_id]
            return {
                'utterance': torch.tensor(features['utterance'], dtype=torch.float32),
                'frame': torch.tensor(features['frame'], dtype=torch.float32)
            }
        elif utt_id in self.source_audio_files:
            # Extract features from audio
            audio_path = self.source_audio_files[utt_id]
            if os.path.exists(audio_path):
                audio, sr = sf.read(audio_path)
                if sr != 16000:  # emotion2vec expects 16kHz
                    audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
                
                audio_tensor = torch.from_numpy(audio).float()
                return self.emotion_processor.extract_features(audio_tensor, sample_rate=16000)
            else:
                # Return dummy features if file doesn't exist
                return self._get_dummy_emotion_features()
        else:
            # Return dummy features as fallback
            return self._get_dummy_emotion_features()
    
    def _get_dummy_emotion_features(self):
        """Generate dummy emotion features for testing"""
        return {
            'utterance': torch.randn(768),
            'frame': torch.randn(100, 768)  # Assume ~100 frames for typical utterance
        }
    
    def __len__(self):
        return len(self.utt_ids)
    
    def __getitem__(self, idx):
        utt_id = self.utt_ids[idx]
        
        try:
            # Load Vevo mel (neutral + timbre)
            if utt_id in self.vevo_mel_files:
                vevo_mel = self._load_mel_spectrogram(self.vevo_mel_files[utt_id])
            else:
                # Dummy vevo mel
                vevo_mel = torch.randn(100, 100)  # [frames, mel_channels]
            
            # Load source mel (with emotion)
            if utt_id in self.source_mel_files:
                source_mel = self._load_mel_spectrogram(self.source_mel_files[utt_id])
            else:
                # Dummy source mel
                source_mel = torch.randn(100, 100)
            
            # Ensure both mels have the same length
            min_len = min(vevo_mel.shape[0], source_mel.shape[0])
            vevo_mel = vevo_mel[:min_len]
            source_mel = source_mel[:min_len]
            
            # Load emotion features
            emotion_features = self._load_emotion_features(utt_id)
            
            # Ensure frame-level emotion features match mel length
            if emotion_features['frame'].shape[0] != min_len:
                # Interpolate to match mel length
                frame_features = emotion_features['frame'].unsqueeze(0).transpose(1, 2)  # [1, dim, frames]
                frame_features = torch.nn.functional.interpolate(
                    frame_features, size=min_len, mode='linear', align_corners=False
                )
                emotion_features['frame'] = frame_features.squeeze(0).transpose(0, 1)  # [frames, dim]
            
            return {
                'utt_id': utt_id,
                'vevo_mel': vevo_mel.transpose(0, 1),  # [mel_channels, frames] for CFM
                'source_mel': source_mel.transpose(0, 1),  # [mel_channels, frames] for CFM
                'emotion_features': emotion_features
            }
            
        except Exception as e:
            print(f"Error loading data for {utt_id}: {e}")
            # Return dummy data
            return {
                'utt_id': utt_id,
                'vevo_mel': torch.randn(100, 100),
                'source_mel': torch.randn(100, 100),
                'emotion_features': self._get_dummy_emotion_features()
            }


def collate_fn(batch):
    """
    Collate function for batching emotion data
    Handles variable-length sequences
    """
    utt_ids = [item['utt_id'] for item in batch]
    
    # Get max length for padding
    max_frames = max(item['vevo_mel'].shape[1] for item in batch)
    mel_channels = batch[0]['vevo_mel'].shape[0]
    batch_size = len(batch)
    
    # Initialize tensors
    vevo_mels = torch.zeros(batch_size, mel_channels, max_frames)
    source_mels = torch.zeros(batch_size, mel_channels, max_frames)
    
    # Emotion features
    utterance_features = torch.stack([item['emotion_features']['utterance'] for item in batch])
    
    # Frame features - pad to max length
    e2v_dim = batch[0]['emotion_features']['frame'].shape[1]
    frame_features = torch.zeros(batch_size, max_frames, e2v_dim)
    
    for i, item in enumerate(batch):
        frames = item['vevo_mel'].shape[1]
        vevo_mels[i, :, :frames] = item['vevo_mel']
        source_mels[i, :, :frames] = item['source_mel']
        frame_features[i, :frames, :] = item['emotion_features']['frame']
    
    return {
        'utt_id': utt_ids,
        'vevo_mel': vevo_mels,
        'source_mel': source_mels,
        'emotion_features': {
            'utterance': utterance_features,
            'frame': frame_features
        }
    }


def make_emotion_loader(vevo_mel_scp=None,
                       source_mel_scp=None,
                       source_audio_scp=None,
                       emotion_features_scp=None,
                       batch_size=4,
                       num_workers=4,
                       shuffle=True,
                       **kwargs):
    """
    Create data loader for emotion2vec + FlowSE training
    """
    dataset = EmotionDataset(
        vevo_mel_scp=vevo_mel_scp,
        source_mel_scp=source_mel_scp,
        source_audio_scp=source_audio_scp,
        emotion_features_scp=emotion_features_scp,
        **kwargs
    )
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    return None, loader  # Return (sampler, loader) to match original interface
