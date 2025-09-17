"""
Emotion Data Reader for FlowSE Inference
Handles loading of Vevo mel and emotion features for inference
"""

import librosa
import numpy as np
import soundfile as sf
import torch
import json
import os


class EmotionDataReader(object):
    """
    Data reader for emotion2vec + FlowSE inference
    
    Loads Vevo mel spectrograms and emotion features for inference
    """
    
    def __init__(self,
                 vevo_mel_json,
                 vevo_mel_dir,
                 emotion_features_json=None,
                 sample_rate=16000):

        # Load vevo mel file list
        with open(vevo_mel_json, 'r') as f:
            self.vevo_mel_json = json.load(f) 
        self.utt_ids = list(self.vevo_mel_json.keys())
        
        self.vevo_mel_dir = vevo_mel_dir
        self.sample_rate = sample_rate
        
        # Load emotion features if available
        if emotion_features_json and os.path.exists(emotion_features_json):
            with open(emotion_features_json, 'r') as f:
                self.emotion_features = json.load(f)
        else:
            self.emotion_features = {}

    def extract_feature(self, utt_id):
        """Extract features for a single utterance"""
        
        # Load Vevo mel spectrogram
        vevo_mel_path = os.path.join(self.vevo_mel_dir, utt_id)
        if not vevo_mel_path.endswith('.npy'):
            vevo_mel_path += '.npy'
            
        if os.path.exists(vevo_mel_path):
            vevo_mel = np.load(vevo_mel_path)
            vevo_mel = torch.from_numpy(vevo_mel).float()
        else:
            # Create dummy mel if file doesn't exist
            print(f"Warning: Vevo mel file not found for {utt_id}, using dummy data")
            vevo_mel = torch.randn(100, 100)  # [frames, mel_channels]
        
        # Ensure correct shape [1, frames, mel_channels] -> [1, mel_channels, frames] 
        if vevo_mel.ndim == 2:
            vevo_mel = vevo_mel.unsqueeze(0).transpose(1, 2)  # [1, mel_channels, frames]
        elif vevo_mel.ndim == 3 and vevo_mel.shape[0] == 1:
            vevo_mel = vevo_mel.transpose(1, 2)  # [1, mel_channels, frames]
        
        # Load emotion features
        if utt_id in self.emotion_features:
            emotion_features = {
                'utterance': torch.tensor(self.emotion_features[utt_id]['utterance'], dtype=torch.float32),
                'frame': torch.tensor(self.emotion_features[utt_id]['frame'], dtype=torch.float32)
            }
        else:
            # Create dummy emotion features
            print(f"Warning: Emotion features not found for {utt_id}, using dummy data")
            emotion_features = {
                'utterance': torch.randn(768),
                'frame': torch.randn(vevo_mel.shape[2], 768)  # Match mel frame length
            }
        
        # Ensure emotion features are properly shaped
        if emotion_features['utterance'].ndim == 1:
            emotion_features['utterance'] = emotion_features['utterance'].unsqueeze(0)  # [1, dim]
        
        if emotion_features['frame'].ndim == 2:
            emotion_features['frame'] = emotion_features['frame'].unsqueeze(0)  # [1, frames, dim]
        
        # Align frame features with mel length
        mel_frames = vevo_mel.shape[2]
        if emotion_features['frame'].shape[1] != mel_frames:
            # Interpolate to match mel frames
            frame_features = emotion_features['frame'].transpose(1, 2)  # [1, dim, frames]
            frame_features = torch.nn.functional.interpolate(
                frame_features, size=mel_frames, mode='linear', align_corners=False
            )
            emotion_features['frame'] = frame_features.transpose(1, 2)  # [1, frames, dim]

        egs = {
            'utt_id': utt_id,
            'vevo_mel': vevo_mel,  # [1, mel_channels, frames]
            'emotion_features': emotion_features
        }

        return egs

    def __len__(self):
        return len(self.utt_ids)

    def __getitem__(self, index):
        return self.extract_feature(self.utt_ids[index])
        
    def __iter__(self):
        for utt_id in self.utt_ids:
            yield self.extract_feature(utt_id)
