"""
Test script for FlowSE-Emo2Vec integration
Verifies that all components work together correctly
"""

import torch
import numpy as np
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(__file__))

from model import DiT, CFM, Emotion2VecConditioner, EmotionProcessor
from model.emotion2vec_conditioner import Emotion2VecConditioner
from loader.emotion_dataloader import EmotionDataset, make_emotion_loader
from loader.emotion_datareader import EmotionDataReader


def test_emotion2vec_conditioner():
    """Test Emotion2VecConditioner module"""
    print("Testing Emotion2VecConditioner...")
    
    # Initialize conditioner
    conditioner = Emotion2VecConditioner(
        e2v_utterance_dim=768,
        e2v_frame_dim=768,
        output_dim=512,
        conv_layers=2
    )
    
    # Create dummy emotion features
    batch_size = 2
    seq_len = 100
    emotion_features = {
        'utterance': torch.randn(batch_size, 768),
        'frame': torch.randn(batch_size, 80, 768)  # Different frame length
    }
    
    # Test forward pass
    output = conditioner(emotion_features, seq_len, drop_emotion=False)
    assert output.shape == (batch_size, seq_len, 512), f"Expected shape {(batch_size, seq_len, 512)}, got {output.shape}"
    
    # Test dropout
    output_dropped = conditioner(emotion_features, seq_len, drop_emotion=True)
    assert torch.all(output_dropped == 0), "Dropped output should be all zeros"
    
    print("✅ Emotion2VecConditioner test passed!")


def test_dit_model():
    """Test DiT model with emotion conditioning"""
    print("Testing DiT model...")
    
    # Initialize DiT model
    model = DiT(
        dim=512,
        depth=4,
        heads=8,
        mel_dim=100,
        e2v_utterance_dim=768,
        e2v_frame_dim=768,
        emotion_dim=256
    )
    
    # Create dummy inputs
    batch_size = 2
    seq_len = 100
    mel_dim = 100
    
    x = torch.randn(batch_size, seq_len, mel_dim)
    cond = torch.randn(batch_size, seq_len, mel_dim)
    emotion_features = {
        'utterance': torch.randn(batch_size, 768),
        'frame': torch.randn(batch_size, seq_len, 768)
    }
    time = torch.rand(batch_size)
    
    # Test forward pass
    output = model(
        x=x, 
        cond=cond, 
        emotion_features=emotion_features, 
        time=time, 
        drop_audio_cond=False, 
        drop_emotion=False
    )
    
    assert output.shape == (batch_size, seq_len, mel_dim), f"Expected shape {(batch_size, seq_len, mel_dim)}, got {output.shape}"
    
    print("✅ DiT model test passed!")


def test_cfm_model():
    """Test CFM model with emotion conditioning"""
    print("Testing CFM model...")
    
    # Initialize CFM model
    dit_model = DiT(
        dim=512,
        depth=2,  # Smaller for testing
        heads=4,
        mel_dim=100,
        e2v_utterance_dim=768,
        e2v_frame_dim=768,
        emotion_dim=256
    )
    
    emotion_processor = EmotionProcessor()
    
    cfm_model = CFM(
        transformer=dit_model,
        emotion_processor=emotion_processor,
        mel_spec_kwargs={
            'target_sample_rate': 24000,
            'n_mel_channels': 100,
            'hop_length': 256,
            'win_length': 1024,
            'n_fft': 1024
        }
    )
    
    # Create dummy inputs
    batch_size = 2
    seq_len = 100
    mel_dim = 100
    
    inp = torch.randn(batch_size, seq_len, mel_dim)
    clean = torch.randn(batch_size, seq_len, mel_dim)
    emotion_features = {
        'utterance': torch.randn(batch_size, 768),
        'frame': torch.randn(batch_size, seq_len, 768)
    }
    
    # Test forward pass (training)
    loss, cond, pred = cfm_model(inp=inp, clean=clean, emotion_features=emotion_features)
    
    assert isinstance(loss, torch.Tensor), "Loss should be a tensor"
    assert loss.dim() == 0, "Loss should be scalar"
    assert pred.shape == clean.shape, f"Prediction shape {pred.shape} should match clean shape {clean.shape}"
    
    # Test sampling (inference)
    cfm_model.eval()
    with torch.no_grad():
        output, trajectory = cfm_model.sample(
            cond=inp,
            emotion_features=emotion_features,
            steps=10  # Small number for testing
        )
    
    assert output.shape == inp.shape, f"Output shape {output.shape} should match input shape {inp.shape}"
    
    print("✅ CFM model test passed!")


def test_data_loading():
    """Test emotion data loading"""
    print("Testing emotion data loading...")
    
    # Create a simple dataset
    dataset = EmotionDataset(
        mel_spec_kwargs={
            'target_sample_rate': 24000,
            'n_mel_channels': 100,
            'hop_length': 256,
            'win_length': 1024,
            'n_fft': 1024
        },
        use_precomputed_emotion=True
    )
    
    # Test dataset
    assert len(dataset) > 0, "Dataset should have at least one sample"
    
    sample = dataset[0]
    assert 'utt_id' in sample, "Sample should have utt_id"
    assert 'vevo_mel' in sample, "Sample should have vevo_mel"
    assert 'source_mel' in sample, "Sample should have source_mel"
    assert 'emotion_features' in sample, "Sample should have emotion_features"
    
    print("✅ Data loading test passed!")


def test_data_reader():
    """Test emotion data reader for inference"""
    print("Testing emotion data reader...")
    
    # Create data reader with test files
    data_reader = EmotionDataReader(
        vevo_mel_json="datalist/test/vevo_mel.json",
        vevo_mel_dir="datalist/test/vevo_mel",
        emotion_features_json="datalist/test/emotion_features.json"
    )
    
    # Test data reader
    assert len(data_reader) > 0, "Data reader should have at least one sample"
    
    sample = data_reader[0]
    assert 'utt_id' in sample, "Sample should have utt_id"
    assert 'vevo_mel' in sample, "Sample should have vevo_mel"
    assert 'emotion_features' in sample, "Sample should have emotion_features"
    
    print("✅ Data reader test passed!")


def run_all_tests():
    """Run all integration tests"""
    print("🚀 Starting FlowSE-Emo2Vec integration tests...")
    print("=" * 50)
    
    try:
        test_emotion2vec_conditioner()
        test_dit_model()
        test_cfm_model()
        test_data_loading()
        test_data_reader()
        
        print("=" * 50)
        print("🎉 All tests passed! FlowSE-Emo2Vec integration is working correctly.")
        print("\n📝 Next steps:")
        print("1. Prepare your training data (vevo mel, source mel, emotion features)")
        print("2. Update configuration files with correct paths")
        print("3. Integrate real emotion2vec model in EmotionProcessor")
        print("4. Start training with: python train.py -conf config/train.yaml")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_all_tests()
