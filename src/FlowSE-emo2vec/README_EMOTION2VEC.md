# FlowSE-Emo2Vec: Emotion2Vec Conditioned FlowSE

This is a modified version of FlowSE that uses **emotion2vec** features instead of text transcriptions for speech enhancement conditioning.

## Key Changes from Original FlowSE

### 🔄 Model Architecture Changes

1. **Replaced TextEmbedding with Emotion2VecConditioner**
   - `TextEmbedding` → `Emotion2VecConditioner` 
   - Processes both utterance-level and frame-level emotion2vec features
   - Supports multiple fusion modes: add, concat, gate

2. **Updated Model Interfaces**
   - `text` parameter → `emotion_features` parameter
   - `drop_text` → `drop_emotion` for classifier-free guidance
   - Support for emotion strength control via CFG

3. **New Data Pipeline**
   - `vevo_mel` (neutral + timbre) as input condition
   - `source_mel` (with emotion) as target
   - `emotion2vec` features as conditioning

### 📁 New Files Added

- `model/emotion2vec_conditioner.py` - Emotion conditioning module
- `loader/emotion_dataloader.py` - Training data loader
- `loader/emotion_datareader.py` - Inference data reader
- `datalist/test/vevo_mel.json` - Sample vevo mel list
- `datalist/test/emotion_features.json` - Sample emotion features

### ⚙️ Configuration Changes

**Model Configuration:**
```yaml
model:
  emotion_processor: emotion2vec
  arch:
    e2v_utterance_dim: 768    # emotion2vec utterance feature dim
    e2v_frame_dim: 768        # emotion2vec frame feature dim  
    emotion_dim: 512          # output emotion conditioning dim
```

**Dataset Configuration:**
```yaml
datasets:
  train:
    vevo_mel_scp: datalist/train/vevo_mel.scp
    source_mel_scp: datalist/train/source_mel.scp
    source_audio_scp: datalist/train/source_audio.scp
    emotion_features_scp: datalist/train/emotion_features.json
```

**Inference Configuration:**
```yaml
infer:
  test:
    cond_type: emotion  # 'emotion' or 'no_emotion'
  datareader:
    vevo_mel_json: datalist/test/vevo_mel.json
    emotion_features_json: datalist/test/emotion_features.json
```

## 🚀 Usage

### Training
```bash
python train.py -conf config/train.yaml
```

### Inference
```bash
python infer.py -conf config/train.yaml
```

## 📊 Data Format

### Vevo Mel Files (.scp format)
```
utt_001 /path/to/vevo_mel/utt_001.npy
utt_002 /path/to/vevo_mel/utt_002.npy
```

### Source Mel Files (.scp format)
```
utt_001 /path/to/source_mel/utt_001.npy
utt_002 /path/to/source_mel/utt_002.npy
```

### Emotion Features (.json format)
```json
{
    "utt_001": {
        "utterance": [768-dim vector],
        "frame": [[frame1_768dim], [frame2_768dim], ...]
    }
}
```

## 🔧 Integration with Your Pipeline

This FlowSE-Emo2Vec module is designed to integrate with your existing pipeline:

1. **Stage A (Vevo)**: Generates neutral mel → used as FlowSE input condition
2. **Emotion2Vec**: Extracts emotion features → used as FlowSE emotion condition
3. **FlowSE-Emo2Vec**: Maps `neutral_mel + emotion → source_mel`
4. **Vocoder**: Converts enhanced mel → final audio

## 🎯 Key Benefits

- ✅ **Minimal Code Changes**: Reuses 90% of original FlowSE code
- ✅ **Flexible Conditioning**: Supports both utterance and frame-level emotion features
- ✅ **CFG Support**: Built-in classifier-free guidance for emotion strength control
- ✅ **Easy Integration**: Compatible with existing Vevo and emotion2vec pipelines
- ✅ **Backward Compatible**: Can still work without emotion features (dummy mode)

## 🛠️ Development Notes

### Emotion2VecConditioner Features
- **Utterance-level**: Global emotion conditioning (broadcasted to all frames)
- **Frame-level**: Fine-grained temporal emotion dynamics
- **Fusion modes**: Configurable combination of utterance and frame features
- **Interpolation**: Automatic alignment of emotion frames to mel frames

### Classifier-Free Guidance
- Training: Random dropout of emotion conditioning
- Inference: Controllable emotion strength via CFG scale
- Supports both conditional and unconditional generation

### Dummy Data Support
- Graceful fallback when emotion features are missing
- Useful for testing and development
- Maintains compatibility with existing workflows

## 📝 TODO for Production Use

1. **Integrate Real Emotion2Vec Model**
   - Replace dummy feature extraction in `EmotionProcessor`
   - Add proper emotion2vec model loading and inference

2. **Data Preprocessing Scripts**
   - Create scripts to extract Vevo mel spectrograms
   - Add emotion2vec feature extraction pipeline
   - Generate proper .scp and .json files

3. **Training Data Preparation**
   - Collect paired (source_audio, vevo_mel) data
   - Extract emotion2vec features for all utterances
   - Create train/validation splits

4. **Hyperparameter Tuning**
   - Optimize emotion conditioning dimensions
   - Tune CFG scales for emotion strength control
   - Adjust fusion modes based on data characteristics

This implementation provides a solid foundation for emotion-conditioned speech enhancement using the FlowSE architecture with emotion2vec features.
