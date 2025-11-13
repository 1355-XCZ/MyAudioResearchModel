"""重新生成RAVDESS标签（修复标签错误）"""

from pathlib import Path

ravdess_root = Path("/data/gpfs/projects/punim2341/haoguangzhou/voice/MyAudioResearchModel/src/Amphion/models/vc/emotion_bottleneck_rate/ravdess/data/RAVDESS/audio_speech_actors_01-24")
output_root = Path("/data/gpfs/projects/punim2341/haoguangzhou/data/evaluation_features/RAVDESS")

all_wavs = list(ravdess_root.glob("*/*.wav"))
# 只保留speech（第2位=01）
wav_files = [w for w in all_wavs if w.stem.split('-')[1] == '01']

print(f"处理{len(wav_files)}个RAVDESS speech文件...")

# 参照emotion_bottleneck_rate: calm映射到neutral
emo_map = {
    1:'neutral', 
    2:'neutral',  # calm -> neutral（合并）
    3:'happy', 4:'sad', 
    5:'angry', 6:'fearful', 
    7:'disgust',  # 注意：disgust不是disgusted
    8:'surprised'
}

emotion_counts = {}

for wav in wav_files:
    parts = wav.stem.split('-')
    # modality-vocal-emotion-intensity-statement-rep-actor
    # parts[2]是emotion编码
    emotion_code = int(parts[2])
    emotion = emo_map.get(emotion_code, 'unknown')
    
    emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
    
    label_file = output_root / (wav.stem + '_emotion.txt')
    with open(label_file, 'w') as f:
        f.write(emotion)

print("✅ RAVDESS标签重新生成完成")
print(f"情感分布: {emotion_counts}")
print(f"保存路径: {output_root}")

