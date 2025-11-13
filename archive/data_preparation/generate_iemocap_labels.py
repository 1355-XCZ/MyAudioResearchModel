"""解析IEMOCAP的EmoEvaluation生成标签（参照data_preparation_iemocap.py）"""

import re
from pathlib import Path

iemocap_root = Path("/data/gpfs/projects/punim2341/haoguangzhou/voice/MyAudioResearchModel/src/Amphion/models/vc/emotion_bottleneck_rate/iemocap/data/IEMOCAP_full_release")
output_root = Path("/data/gpfs/projects/punim2341/haoguangzhou/data/evaluation_features/IEMOCAP")

# 只保留4个核心类别：angry, happy, sad, neutral（原始neu）
emotion_mapping = {
    'ang': 'angry',
    'hap': 'happy',
    'exc': 'happy',  # excited -> happy
    'sad': 'sad',
    'neu': 'neutral',  # 只用原始neutral
    # 排除其他所有类别（保证实验严谨）
    'fru': 'SKIP',  # frustrated不合并，排除
    'sur': 'SKIP',  # surprised排除
    'fea': 'SKIP',  # fearful排除
    'dis': 'SKIP',  # disgusted排除
    'oth': 'SKIP',  # other排除
    'xxx': 'SKIP'   # unknown排除
}

emotion_counts = {}
total_parsed = 0

for session in range(1, 6):
    eval_dir = iemocap_root / f"Session{session}" / "dialog" / "EmoEvaluation"
    if not eval_dir.exists():
        continue
    
    for eval_file in eval_dir.glob("*.txt"):
        with open(eval_file, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('%'):
                    continue
                
                # 格式: [start - end]	utterance_id	emotion	[v, a, d]
                if line.startswith('['):
                    match = re.match(
                        r'\[(\d+\.\d+)\s*-\s*(\d+\.\d+)\]\s+(\S+)\s+(\w+)\s+\[([^\]]+)\]',
                        line
                    )
                    if match:
                        utterance_id = match.group(3)
                        raw_emotion = match.group(4)
                        
                        # 映射情感
                        emotion = emotion_mapping.get(raw_emotion, raw_emotion)
                        
                        # 跳过样本数<50的类别
                        if emotion == 'SKIP':
                            continue
                        
                        # 查找对应的特征文件
                        # utterance_id格式: Ses01F_impro01_F000
                        dialog_name = re.sub(r'_[MF]\d+$', '', utterance_id)
                        
                        feat_file = output_root / f"Session{session}" / dialog_name / (utterance_id + '_ev2_frame.npy')
                        
                        if feat_file.exists():
                            label_file = feat_file.parent / (utterance_id + '_emotion.txt')
                            with open(label_file, 'w') as f:
                                f.write(emotion)
                            
                            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
                            total_parsed += 1

print(f"✅ IEMOCAP标签生成完成")
print(f"总数: {total_parsed}")
print(f"情感分布: {emotion_counts}")

