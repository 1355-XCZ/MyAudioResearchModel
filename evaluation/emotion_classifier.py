"""
情感分类器 - 使用emotion2vec进行情感识别
从emotion_information_bottleneck复制的EmotionClassifierV2类
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class EmotionClassifierV2:
    """情感分类器V2 - 使用emotion2vec的9类原生输出"""
    
    def __init__(self, model_name: str, hub: str, device: str):
        self.model_name = model_name
        self.hub = hub
        self.device = device
        self.model = None
        
        # emotion2vec的9类情感（正确顺序，参考官方文档）
        # 0:angry, 1:disgusted, 2:fearful, 3:happy, 4:neutral, 5:other, 6:sad, 7:surprised, 8:unknown
        self.ev2_emotions = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'other', 'sad', 'surprised', 'unknown']
        
        # ESD的5类情感（作为参考ground truth）
        self.esd_emotions = ['angry', 'happy', 'neutral', 'sad', 'surprise']
        
        # 可选的宽松映射（用于参考对比）
        # emotion2vec的9类 -> ESD的5类（仅用于参考分析）
        self.ev2_to_esd_loose_mapping = {
            'angry': 'angry',
            'disgusted': 'angry',      # 厌恶 -> 愤怒
            'fearful': 'sad',          # 恐惧 -> 悲伤
            'happy': 'happy',
            'neutral': 'neutral',
            'sad': 'sad',
            'surprised': 'surprise',   # 注意拼写差异
            'other': None,             # 不映射
            'unknown': None            # 不映射
        }
        
        logger.info(f"初始化情感分类器V2（emotion2vec 9类原生输出）")
        logger.info(f"  模型: {model_name}")
        logger.info(f"  emotion2vec类别: {self.ev2_emotions}")
        logger.info(f"  ESD参考类别: {self.esd_emotions}")
    
    def load_model(self):
        """加载emotion2vec模型"""
        if self.model is not None:
            return
        
        try:
            from funasr import AutoModel
            
            logger.info("正在加载 emotion2vec 完整模型...")
            
            self.model = AutoModel(
                model=self.model_name,
                hub=self.hub
            )
            
            logger.info("✅ emotion2vec模型加载成功")
            logger.info("✅ 将使用9类原生情感分类")
            
            # 运行时获取官方标签顺序（健壮性检查）
            try:
                if hasattr(self.model, 'id2label'):
                    self.ev2_emotions = [self.model.id2label[i] for i in range(len(self.model.id2label))]
                    logger.info(f"✅ 从模型获取标签顺序: {self.ev2_emotions}")
                elif hasattr(self.model, 'model') and hasattr(self.model.model, 'label_list'):
                    self.ev2_emotions = list(self.model.model.label_list)
                    logger.info(f"✅ 从模型获取标签顺序: {self.ev2_emotions}")
                else:
                    logger.warning("⚠️ 无法从模型获取标签顺序，使用默认顺序")
            except Exception as e:
                logger.warning(f"⚠️ 获取标签顺序失败: {e}，使用默认顺序")
            
        except Exception as e:
            logger.error(f"❌ 模型加载失败: {e}")
            raise
    
    def classify_from_audio(self, audio_path: str, esd_ground_truth: Optional[str] = None) -> Dict:
        """
        从音频文件分类，返回emotion2vec的9类预测
        
        Args:
            audio_path: 音频文件路径
            esd_ground_truth: ESD数据集的真实标签（可选，用于参考）
        
        Returns:
            {
                'ev2_predicted': str,           # emotion2vec预测的情感（9类之一）
                'ev2_confidence': float,        # 预测置信度
                'ev2_probs': np.ndarray (9,),  # 9类的概率分布
                'esd_ground_truth': str,       # ESD的真实标签（如果提供）
                'esd_mapped': str,             # 通过宽松映射得到的ESD类别
                'is_consistent': bool          # 映射后是否与ESD ground truth一致
            }
        """
        if self.model is None:
            self.load_model()
        
        try:
            # 使用emotion2vec进行分类
            result = self.model.generate(
                audio_path,
                output_dir=None,
                granularity="utterance",
                extract_embedding=False
            )
            
            if result and len(result) > 0:
                # 获取emotion2vec的原始预测
                ev2_label_raw = result[0].get('labels', result[0].get('label', 'unknown'))
                # 处理labels可能是list的情况
                if isinstance(ev2_label_raw, list):
                    ev2_label_full = ev2_label_raw[0] if len(ev2_label_raw) > 0 else 'unknown'
                else:
                    ev2_label_full = ev2_label_raw
                
                # 处理中英文混合标签，提取英文部分（如 "生气/angry" -> "angry"）
                if '/' in str(ev2_label_full):
                    ev2_label = str(ev2_label_full).split('/')[-1].strip()
                else:
                    ev2_label = str(ev2_label_full)
                
                scores = result[0].get('scores', None)
                
                # 如果scores是单个值，构造概率分布
                if scores is not None and not isinstance(scores, (list, np.ndarray)):
                    # 假设这是预测类别的置信度
                    ev2_confidence = float(scores)
                    ev2_probs = np.zeros(len(self.ev2_emotions))
                    try:
                        pred_idx = self.ev2_emotions.index(ev2_label)
                        ev2_probs[pred_idx] = ev2_confidence
                        # 剩余概率均分
                        remaining = (1.0 - ev2_confidence) / (len(self.ev2_emotions) - 1)
                        for i in range(len(self.ev2_emotions)):
                            if i != pred_idx:
                                ev2_probs[i] = remaining
                    except ValueError:
                        # 如果标签不在列表中，使用均匀分布
                        ev2_probs = np.ones(len(self.ev2_emotions)) / len(self.ev2_emotions)
                        ev2_confidence = 1.0 / len(self.ev2_emotions)
                elif scores is not None:
                    # scores已经是概率分布
                    ev2_probs = np.array(scores)
                    ev2_confidence = ev2_probs.max()
                    # 根据概率分布重新确定标签（修复bug：不使用模型返回的labels）
                    pred_idx = int(ev2_probs.argmax())
                    if pred_idx < len(self.ev2_emotions):
                        ev2_label = self.ev2_emotions[pred_idx]
                else:
                    # 没有scores，使用默认高置信度
                    ev2_confidence = 0.9
                    ev2_probs = np.zeros(len(self.ev2_emotions))
                    try:
                        pred_idx = self.ev2_emotions.index(ev2_label)
                        ev2_probs[pred_idx] = ev2_confidence
                        remaining = (1.0 - ev2_confidence) / (len(self.ev2_emotions) - 1)
                        for i in range(len(self.ev2_emotions)):
                            if i != pred_idx:
                                ev2_probs[i] = remaining
                    except ValueError:
                        ev2_probs = np.ones(len(self.ev2_emotions)) / len(self.ev2_emotions)
                        ev2_confidence = 1.0 / len(self.ev2_emotions)
                
                # 通过宽松映射转换到ESD类别（仅用于参考）
                esd_mapped = self.ev2_to_esd_loose_mapping.get(ev2_label, None)
                
                # 检查一致性
                is_consistent = False
                if esd_ground_truth and esd_mapped:
                    is_consistent = (esd_mapped == esd_ground_truth)
                
                return {
                    'ev2_predicted': ev2_label,
                    'ev2_confidence': float(ev2_confidence),
                    'ev2_probs': ev2_probs,
                    'ev2_all_labels': self.ev2_emotions,
                    'esd_ground_truth': esd_ground_truth,
                    'esd_mapped': esd_mapped,
                    'is_consistent': is_consistent
                }
            else:
                raise ValueError("分类失败：模型返回空结果")
                
        except Exception as e:
            logger.error(f"分类失败 ({audio_path}): {e}")
            raise
    
    def classify_from_features(self, features: np.ndarray, audio_path: Optional[str] = None, 
                              esd_ground_truth: Optional[str] = None) -> Dict:
        """
        从特征进行分类（科学实验，无后备方案）
        
        如果提供audio_path，优先从音频分类；否则必须从特征分类
        """
        # 如果有audio_path，优先从音频分类
        if audio_path and os.path.exists(audio_path):
            return self.classify_from_audio(audio_path, esd_ground_truth)
        
        # 从特征分类
        if self.model is None:
            self.load_model()
        
        if isinstance(features, np.ndarray):
            features = torch.from_numpy(features).float()
        
        features = features.to(self.device)
        
        # 平均池化到 [1, 1024]
        if features.dim() == 2:  # [T, D]
            pooled_features = features.mean(dim=0, keepdim=False)  # [D]
        else:
            pooled_features = features
        
        if pooled_features.dim() == 1:
            pooled_features = pooled_features.unsqueeze(0)  # [1, D]
        
        # 直接访问emotion2vec_plus的分类头：model.model.proj
        # 根据测试结果，这是一个 Linear(1024 -> 9)
        if not (hasattr(self.model, 'model') and hasattr(self.model.model, 'proj')):
            raise RuntimeError("无法访问emotion2vec_plus分类头 model.model.proj")
        
        classifier = self.model.model.proj
        
        # 维度一致性检查
        in_features = int(classifier.in_features)
        if pooled_features.size(-1) != in_features:
            raise RuntimeError(
                f"特征维度不匹配: 得到 {pooled_features.size(-1)}, 期望 {in_features}. "
                "请确保使用与分类头匹配的 emotion2vec 变体或在提取端对齐维度。"
            )
        
        with torch.no_grad():
            # 通过分类头
            logits = classifier(pooled_features)  # [1, 9]
            
            if logits.dim() > 1:
                logits = logits.squeeze(0)  # [9]
            
            # 检查维度
            if logits.shape[0] != len(self.ev2_emotions):
                raise RuntimeError(f"分类头输出维度 {logits.shape[0]} 与期望的 {len(self.ev2_emotions)} 不匹配")
            
            # 计算概率
            probs = F.softmax(logits, dim=0)
            
            # 预测
            pred_idx = probs.argmax().item()
            ev2_confidence = probs[pred_idx].item()
            ev2_predicted = self.ev2_emotions[pred_idx]
            
            # 映射到ESD
            esd_mapped = self.ev2_to_esd_loose_mapping.get(ev2_predicted, None)
            is_consistent = False
            if esd_ground_truth and esd_mapped:
                is_consistent = (esd_mapped == esd_ground_truth)
            
            return {
                'ev2_predicted': ev2_predicted,
                'ev2_confidence': ev2_confidence,
                'ev2_probs': probs.cpu().numpy(),
                'ev2_all_labels': self.ev2_emotions,
                'esd_ground_truth': esd_ground_truth,
                'esd_mapped': esd_mapped,
                'is_consistent': is_consistent
            }
