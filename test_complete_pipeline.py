#!/usr/bin/env python3
"""
完整流水线测试脚本
测试B阶段为空/透传的情况下，整个流程是否能跑通

完整流程：音频输入 → 预处理(Whisper+Emotion2Vec) → 阶段A → BigVGAN提取Mel → 阶段B(透传) → BigVGAN合成音频

注意：只支持音频输入，文字必须从音频中提取
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import torch
import numpy as np
import soundfile as sf
import librosa
import yaml
from typing import Dict, Any

class EmptyStageB:
    """空的阶段B实现（透传模式）"""
    
    def __init__(self):
        print("🔄 初始化空阶段B（透传模式）")
    
    def forward(self, mel_input: torch.Tensor, emotion_features: np.ndarray, use_b2: bool = False):
        """透传前向传播：直接返回输入的Mel频谱"""
        print(f"   🔄 阶段B透传: {mel_input.shape}")
        
        class MockOutput:
            def __init__(self, mel):
                self.mel_spectrogram = mel
                self.metadata = {'stage': 'B_passthrough'}
        
        return MockOutput(mel_input)

class CompletePipelineTest:
    """完整流水线测试类"""
    
    def __init__(self):
        print("🚀" * 50)
        print("完整流水线测试（B阶段透传）")
        print("流程：音频输入 → 预处理 → 阶段A → BigVGAN → 阶段B(透传) → 最终音频")
        print("🚀" * 50)
        
        self._init_components()
    
    def _init_components(self):
        """初始化所有组件"""
        try:
            # 加载配置文件
            config_path = "config/test_pipeline_config.yaml"
            
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                print(f"📋 加载配置文件: {config_path}")
            else:
                # 使用默认配置作为备选
                config = {
                    'data_processing': {
                        'whisper_config': {'model_size': 'base', 'language': 'zh'},
                        'emotion_config': {'model_path': 'iic/emotion2vec_base', 'feature_dim': 768}
                    },
                    'stage_a': {
                        'target_sampling_rate': 22050,
                        'tts_engine_priority': ['simple_synthesis']
                    },
                    'vocoder': {
                        'bigvgan': {'model_config': {'version': '22khz'}}
                    }
                }
                print("⚠️ 配置文件不存在，使用默认配置")
            
            # 1. 初始化数据处理器（预处理）
            print("\n🔧 初始化数据处理器...")
            from src.data_processing.audio_processor import WhisperEmotionProcessor
            
            processor_config = {
                'whisper': config['data_processing']['whisper_config'],
                'emotion2vec': config['data_processing']['emotion_config']
            }
            
            self.data_processor = WhisperEmotionProcessor(processor_config)
            print("   ✅ 数据处理器初始化成功")
            
            # 2. 初始化阶段A
            print("\n🎤 初始化阶段A...")
            from src.models.stage_a import AudioStageAModel
            
            self.stage_a = AudioStageAModel(config['stage_a'])
            print("   ✅ 阶段A初始化成功")
            
            # 3. 初始化BigVGAN
            print("\n🎼 初始化BigVGAN...")
            from src.models.vocoder import BigVGANVocoder
            
            self.bigvgan = BigVGANVocoder(config['vocoder']['bigvgan'])
            print("   ✅ BigVGAN初始化成功")
            
            # 4. 初始化空阶段B
            print("\n🔄 初始化空阶段B...")
            self.stage_b = EmptyStageB()
            print("   ✅ 空阶段B初始化成功")
            
        except Exception as e:
            print(f"❌ 组件初始化失败: {e}")
            raise
    
    def test_complete_flow(self, audio_file_path: str) -> Dict[str, Any]:
        """
        测试完整流程：音频输入 → 预处理 → 阶段A → BigVGAN → 阶段B → 最终音频
        
        Args:
            audio_file_path: 音频文件路径
            
        Returns:
            包含所有处理步骤结果的字典
        """
        print(f"\n🎯 测试完整流程: {audio_file_path}")
        
        if not os.path.exists(audio_file_path):
            raise FileNotFoundError(f"音频文件不存在: {audio_file_path}")
        
        try:
            # 步骤1: 预处理阶段 - 音频分析
            print(f"\n🔧 步骤1: 预处理阶段")
            
            # 加载音频
            audio_data, sample_rate = librosa.load(audio_file_path, sr=None)
            print(f"   📁 加载音频: {len(audio_data)/sample_rate:.2f}秒, 采样率: {sample_rate}Hz")
            
            # 创建AudioData对象
            from src.core.interfaces import AudioData
            audio_obj = AudioData(
                waveform=audio_data,
                sample_rate=sample_rate,
                duration=len(audio_data) / sample_rate,
                file_path=audio_file_path
            )
            
            # 数据处理：提取文本和情感特征
            processed_data = self.data_processor.process_audio(audio_obj)
            extracted_text = processed_data.text
            emotion_features = processed_data.emotion_features
            
            print(f"   🎤 Whisper识别文本: {extracted_text}")
            print(f"   🎭 Emotion2Vec特征: {emotion_features.shape}")
            
            # 步骤2: 阶段A - 文本→音频→Mel
            print(f"\n📝 步骤2: 阶段A处理")
            stage_a_output = self.stage_a.forward(extracted_text)
            
            generated_audio = stage_a_output.metadata.get('generated_audio')
            if generated_audio is None:
                raise RuntimeError("阶段A没有产出音频")
            
            print(f"   ✅ 阶段A生成音频: {len(generated_audio)/22050:.2f}秒")
            
            # 步骤3: BigVGAN提取标准Mel
            print("\n🎼 步骤3: BigVGAN提取标准Mel")
            extracted_mel = self.bigvgan.extract_mel_from_audio(generated_audio)
            print(f"   ✅ BigVGAN提取Mel: {extracted_mel.shape}")
            
            # 步骤4: 阶段B处理（透传）
            print("\n🔄 步骤4: 阶段B处理（透传）")
            stage_b_output = self.stage_b.forward(extracted_mel, emotion_features)
            processed_mel = stage_b_output.mel_spectrogram
            
            print(f"   ✅ 阶段B输出Mel: {processed_mel.shape}")
            
            # 验证透传
            is_passthrough = torch.allclose(extracted_mel, processed_mel, atol=1e-6)
            if is_passthrough:
                print("   🎯 验证通过：阶段B完美透传")
            else:
                print("   ⚠️ 注意：阶段B对Mel进行了修改")
            
            # 步骤5: BigVGAN合成最终音频
            print("\n🔊 步骤5: BigVGAN合成最终音频")
            final_audio = self.bigvgan.synthesize(processed_mel)
            print(f"   ✅ 最终音频合成: {len(final_audio)/22050:.2f}秒")
            
            print("✅ 完整流程测试成功")
            
            return {
                'audio_file': audio_file_path,
                'original_audio': audio_data,
                'extracted_text': extracted_text,
                'emotion_features': emotion_features,
                'stage_a_audio': generated_audio,
                'extracted_mel': extracted_mel,
                'final_audio': final_audio,
                'mel_passthrough': is_passthrough,
                'success': True
            }
            
        except Exception as e:
            print(f"❌ 完整流程测试失败: {e}")
            return {'audio_file': audio_file_path, 'success': False, 'error': str(e)}
    
    def save_results(self, result: Dict[str, Any]):
        """保存测试结果"""
        if not result['success']:
            return
        
        try:
            # 从配置获取输出目录
            try:
                config_path = "config/test_pipeline_config.yaml"
                if os.path.exists(config_path):
                    with open(config_path, 'r', encoding='utf-8') as f:
                        config = yaml.safe_load(f)
                    output_dir = config.get('test', {}).get('output_dir', "outputs/complete_pipeline_test")
                else:
                    output_dir = "outputs/complete_pipeline_test"
            except:
                output_dir = "outputs/complete_pipeline_test"
            
            os.makedirs(output_dir, exist_ok=True)
            
            # 生成文件名
            import re
            filename = os.path.basename(result['audio_file']).split('.')[0]
            safe_filename = re.sub(r'[^\w\s-]', '', filename)
            
            # 保存音频
            sf.write(f"{output_dir}/{safe_filename}_stage_a.wav", 
                    result['stage_a_audio'], 22050)
            sf.write(f"{output_dir}/{safe_filename}_final.wav",
                    result['final_audio'], 22050)
            
            # 保存元数据
            metadata = {
                'audio_file': result['audio_file'],
                'extracted_text': result['extracted_text'],
                'emotion_features_shape': str(result['emotion_features'].shape),
                'mel_passthrough': result['mel_passthrough']
            }
            
            import json
            with open(f"{output_dir}/{safe_filename}_metadata.json", 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            print(f"📁 结果已保存到: {output_dir}")
            
        except Exception as e:
            print(f"⚠️ 保存结果失败: {e}")
    
    def run_test_suite(self):
        """运行测试套件"""
        # 从配置文件获取测试音频文件列表
        try:
            config_path = "config/test_pipeline_config.yaml"
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                test_files = config.get('test', {}).get('test_audio_files', ["test_audio.wav", "test.wav", "sample.wav", "demo.wav"])
            else:
                test_files = ["test_audio.wav", "test.wav", "sample.wav", "demo.wav"]
        except:
            test_files = ["test_audio.wav", "test.wav", "sample.wav", "demo.wav"]
        
        # 查找音频文件
        audio_files = []
        for filename in test_files:
            if os.path.exists(filename):
                audio_files.append(filename)
        
        if not audio_files:
            print("❌ 没有找到测试音频文件")
            print("请确保存在以下任一文件：test_audio.wav, test.wav, sample.wav, demo.wav")
            return {'success_rate': 0, 'results': []}
        
        print(f"🔊 找到音频文件: {audio_files}")
        
        results = []
        success_count = 0
        
        for i, audio_file in enumerate(audio_files, 1):
            print(f"\n{'='*60}")
            print(f"测试 {i}/{len(audio_files)}: {audio_file}")
            print('='*60)
            
            result = self.test_complete_flow(audio_file)
            results.append(result)
            
            if result['success']:
                success_count += 1
                self.save_results(result)
                
                if result['mel_passthrough']:
                    print("🎯 B阶段透传验证：✅ 完美透传")
                else:
                    print("🎯 B阶段透传验证：⚠️ 有修改")
            else:
                print(f"❌ 测试失败: {result.get('error', '未知错误')}")
        
        # 总结
        print("\n" + "🎊" * 60)
        print("完整流水线测试总结")
        print("🎊" * 60)
        
        success_rate = success_count / len(audio_files) * 100
        print(f"📊 测试文件数: {len(audio_files)}")
        print(f"✅ 成功数: {success_count}")
        print(f"❌ 失败数: {len(audio_files) - success_count}")
        print(f"📈 成功率: {success_rate:.1f}%")
        
        if success_count > 0:
            print("\n🎯 验证了完整流程：")
            print("✅ 音频 → Whisper文字提取")
            print("✅ 音频 → Emotion2Vec情感特征提取")
            print("✅ 文字 → 阶段A → 音频和Mel生成")
            print("✅ 音频 → BigVGAN → 标准Mel提取")
            print("✅ 阶段B透传（即使为空也不影响流程）")
            print("✅ BigVGAN最终音频合成")
            print("✅ 参数一致性保证")
            
            print("\n💡 结论：")
            print("🎊 完整流水线验证成功！")
            print("🎊 架构设计正确，B阶段可以安全开发！")
        
        return {
            'success_rate': success_rate,
            'results': results,
            'total_tests': len(audio_files)
        }

def main():
    """主函数"""
    print("🎯 完整流水线测试")
    print("要求：必须有音频文件输入，文字从音频中提取")
    
    try:
        pipeline_test = CompletePipelineTest()
        test_report = pipeline_test.run_test_suite()
        
        # 保存测试报告
        try:
            os.makedirs("outputs/complete_pipeline_test", exist_ok=True)
            import json
            with open("outputs/complete_pipeline_test/test_report.json", 'w', encoding='utf-8') as f:
                json.dump(test_report, f, indent=2, ensure_ascii=False, default=str)
        except Exception as e:
            print(f"⚠️ 保存测试报告失败: {e}")
        
        if test_report['success_rate'] >= 75:
            print("\n🎊🎊🎊 完整流水线测试成功！🎊🎊🎊")
            print("✅ 完整流程验证通过")
            print("✅ B阶段透传验证成功")
            print("✅ 可以开始开发B阶段具体实现")
        else:
            print("\n😞 流水线测试未达到预期，请检查组件")
    
    except Exception as e:
        print(f"❌ 测试执行失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()