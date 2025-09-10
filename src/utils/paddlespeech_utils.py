#!/usr/bin/env python3
"""
PaddleSpeech工具集
提供稳定的PaddleSpeech TTS调用接口
"""

import os
import sys
import subprocess
import tempfile
import numpy as np
import soundfile as sf
from typing import Optional, Union
import time
import threading
import queue

class PaddleSpeechTTS:
    """
    PaddleSpeech TTS工具类
    
    特点：
    1. 使用超时机制防止卡死
    2. 使用独立进程确保稳定性
    3. 添加重试逻辑提高成功率
    4. 简化调用接口
    """
    
    def __init__(self, timeout: int = 30):
        self.timeout = timeout
        self.target_sr = 22050
    
    def _create_tts_script(self, text: str, output_file: str) -> str:
        """创建TTS脚本"""
        # 处理Windows路径转义
        escaped_output_file = output_file.replace('\\', '\\\\')
        
        script_content = f'''#!/usr/bin/env python3
import sys
import os
import numpy as np
import soundfile as sf

def main():
    try:
        # 导入PaddleSpeech
        from paddlespeech.cli.tts import TTSExecutor
        
        # 创建TTS执行器
        tts = TTSExecutor()
        
        # 合成音频
        text = "{text}"
        wav = tts(text=text)
        
        if isinstance(wav, np.ndarray) and len(wav) > 0:
            # 保存音频
            output_file = "{escaped_output_file}"
            sf.write(output_file, wav, 22050)
            print("SUCCESS")
            return True
        else:
            print("ERROR: Invalid audio output")
            return False
            
    except Exception as e:
        print(f"ERROR: {{e}}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    main()
'''
        return script_content
    
    def _run_with_timeout(self, script_path: str) -> tuple[bool, str, str]:
        """带超时的脚本执行"""
        try:
            process = subprocess.Popen(
                [sys.executable, script_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=os.getcwd()
            )
            
            stdout, stderr = process.communicate(timeout=self.timeout)
            return process.returncode == 0, stdout, stderr
            
        except subprocess.TimeoutExpired:
            process.kill()
            return False, "", "Process timed out"
        except Exception as e:
            return False, "", str(e)
    
    def synthesize(self, text: str, max_retries: int = 2) -> Optional[np.ndarray]:
        """合成语音"""
        print(f"🎤 PaddleSpeech合成: '{text}'")
        
        for attempt in range(max_retries + 1):
            if attempt > 0:
                print(f"   🔄 重试 {attempt}/{max_retries}")
            
            # 创建临时文件
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
                temp_audio_path = temp_audio.name
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as temp_script:
                script_content = self._create_tts_script(text, temp_audio_path)
                temp_script.write(script_content)
                temp_script_path = temp_script.name
            
            try:
                # 执行脚本
                success, stdout, stderr = self._run_with_timeout(temp_script_path)
                
                if success and "SUCCESS" in stdout and os.path.exists(temp_audio_path):
                    # 加载音频
                    try:
                        audio, sr = sf.read(temp_audio_path)
                        
                        # 重采样到目标采样率
                        if sr != self.target_sr:
                            import librosa
                            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.target_sr)
                        
                        print(f"   ✅ 合成成功: {len(audio)/self.target_sr:.2f}秒")
                        return audio
                        
                    except Exception as e:
                        print(f"   ❌ 音频加载失败: {e}")
                        
                else:
                    print(f"   ❌ 合成失败: {stderr}")
                    
            finally:
                # 清理临时文件
                for temp_file in [temp_audio_path, temp_script_path]:
                    if os.path.exists(temp_file):
                        try:
                            os.unlink(temp_file)
                        except:
                            pass
        
        print(f"   ❌ 所有重试都失败了")
        return None

# 全局实例
_paddlespeech_tts = None

def get_paddlespeech_tts() -> PaddleSpeechTTS:
    """获取PaddleSpeech TTS实例"""
    global _paddlespeech_tts
    if _paddlespeech_tts is None:
        _paddlespeech_tts = PaddleSpeechTTS()
    return _paddlespeech_tts

def text_to_speech(text: str) -> Optional[np.ndarray]:
    """
    简单的文本转语音接口
    
    Args:
        text: 输入文本
    
    Returns:
        音频数组或None（如果失败）
    """
    tts = get_paddlespeech_tts()
    return tts.synthesize(text)

def create_paddlespeech_executor():
    """
    创建PaddleSpeech执行器（兼容性接口）
    
    Returns:
        (executor, success)
    """
    try:
        tts = get_paddlespeech_tts()
        
        # 创建一个兼容的执行器包装
        class ExecutorWrapper:
            def __init__(self, tts_instance):
                self.tts = tts_instance
            
            def __call__(self, text: str, output: str = None):
                """执行TTS"""
                audio = self.tts.synthesize(text)
                if audio is not None and output is not None:
                    # 保存到指定路径
                    sf.write(output, audio, 22050)
                    return output
                return audio
        
        executor = ExecutorWrapper(tts)
        return executor, True
        
    except Exception as e:
        print(f"创建PaddleSpeech执行器失败: {e}")
        return None, False
