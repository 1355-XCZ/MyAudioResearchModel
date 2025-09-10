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
        print("开始初始化PaddleSpeech TTS...")
        
        # 导入PaddleSpeech
        from paddlespeech.cli.tts import TTSExecutor
        
        # 创建TTS执行器
        tts = TTSExecutor()
        print("TTS执行器创建成功")
        
        # 合成音频
        text = "{text}"
        print(f"开始合成文本: {{text}}")
        
        wav = tts(text=text)
        print(f"TTS合成完成，音频类型: {{type(wav)}}")
        
        # 检查TTS返回类型
        if isinstance(wav, str) and os.path.exists(wav):
            print(f"TTS返回音频文件路径: {{wav}}")
            try:
                # 从文件加载音频
                audio_data, sr = sf.read(wav)
                print(f"从TTS文件加载音频: 长度={{len(audio_data)}}, 采样率={{sr}}")
                
                # 保存到目标文件
                output_file = "{escaped_output_file}"
                print(f"保存音频到: {{output_file}}")
                
                # 重采样到22050Hz如果需要
                if sr != 22050:
                    import librosa
                    audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=22050)
                    print(f"重采样到22050Hz")
                
                sf.write(output_file, audio_data, 22050)
                
                # 验证文件是否成功保存
                if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
                    print(f"音频文件保存成功，大小: {{os.path.getsize(output_file)}} 字节")
                    print("SUCCESS")
                    
                    # 清理TTS生成的临时文件
                    try:
                        if wav != output_file:  # 避免删除目标文件
                            os.remove(wav)
                            print(f"清理TTS临时文件: {{wav}}")
                    except:
                        pass
                    
                    return True
                else:
                    print("ERROR: 音频文件保存失败")
                    return False
                    
            except Exception as e:
                print(f"ERROR: 处理TTS音频文件失败: {{e}}")
                return False
                
        elif isinstance(wav, np.ndarray):
            print(f"TTS返回音频数组，长度: {{len(wav)}}")
            if len(wav) > 0:
                # 保存音频
                output_file = "{escaped_output_file}"
                print(f"保存音频到: {{output_file}}")
                sf.write(output_file, wav, 22050)
                
                # 验证文件是否成功保存
                if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
                    print(f"音频文件保存成功，大小: {{os.path.getsize(output_file)}} 字节")
                    print("SUCCESS")
                    return True
                else:
                    print("ERROR: 音频文件保存失败")
                    return False
            else:
                print("ERROR: 音频数组为空")
                return False
        else:
            print(f"ERROR: 无效的TTS输出类型: {{type(wav)}}, 内容: {{wav}}")
            return False
            
    except Exception as e:
        print(f"ERROR: {{e}}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
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
            
            # 创建临时文件（覆盖模式）
            import uuid
            unique_id = str(uuid.uuid4())[:8]
            temp_audio_path = f"temp_tts_{unique_id}.wav"
            temp_script_path = f"temp_tts_script_{unique_id}.py"
            
            # 如果文件已存在，直接覆盖
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)
            if os.path.exists(temp_script_path):
                os.remove(temp_script_path)
            
            try:
                # 创建脚本文件
                script_content = self._create_tts_script(text, temp_audio_path)
                with open(temp_script_path, 'w', encoding='utf-8') as f:
                    f.write(script_content)
                
                print(f"   ⏳ 正在合成音频...")
                
                # 添加简单的进度指示器
                import threading
                import time
                
                def progress_indicator():
                    chars = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
                    i = 0
                    while not getattr(progress_indicator, 'stop', False):
                        print(f"\r   {chars[i % len(chars)]} 合成中...", end='', flush=True)
                        i += 1
                        time.sleep(0.1)
                    print("\r" + " " * 20 + "\r", end='', flush=True)  # 清除进度指示器
                
                # 启动进度指示器
                progress_thread = threading.Thread(target=progress_indicator)
                progress_thread.daemon = True
                progress_thread.start()
                
                try:
                    # 执行脚本
                    success, stdout, stderr = self._run_with_timeout(temp_script_path)
                finally:
                    # 停止进度指示器
                    progress_indicator.stop = True
                    progress_thread.join(timeout=0.5)
                
                # 检查是否成功 - 改进判断逻辑
                audio_exists = os.path.exists(temp_audio_path)
                has_success_marker = "SUCCESS" in stdout
                
                print(f"   🔍 调试信息: success={success}, audio_exists={audio_exists}, has_success_marker={has_success_marker}")
                
                if audio_exists and os.path.getsize(temp_audio_path) > 0:
                    # 加载音频
                    try:
                        audio, sr = sf.read(temp_audio_path)
                        
                        if len(audio) == 0:
                            print(f"   ❌ 音频文件为空")
                            continue
                        
                        # 重采样到目标采样率
                        if sr != self.target_sr:
                            import librosa
                            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.target_sr)
                        
                        print(f"   ✅ 合成成功: {len(audio)/self.target_sr:.2f}秒")
                        return audio
                        
                    except Exception as e:
                        print(f"   ❌ 音频加载失败: {e}")
                        continue
                        
                else:
                    # 改进错误信息过滤
                    real_errors = []
                    if stderr:
                        error_lines = stderr.split('\n')
                        for line in error_lines:
                            line = line.strip()
                            # 过滤掉各种警告和信息
                            skip_patterns = [
                                'UserWarning', 'pkg_resources', 'distutils', 
                                'WARNING', 'INFO', 'DEBUG',
                                'Already cached', 'tokenizer config file saved',
                                'Building prefix dict', 'Loading model',
                                'Prefix dict has been built'
                            ]
                            
                            if line and not any(pattern in line for pattern in skip_patterns):
                                # 只保留真正的错误信息
                                if 'ERROR' in line.upper() or 'FAILED' in line.upper() or 'Exception' in line:
                                    real_errors.append(line)
                    
                    if real_errors:
                        print(f"   ❌ 合成失败: {'; '.join(real_errors[:3])}")  # 只显示前3个错误
                    else:
                        print(f"   ⚠️ 合成可能失败: 未生成音频文件")
                        if not success:
                            print(f"   📋 进程返回码非零: {success}")
                        if not audio_exists:
                            print(f"   📋 音频文件不存在: {temp_audio_path}")
                    
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
