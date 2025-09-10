#!/usr/bin/env python3
"""
PaddleSpeech包装器 - 使用独立进程完全避免导入冲突
"""
import os
import sys
import subprocess
import json
import tempfile


def create_paddlespeech_executor():
    """
    使用独立进程创建PaddleSpeech TTSExecutor，完全避免与PyTorch的导入冲突。
    """
    print("正在使用独立进程初始化PaddleSpeech...")
    
    # 检查是否可以使用独立进程方式
    try:
        # 创建一个独立的Python脚本来测试PaddleSpeech
        script_content = """
import sys
import os
import json

# 确保不导入任何可能冲突的库
def test_paddlespeech():
    try:
        # 在完全独立的环境中测试PaddleSpeech
        import paddle
        from paddlespeech.cli.tts import TTSExecutor
        
        # 创建执行器
        tts_executor = TTSExecutor()
        
        result = {
            "success": True,
            "paddle_version": paddle.__version__,
            "message": "PaddleSpeech在独立进程中可用"
        }
        return result
    except Exception as e:
        result = {
            "success": False,
            "error": str(e),
            "traceback": __import__('traceback').format_exc()
        }
        return result

if __name__ == "__main__":
    result = test_paddlespeech()
    print(json.dumps(result))
        """
        
        # 获取当前conda环境的Python解释器路径
        python_executable = sys.executable
        
        # 运行独立进程测试
        process = subprocess.run(
            [python_executable, "-c", script_content],
            capture_output=True,
            text=True,
            timeout=30,  # 30秒超时
            env=os.environ.copy()
        )
        
        if process.returncode == 0:
            try:
                output = json.loads(process.stdout)
                if output["success"]:
                    print(f"✅ PaddleSpeech在独立进程中可用 (PaddlePaddle {output['paddle_version']})")
                    # 返回一个特殊的执行器包装器
                    return PaddleSpeechProcessWrapper(), True
                else:
                    print(f"❌ PaddleSpeech在独立进程中不可用: {output['error']}")
                    return None, False
            except json.JSONDecodeError as e:
                print(f"❌ 独立进程输出解析失败: {e}")
                print(f"Raw stdout: {process.stdout}")
                print(f"Raw stderr: {process.stderr}")
                return None, False
        else:
            print(f"❌ 独立进程执行失败 (返回码: {process.returncode})")
            print(f"Stderr: {process.stderr}")
            return None, False
            
    except subprocess.TimeoutExpired:
        print("❌ PaddleSpeech初始化超时")
        return None, False
    except Exception as e:
        print(f"❌ 独立进程测试失败: {e}")
        return None, False


class PaddleSpeechProcessWrapper:
    """
    PaddleSpeech的进程包装器，通过独立进程调用PaddleSpeech以避免导入冲突
    """
    
    def __init__(self):
        self.python_executable = sys.executable
        
    def __call__(self, text, output=None):
        """
        兼容原始TTSExecutor的调用接口
        """
        return self.synthesize(text, output)
        
    def synthesize(self, text, output_path=None):
        """
        使用独立进程进行语音合成
        """
        try:
            # 转义文本中的引号
            escaped_text = text.replace('"', '\\"').replace("'", "\\'")
            # 处理Windows路径的反斜杠转义问题
            if output_path:
                escaped_path = output_path.replace('\\', '\\\\')
                output_arg = f'r"{escaped_path}"'
            else:
                output_arg = 'None'
            
            # 创建TTS脚本
            script_content = f'''
import sys
import os
import json
import numpy as np

def tts_synthesis():
    try:
        from paddlespeech.cli.tts import TTSExecutor
        
        # 创建TTS执行器
        tts_executor = TTSExecutor()
        
        # 进行语音合成
        text = "{escaped_text}"
        output_path = {output_arg}
        
        # 合成音频
        wav = tts_executor(text=text, output=output_path)
        
        if isinstance(wav, np.ndarray):
            # 如果返回numpy数组，保存为临时文件
            import tempfile
            import soundfile as sf
            
            if output_path is None:
                temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                output_path = temp_file.name
                temp_file.close()
            
            sf.write(output_path, wav, 22050)
            
            result = {{
                "success": True,
                "output_path": output_path,
                "audio_shape": list(wav.shape),
                "sample_rate": 22050,
                "audio_data": wav.tolist() if wav.size < 10000 else "too_large"
            }}
        else:
            result = {{
                "success": True,
                "output_path": output_path,
                "message": "Audio saved successfully"
            }}
        
        return result
        
    except Exception as e:
        result = {{
            "success": False,
            "error": str(e),
            "traceback": __import__('traceback').format_exc()
        }}
        return result

if __name__ == "__main__":
    result = tts_synthesis()
    print(json.dumps(result))
            '''
            
            # 运行TTS合成
            process = subprocess.run(
                [self.python_executable, "-c", script_content],
                capture_output=True,
                text=True,
                timeout=120,  # 120秒超时，首次运行需要下载模型
                env=os.environ.copy()
            )
            
            if process.returncode == 0:
                try:
                    output = json.loads(process.stdout)
                    if output["success"]:
                        print(f"✅ PaddleSpeech合成成功: {output.get('audio_shape', 'unknown shape')}")
                        if output_path is None and 'audio_data' in output and output['audio_data'] != "too_large":
                            # 返回音频数据
                            import numpy as np
                            return np.array(output['audio_data'], dtype=np.float32)
                        else:
                            return output.get('output_path', output_path)
                    else:
                        print(f"❌ TTS合成失败: {output['error']}")
                        return None
                except json.JSONDecodeError as e:
                    print(f"❌ TTS输出解析失败: {e}")
                    print(f"Stdout: {process.stdout}")
                    print(f"Stderr: {process.stderr}")
                    return None
            else:
                print(f"❌ TTS进程执行失败 (返回码: {process.returncode})")
                print(f"Stderr: {process.stderr}")
                return None
                
        except subprocess.TimeoutExpired:
            print("❌ TTS合成超时")
            return None
        except Exception as e:
            print(f"❌ TTS合成过程出错: {e}")
            return None


def test_paddlespeech_executor(executor):
    """
    测试PaddleSpeech执行器功能
    """
    try:
        # 简单功能测试
        result = executor(text="测试", output=None)
        if result is not None:
            print("✅ PaddleSpeech功能测试成功")
            return True
        else:
            print("⚠️ PaddleSpeech功能测试失败: 返回None")
            return False
    except Exception as e:
        print(f"⚠️ PaddleSpeech功能测试失败: {e}")
        return False


if __name__ == "__main__":
    print("🧪 PaddleSpeech独立进程包装器测试")
    print("=" * 50)
    
    executor, success = create_paddlespeech_executor()
    
    if success and executor:
        print("测试PaddleSpeech功能...")
        test_paddlespeech_executor(executor)
    else:
        print("❌ 无法创建PaddleSpeech执行器")