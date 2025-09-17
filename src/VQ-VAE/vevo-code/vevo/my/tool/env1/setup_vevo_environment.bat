@echo off
REM VEVO项目环境配置脚本 (Windows版本)
REM 使用方法: setup_vevo_environment.bat

echo === VEVO项目环境配置开始 ===

REM 检查Python版本
python --version
if errorlevel 1 (
    echo 错误: Python未安装或未添加到PATH
    pause
    exit /b 1
)

REM 询问是否创建虚拟环境
set /p create_venv="是否创建新的虚拟环境? (y/n): "
if /i "%create_venv%"=="y" (
    echo 创建虚拟环境 vevo_env...
    python -m venv vevo_env
    call vevo_env\Scripts\activate.bat
    echo 虚拟环境已激活
)

REM 更新pip
echo 更新pip...
python -m pip install --upgrade pip setuptools wheel

echo === 安装核心深度学习框架 ===
pip install torch==2.0.1 torchaudio==2.0.2 torchvision==0.15.2

echo === 安装Transformers和相关库 ===
pip install transformers==4.41.2 accelerate==0.24.1 diffusers safetensors

echo === 安装数值计算库 ===
pip install numpy==1.26.0 scipy==1.12.0 pandas matplotlib scikit-learn

echo === 安装音频处理库 ===
pip install librosa soundfile audiomentations pyworld praat-parselmouth
pip install diffsptk pysptk resampy soxr

echo === 安装编解码器 ===
pip install encodec
pip install vocos speechtokenizer descript-audio-codec

echo === 安装语音识别库 ===
pip install openai-whisper fairseq
pip install git+https://github.com/m-bain/whisperx.git

echo === 安装文本处理库 ===
pip install phonemizer==3.2.1 g2p_en pypinyin==0.48.0
pip install jieba cn2an unidecode pyopenjtalk pykakasi

echo === 安装配置和工具库 ===
pip install omegaconf hydra-core ruamel.yaml json5 easydict
pip install datasets huggingface_hub

echo === 安装评估指标库 ===
pip install torchmetrics frechet_audio_distance
pip install https://github.com/vBaiCai/python-pesq/archive/master.zip
pip install pystoi pymcd mir_eval jiwer

echo === 安装训练工具 ===
pip install pytorch-lightning tensorboard tensorboardX wandb

echo === 安装其他依赖 ===
pip install tqdm loguru einops vector-quantize-pytorch
pip install gradio fastapi uvicorn
pip install black ruff Cython
pip install tgt typeguard humanfriendly munch
pip install nnAudio ptwt ffmpeg-python==0.2.0
pip install PyYAML ipython

echo === 安装特殊依赖 ===
pip install git+https://github.com/lhotse-speech/lhotse

echo === 编译Cython模块 ===
cd ..\..\..\..\..\modules\monotonic_align
python setup.py build_ext --inplace
cd ..\..\models\vc\vevo\my\tool\env1

echo === 环境配置完成! ===
echo.
echo 请确认以下事项:
echo 1. espeak-ng 已正确安装 (需要手动下载安装)
echo 2. ffmpeg 已正确安装 (需要手动下载安装或使用conda)
echo 3. CUDA环境配置正确 (如果使用GPU)
echo.
echo 测试安装:
python -c "import torch; print('PyTorch版本:', torch.__version__)"
python -c "import librosa; print('Librosa版本:', librosa.__version__)"
python -c "import transformers; print('Transformers版本:', transformers.__version__)"
echo.
echo 如果遇到CUDA相关问题，可能需要运行:
echo pip uninstall nvidia-cublas-cu11

pause
