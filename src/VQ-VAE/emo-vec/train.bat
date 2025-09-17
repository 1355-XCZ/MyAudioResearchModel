@echo off
REM Emotion2Vec VQ-VAE 训练脚本 (Windows)

REM 设置环境变量
set PYTHONPATH=%PYTHONPATH%;%CD%;%CD%\..\vevo-code

REM 默认参数
set DATA_ROOT=.\data
set EXP_DIR=.\experiments\emotion2vec_vqvae
set CONFIG=config\emotion2vec_vqvae_config.json
set BATCH_SIZE=8
set LEARNING_RATE=1e-4
set MAX_STEPS=50000
set USE_DUMMY=--use_dummy

REM 解析命令行参数
:parse_args
if "%~1"=="" goto start_training
if "%~1"=="--data_root" (
    set DATA_ROOT=%~2
    shift
    shift
    goto parse_args
)
if "%~1"=="--exp_dir" (
    set EXP_DIR=%~2
    shift
    shift
    goto parse_args
)
if "%~1"=="--config" (
    set CONFIG=%~2
    shift
    shift
    goto parse_args
)
if "%~1"=="--batch_size" (
    set BATCH_SIZE=%~2
    shift
    shift
    goto parse_args
)
if "%~1"=="--learning_rate" (
    set LEARNING_RATE=%~2
    shift
    shift
    goto parse_args
)
if "%~1"=="--max_steps" (
    set MAX_STEPS=%~2
    shift
    shift
    goto parse_args
)
if "%~1"=="--no_dummy" (
    set USE_DUMMY=
    shift
    goto parse_args
)
if "%~1"=="--help" (
    echo Usage: %0 [options]
    echo Options:
    echo   --data_root PATH        Path to training data (default: .\data)
    echo   --exp_dir PATH          Experiment directory (default: .\experiments\emotion2vec_vqvae)
    echo   --config PATH           Config file path (default: config\emotion2vec_vqvae_config.json)
    echo   --batch_size INT        Batch size (default: 8)
    echo   --learning_rate FLOAT   Learning rate (default: 1e-4)
    echo   --max_steps INT         Max training steps (default: 50000)
    echo   --no_dummy              Use real emotion2vec features instead of dummy
    echo   --help                  Show this help message
    exit /b 0
)
echo Unknown option: %~1
exit /b 1

:start_training
REM 创建必要的目录
if not exist "%EXP_DIR%" mkdir "%EXP_DIR%"
if not exist "%DATA_ROOT%" mkdir "%DATA_ROOT%"

echo Starting Emotion2Vec VQ-VAE Training...
echo Data root: %DATA_ROOT%
echo Experiment dir: %EXP_DIR%
echo Config: %CONFIG%
echo Batch size: %BATCH_SIZE%
echo Learning rate: %LEARNING_RATE%
echo Max steps: %MAX_STEPS%
if "%USE_DUMMY%"=="--use_dummy" (
    echo Use dummy features: Yes
) else (
    echo Use dummy features: No
)

REM 启动训练
python train_emotion2vec_vqvae.py --config "%CONFIG%" --data_root "%DATA_ROOT%" --exp_dir "%EXP_DIR%" --batch_size %BATCH_SIZE% --learning_rate %LEARNING_RATE% --max_steps %MAX_STEPS% %USE_DUMMY%

echo Training completed!
pause
