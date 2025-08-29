@echo off
REM --- Config ---
set CONTAINER_NAME=salamander-software
set CONFIG_FILE=%~dp0config.txt

REM --- GPU Troubleshooting Helper ---
if "%1"=="--gpu-check" goto gpu_check
if "%1"=="--help" goto help
goto main

:gpu_check
echo ========================================
echo    GPU TROUBLESHOOTING CHECK
echo ========================================
echo.

echo 1. Checking NVIDIA drivers...
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>nul
if %errorlevel% neq 0 (
    echo    ERROR: NVIDIA drivers not found or not working
    echo    Please install/update NVIDIA drivers
) else (
    echo    ✓ NVIDIA drivers detected
)

echo.
echo 2. Checking Docker GPU support...
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi --query-gpu=name --format=csv,noheader 2>nul >nul
if %errorlevel% neq 0 (
    echo    ERROR: Docker GPU support not working
    echo    Please install NVIDIA Docker support:
    echo    https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html
) else (
    echo    ✓ Docker GPU support detected
)

echo.
echo 3. Checking our container image...
docker run --rm --gpus all yolo-ibeis-gpu nvidia-smi --query-gpu=name --format=csv,noheader 2>nul >nul
if %errorlevel% neq 0 (
    echo    ERROR: Container GPU support not working
    echo    The container image may not have GPU libraries
) else (
    echo    ✓ Container GPU support working
)

echo.
echo For more help, visit:
echo https://docs.docker.com/desktop/gpu/
pause
exit /b 0

:help
echo ========================================
echo    SALAMANDER SOFTWARE HELPER
echo ========================================
echo.
echo Usage: run.bat [options]
echo.
echo Options:
echo   (no options)  - Run the image processing pipeline
echo   --gpu-check   - Check GPU configuration
echo   --help        - Show this help
echo.
echo The script will:
echo 1. Ask for project directory (first time only)
echo 2. Ask for input folder (every time)
echo 3. Ask for output folder (every time)
echo 4. Run Docker container with GPU acceleration
echo.
pause
exit /b 0

:main

REM --- Check for saved project directory ---
if exist "%CONFIG_FILE%" (
    REM Read project directory from config file
    for /f "tokens=*" %%a in (%CONFIG_FILE%) do set PROJECT_DIR=%%a
    echo Using saved project directory: %PROJECT_DIR%
) else (
    REM Ask for project directory on first run
    echo Salamander Software - First Time Setup
    echo ======================================
    echo.
    echo Please enter the full path to your project directory
    echo (where the Salamander-Software code is located)
    echo.
    set /p PROJECT_DIR="Project directory: "

    REM Validate project directory
    if not exist "%PROJECT_DIR%" (
        echo Error: Project directory does not exist: %PROJECT_DIR%
        pause
        exit /b 1
    )

    REM Save to config file
    echo %PROJECT_DIR% > "%CONFIG_FILE%"
    echo.
    echo Project directory saved to: %CONFIG_FILE%
    echo You won't need to enter it again on future runs.
    echo.
)

REM --- Clean up any existing container ---
docker rm -f %CONTAINER_NAME% 2>nul

REM --- Ask for input and output folders ---
echo Salamander Software - Image Processing Pipeline
echo ================================================
echo Project Directory: %PROJECT_DIR%
echo.
set /p INPUT_FOLDER="Enter the full path to your input folder (containing images): "
set /p OUTPUT_FOLDER="Enter the full path to your output folder: "

REM --- Optional: Reset project directory ---
echo.
set /p RESET_PROJECT="Change project directory? (y/N): "
if /i "%RESET_PROJECT%"=="y" (
    del "%CONFIG_FILE%" 2>nul
    echo Config file deleted. Please restart the script to set a new project directory.
    pause
    exit /b 0
)

REM --- Validate folders exist ---
if not exist "%INPUT_FOLDER%" (
    echo Error: Input folder does not exist: %INPUT_FOLDER%
    pause
    exit /b 1
)

if not exist "%OUTPUT_FOLDER%" (
    echo Creating output folder: %OUTPUT_FOLDER%
    mkdir "%OUTPUT_FOLDER%" 2>nul
)

echo.
echo Starting Docker container with:
echo   Input folder: %INPUT_FOLDER%
echo   Output folder: %OUTPUT_FOLDER%
echo   GPU support: %GPU_FLAGS%
echo.

REM --- GPU Configuration ---
if "%GPU_FLAGS%"=="" (
    echo Checking GPU availability...
    docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi --query-gpu=name --format=csv,noheader,nounits 2>nul >nul
    if %errorlevel% neq 0 (
        echo Warning: GPU not detected or NVIDIA Docker not properly configured.
        echo Falling back to CPU mode.
        echo.
        set GPU_FLAGS=--device /dev/dri
    ) else (
        echo GPU detected! Using GPU acceleration.
        echo.
        set GPU_FLAGS=--gpus all
    )
)

docker run -it ^
  --name %CONTAINER_NAME% ^
  %GPU_FLAGS% ^
  -v "%PROJECT_DIR%:/app" ^
  -v "%INPUT_FOLDER%:/app/input" ^
  -v "%OUTPUT_FOLDER%:/app/output" ^
  -e INPUT_FOLDER="%INPUT_FOLDER%" ^
  -e OUTPUT_FOLDER="%OUTPUT_FOLDER%" ^
  -e NVIDIA_VISIBLE_DEVICES=all ^
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility ^
  -w /app ^
  yolo-ibeis-gpu ^
  sh -c "echo 'Container GPU info:' && nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null || echo 'No GPU detected inside container' && echo '' && uv run python ./pipeline_pose_then_seg.py --folder /app/input --output /app/output"

REM --- Keep the window open after exit ---
echo.
echo Container "%CONTAINER_NAME%" has stopped.
echo Processed images should be in: %OUTPUT_FOLDER%
pause
