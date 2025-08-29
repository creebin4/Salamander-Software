@echo off
REM --- Config ---
set CONTAINER_NAME=salamander-software
set PROJECT_DIR=C:\Users\webst\OneDrive\Documents\GitHub\Salamander-Software

REM --- Clean up any existing container ---
docker rm -f %CONTAINER_NAME% 2>nul

REM --- Ask for input and output folders ---
echo Salamander Software - Image Processing Pipeline
echo ================================================
echo.
set /p INPUT_FOLDER="Enter the full path to your input folder (containing images): "
set /p OUTPUT_FOLDER="Enter the full path to your output folder: "

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
echo.

REM --- Run container with GPU support and folder mounts ---
docker run -it ^
  --name %CONTAINER_NAME% ^
  --gpus all ^
  --runtime=nvidia ^
  -v "%PROJECT_DIR%:/app" ^
  -v "%INPUT_FOLDER%:/app/input" ^
  -v "%OUTPUT_FOLDER%:/app/output" ^
  -e INPUT_FOLDER="%INPUT_FOLDER%" ^
  -e OUTPUT_FOLDER="%OUTPUT_FOLDER%" ^
  -w /app ^
  yolo-ibeis-gpu ^
  sh -c "uv run python ./pipeline_pose_then_seg.py --folder /app/input --output /app/output"

REM --- Keep the window open after exit ---
echo.
echo Container "%CONTAINER_NAME%" has stopped.
echo Processed images should be in: %OUTPUT_FOLDER%
pause
