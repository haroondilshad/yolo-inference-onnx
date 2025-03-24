@echo off
echo Creating Python virtual environment...
python -m venv venv

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo Installing dependencies...
pip install -r requirements.txt

echo Running YOLO inference and ONNX conversion...
python yolo_inference.py

REM If for some reason the visualization wasn't run in the main script, run it separately
if not exist "comparison_results.png" (
    echo Running visualization tool...
    python visualize_results.py
)

echo Completed successfully! Check the output files:
echo - pytorch_results/: Directory containing PyTorch inference results
echo - onnx_results.jpg: Image with ONNX model detections
echo - comparison_results.png: Side-by-side comparison of original image and model results
pause 