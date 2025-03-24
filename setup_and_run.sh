#!/bin/bash

# Create virtual environment
echo "Creating Python virtual environment..."
python -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Run the inference script
echo "Running YOLO inference and ONNX conversion..."
python yolo_inference.py

# If for some reason the visualization wasn't run in the main script, run it separately
if [ ! -f "comparison_results.png" ]; then
    echo "Running visualization tool..."
    python visualize_results.py
fi

echo "Completed successfully! Check the output files:"
echo "- pytorch_results/: Directory containing PyTorch inference results"
echo "- onnx_results.jpg: Image with ONNX model detections"
echo "- comparison_results.png: Side-by-side comparison of original image and model results" 