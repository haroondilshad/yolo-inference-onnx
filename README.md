# YOLO Model Inference and ONNX Conversion

This project demonstrates how to:
1. Run inference using a YOLO PyTorch model
2. Convert the YOLO model to ONNX format
3. Run inference using the converted ONNX model
4. Visualize and compare the results

## Project Structure

```
├── yolo_inference.py           # Main script for inference and conversion
├── visualize_results.py        # Script for visualizing and comparing results
├── setup_and_run.sh            # Shell script for Linux/Mac
├── setup_and_run.bat           # Batch script for Windows
├── requirements.txt            # Python dependencies
└── submission_summary.md       # Summary of the implementation
```

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Code

### Option 1: Using the provided scripts

For Linux/Mac:
```bash
./setup_and_run.sh
```

For Windows:
```bash
setup_and_run.bat
```

### Option 2: Manual execution

Run the main script directly:
```bash
python yolo_inference.py
```

This script will:
1. Run inference on `image.png` using the PyTorch YOLO model (`yolo11n.pt`)
2. Convert the PyTorch model to ONNX format (`yolo11n.onnx`)
3. Run inference on the same image using the ONNX model
4. Generate a comparison visualization

## Required Files

You need to add these files to the repository:
- `yolo11n.pt`: The YOLO model in PyTorch format
- `image.png`: The sample image for inference

## Outputs

- `pytorch_results/`: Directory containing PyTorch inference results
- `onnx_results/`: Directory containing ONNX inference results
- `comparison_results.png`: Side-by-side comparison of original image, PyTorch results, and ONNX results

## Visualization

You can also run the visualization tool separately:
```bash
python visualize_results.py
```

## Implementation Summary

For a concise summary of the implementation approach, see [submission_summary.md](submission_summary.md).

## Requirements

- Python 3.8+
- PyTorch
- ONNX Runtime
- OpenCV
- Other dependencies listed in requirements.txt 