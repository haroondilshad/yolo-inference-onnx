import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
from pathlib import Path

def visualize_results():
    """
    Visualize and compare results from PyTorch and ONNX models.
    """
    print("Visualizing and comparing results...")
    
    # Load original image
    original_img = cv2.imread('image.png')
    if original_img is None:
        print("Original image not found!")
        return
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    
    # Load PyTorch results (check multiple possible paths)
    pytorch_img_paths = [
        Path('pytorch_results/predict/image.jpg'),  # Default ultralytics path
        Path('pytorch_results/image.jpg'),          # Alternative path
        Path('pytorch_results/image0.jpg')          # Another alternative path
    ]
    
    pytorch_img = None
    for path in pytorch_img_paths:
        if path.exists():
            print(f"Found PyTorch results at {path}")
            pytorch_img = cv2.imread(str(path))
            pytorch_img = cv2.cvtColor(pytorch_img, cv2.COLOR_BGR2RGB)
            break
    
    if pytorch_img is None:
        print("PyTorch results image not found in any of the expected locations!")
    
    # Load ONNX results (check multiple possible paths)
    onnx_img_paths = [
        Path('onnx_results.jpg'),              # Direct output path
        Path('onnx_results/image.jpg'),        # Ultralytics path
        Path('onnx_results/predict/image.jpg') # Another ultralytics path
    ]
    
    onnx_img = None
    for path in onnx_img_paths:
        if path.exists():
            print(f"Found ONNX results at {path}")
            onnx_img = cv2.imread(str(path))
            onnx_img = cv2.cvtColor(onnx_img, cv2.COLOR_BGR2RGB)
            break
    
    if onnx_img is None:
        print("ONNX results image not found in any of the expected locations!")
    
    # Create figure for visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot original image
    axes[0].imshow(original_img)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Plot PyTorch results
    if pytorch_img is not None:
        axes[1].imshow(pytorch_img)
        axes[1].set_title('PyTorch Model Results')
    else:
        axes[1].text(0.5, 0.5, 'PyTorch results not found', 
                    horizontalalignment='center', verticalalignment='center')
    axes[1].axis('off')
    
    # Plot ONNX results
    if onnx_img is not None:
        axes[2].imshow(onnx_img)
        axes[2].set_title('ONNX Model Results')
    else:
        axes[2].text(0.5, 0.5, 'ONNX results not found', 
                    horizontalalignment='center', verticalalignment='center')
    axes[2].axis('off')
    
    # Save the comparison figure
    plt.tight_layout()
    plt.savefig('comparison_results.png')
    plt.close()
    
    print("Comparison saved as 'comparison_results.png'")

if __name__ == "__main__":
    visualize_results() 