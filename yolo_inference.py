import torch
import cv2
import numpy as np
import onnx
import onnxruntime as ort
from pathlib import Path
import matplotlib.pyplot as plt
import os
import sys

# Function to perform inference with PyTorch model
def run_pytorch_inference(model_path, image_path):
    print("Running inference with PyTorch model...")
    
    try:
        # Direct loading of the model with weights_only=False to bypass security restrictions
        print(f"Loading model from: {model_path}")
        
        # Add ultralytics classes to safe globals if available
        try:
            # Try to import ultralytics 
            import ultralytics
            from ultralytics.nn.tasks import DetectionModel
            torch.serialization.add_safe_globals([DetectionModel])
            print("Added ultralytics.nn.tasks.DetectionModel to safe globals")
        except (ImportError, AttributeError) as e:
            print(f"Could not import ultralytics classes: {e}")
        
        # Load model with weights_only=False (need to trust the source)
        model = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
        
        # Extract model from state dict if needed
        if isinstance(model, dict):
            print(f"Model keys: {model.keys() if hasattr(model, 'keys') else 'No keys'}")
            if 'model' in model:
                model = model['model']
            elif 'state_dict' in model:
                model = model['state_dict']
        
        # Convert from half precision to full precision
        model = model.float()
        
        # Set model to evaluation mode
        if hasattr(model, 'eval'):
            model.eval()
        
        # Print model info for debugging
        print(f"Model type: {type(model)}")
        if hasattr(model, 'names'):
            print(f"Model class names: {model.names}")
        
        # Attempt to use ultralytics loader as alternative
        if not hasattr(model, 'forward') and 'ultralytics' in sys.modules:
            print("Trying ultralytics YOLO model loader")
            try:
                from ultralytics import YOLO
                model = YOLO(model_path)
                # Use the ultralytics predict method
                results = model.predict(image_path, save=True, project='pytorch_results', name='')
                print(f"Ultralytics results: {results}")
                return results
            except Exception as e:
                print(f"Ultralytics loading failed: {e}")
                # Continue with manual method
        
        # Load and preprocess the image for inference
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Could not load image: {image_path}")
        
        # Create output directory
        os.makedirs('pytorch_results', exist_ok=True)
        
        # Save original image
        img_display = img.copy()
        
        # Resize image to the required input size (typically 640x640 for YOLO)
        img_resized = cv2.resize(img, (640, 640))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        
        # Convert to tensor and normalize
        img_tensor = torch.from_numpy(img_rgb.transpose(2, 0, 1)).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension
        
        # Run inference
        with torch.no_grad():
            try:
                outputs = model(img_tensor)
            except Exception as e:
                print(f"Error during model inference: {e}")
                
                # Alternative: try using forward method directly
                try:
                    print("Trying forward method directly...")
                    outputs = model.forward(img_tensor)
                except Exception as e2:
                    print(f"Error with forward method: {e2}")
                    raise
        
        # Process outputs (assuming standard YOLO format)
        if isinstance(outputs, tuple) and len(outputs) > 0:
            # Sometimes the model returns a tuple of tensors
            detections = outputs[0]
        elif isinstance(outputs, torch.Tensor):
            detections = outputs
        else:
            print(f"Unexpected output type: {type(outputs)}")
            detections = outputs
        
        print(f"Detection shape: {detections.shape if isinstance(detections, torch.Tensor) else 'Not a tensor'}")
        
        # Convert detections to numpy for processing
        if isinstance(detections, torch.Tensor):
            detections = detections.cpu().numpy()
        
        # Draw detections on the original image
        h, w, _ = img.shape
        
        # Save results without detections first
        cv2.imwrite('pytorch_results/original.jpg', img)
        
        # Process and draw detections
        if isinstance(detections, np.ndarray):
            for i, detection in enumerate(detections):
                # YOLO output format: [x1, y1, x2, y2, confidence, class_id]
                if len(detection) >= 6 and detection[4] > 0.25:  # Confidence threshold
                    x1, y1, x2, y2 = detection[0], detection[1], detection[2], detection[3]
                    
                    # Scale coordinates to original image size
                    x1 = int(x1 * w / 640)
                    y1 = int(y1 * h / 640)
                    x2 = int(x2 * w / 640)
                    y2 = int(y2 * h / 640)
                    
                    # Draw bounding box
                    cv2.rectangle(img_display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Add label
                    confidence = detection[4]
                    class_id = int(detection[5])
                    label = f"Class {class_id}: {confidence:.2f}"
                    cv2.putText(img_display, label, (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Save the image with detections
        cv2.imwrite('pytorch_results/image.jpg', img_display)
        print(f"PyTorch detection results saved to pytorch_results/image.jpg")
        
        return detections
    
    except Exception as e:
        print(f"Error during PyTorch inference: {e}")
        
        # Fallback to using ultralytics YOLO model
        try:
            print("Attempting to use ultralytics YOLO model as fallback")
            from ultralytics import YOLO
            model = YOLO(model_path)
            results = model.predict(image_path, save=True, project='pytorch_results', name='')
            print(f"Ultralytics results: {results}")
            return results
        except Exception as fallback_e:
            print(f"Ultralytics fallback failed: {fallback_e}")
        
        return None

# Function to convert PyTorch model to ONNX
def convert_to_onnx(model_path, onnx_path):
    print("Converting PyTorch model to ONNX...")
    
    try:
        # Use ultralytics export function if available
        try:
            from ultralytics import YOLO
            print("Using YOLO.export method")
            model = YOLO(model_path)
            result = model.export(format='onnx', imgsz=640)
            if os.path.exists('yolo11n.onnx'):
                print(f"ONNX model saved at: {onnx_path}")
                return onnx_path
            else:
                print("YOLO.export didn't create the expected ONNX file")
        except Exception as e:
            print(f"YOLO export failed: {e}")
        
        # Load the model with weights_only=False
        print(f"Loading model from: {model_path}")
        
        # Add ultralytics classes to safe globals if available
        try:
            import ultralytics
            from ultralytics.nn.tasks import DetectionModel
            torch.serialization.add_safe_globals([DetectionModel])
            print("Added ultralytics.nn.tasks.DetectionModel to safe globals")
        except (ImportError, AttributeError) as e:
            print(f"Could not import ultralytics classes: {e}")
        
        model = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
        
        # Extract model from state dict if needed
        if isinstance(model, dict):
            print(f"Model keys: {model.keys() if hasattr(model, 'keys') else 'No keys'}")
            if 'model' in model:
                model = model['model']
            elif 'state_dict' in model:
                model = model['state_dict']
        
        # Convert from half precision to full precision
        model = model.float()
        
        # Set model to evaluation mode
        if hasattr(model, 'eval'):
            model.eval()
        
        # Create dummy input
        dummy_input = torch.randn(1, 3, 640, 640, dtype=torch.float32)
        
        # Export to ONNX
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=12,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )
        
        print(f"ONNX model saved at: {onnx_path}")
        return onnx_path
    
    except Exception as e:
        print(f"Error during ONNX conversion: {e}")
        return None

# Function to perform inference with ONNX model
def run_onnx_inference(onnx_path, image_path):
    print("Running inference with ONNX model...")
    
    try:
        # Check if ONNX file exists
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f"ONNX model file not found: {onnx_path}")
        
        # Check the ONNX model
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        
        # Create ONNX Runtime session
        providers = ['CPUExecutionProvider']
        if 'CUDAExecutionProvider' in ort.get_available_providers():
            providers.insert(0, 'CUDAExecutionProvider')
        
        session = ort.InferenceSession(onnx_path, providers=providers)
        
        # Load and preprocess the image for ONNX inference
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Could not load image: {image_path}")
        
        # Save original image for drawing detections
        img_display = img.copy()
        h, w, _ = img.shape
        
        # Preprocess image
        img_resized = cv2.resize(img, (640, 640))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_transpose = img_rgb.transpose(2, 0, 1)  # HWC to CHW
        img_expanded = np.expand_dims(img_transpose, axis=0)
        img_normalized = img_expanded.astype(np.float32) / 255.0  # Normalize
        
        # Get input and output names
        input_name = session.get_inputs()[0].name
        output_names = [o.name for o in session.get_outputs()]
        
        # Run inference
        outputs = session.run(output_names, {input_name: img_normalized})
        
        # Process outputs - for YOLOv8 format 
        # YOLOv8 output is typically [batch, num_classes+4, num_anchors]
        output = outputs[0]
        print(f"ONNX Raw output shape: {output.shape}")
        
        # ALTERNATIVE: Try using ultralytics for ONNX prediction
        try:
            print("Attempting to use ultralytics for ONNX inference")
            from ultralytics import YOLO
            
            # Save ONNX results using ultralytics
            model = YOLO(onnx_path)
            results = model.predict(image_path, save=True, project='.', name='onnx_results')
            
            # If successful, we're done
            print("Successfully used ultralytics for ONNX inference")
            return outputs
        except Exception as e:
            print(f"Ultralytics ONNX inference failed: {e}")
            print("Falling back to manual processing...")
        
        # Manual processing - transpose if needed
        if output.shape[1] > output.shape[2]:  # If shape is [batch, 84, 8400]
            output = np.transpose(output, (0, 2, 1))  # Transpose to [batch, 8400, 84]
        
        # Process output to get detections
        # Assuming format [batch, num_boxes, 4+1+num_classes]
        # First 4 values are bounding box coordinates, next is confidence, rest are class probabilities
        boxes = []
        confidences = []
        class_ids = []
        
        # Confidence threshold
        conf_threshold = 0.25
        
        # Process each detection
        num_boxes = output.shape[1]
        num_classes = output.shape[2] - 5
        
        for i in range(num_boxes):
            # Get box confidence
            box_confidence = output[0, i, 4]
            
            if box_confidence > conf_threshold:
                # Get class with highest probability
                class_scores = output[0, i, 5:5+num_classes]
                class_id = np.argmax(class_scores)
                class_score = class_scores[class_id]
                
                # Combine confidence scores
                confidence = box_confidence * class_score
                
                if confidence > conf_threshold:
                    # Get bounding box coordinates
                    x, y, width, height = output[0, i, 0:4]
                    
                    # Convert from center format to corner format
                    x1 = int((x - width/2) * w)
                    y1 = int((y - height/2) * h)
                    x2 = int((x + width/2) * w)
                    y2 = int((y + height/2) * h)
                    
                    # Add to lists
                    boxes.append([x1, y1, x2, y2])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        # Apply non-maximum suppression to remove overlapping boxes
        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, 0.4)
        
        # Draw detections
        for i in indices:
            # Ensure i is a scalar for OpenCV 4.5+
            if isinstance(i, np.ndarray):
                i = i[0]
                
            box = boxes[i]
            x1, y1, x2, y2 = box
            confidence = confidences[i]
            class_id = class_ids[i]
            
            # Draw bounding box
            cv2.rectangle(img_display, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Add label
            label = f"Class {class_id}: {confidence:.2f}"
            cv2.putText(img_display, label, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Save the image with ONNX detections
        cv2.imwrite('onnx_results.jpg', img_display)
        print(f"ONNX detection results saved to onnx_results.jpg")
        
        return outputs
    
    except Exception as e:
        print(f"Error during ONNX inference: {e}")
        
        # Try alternative using ultralytics
        try:
            print("Attempting alternative ONNX inference with ultralytics")
            from ultralytics import YOLO
            model = YOLO(onnx_path)
            results = model.predict(image_path, save=True, project='.', name='onnx_results')
            print(f"Ultralytics ONNX results: {results}")
            return outputs
        except Exception as alt_e:
            print(f"Alternative ONNX inference failed: {alt_e}")
        
        return None

if __name__ == "__main__":
    model_path = 'yolo11n.pt'
    image_path = 'image.png'
    onnx_path = 'yolo11n.onnx'
    
    # Check if files exist
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found!")
        sys.exit(1)
    
    if not os.path.exists(image_path):
        print(f"Error: Image file {image_path} not found!")
        sys.exit(1)
    
    # 1. Run inference with PyTorch model
    pytorch_results = run_pytorch_inference(model_path, image_path)
    
    # 2. Convert model to ONNX
    onnx_model_path = convert_to_onnx(model_path, onnx_path)
    
    # 3. Run inference with ONNX model
    if onnx_model_path and os.path.exists(onnx_model_path):
        onnx_results = run_onnx_inference(onnx_model_path, image_path)
    else:
        print("Skipping ONNX inference due to conversion failure")
    
    print("Completed all inference tasks!")
    
    # 4. Visualize and compare results
    try:
        from visualize_results import visualize_results
        print("Generating comparison visualization...")
        visualize_results()
    except Exception as e:
        print(f"Failed to generate visualization: {e}")
    
    print("All tasks completed successfully!") 