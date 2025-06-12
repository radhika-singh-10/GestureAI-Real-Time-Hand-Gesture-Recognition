import cv2
import torch
from torchvision.transforms import Compose, CenterCrop, ToTensor, Normalize
from model import GestureDetection
from PIL import Image
import numpy as np
import subprocess
#references are added in comments
# https://docs.python.org/3/library/subprocess.html
subprocess.run(["defaults", "write", "Info.plist", "NSCameraUseContinuityCameraDeviceType", "-bool", "YES"])

label_mapping_path = './20bn-jester-v1/annotations/jester-v1-labels-quick-testing copy.csv'

    
model = GestureDetection(num_classes=7)  
# https://pytorch.org/docs/stable/generated/torch.load.html
checkpoint = torch.load('./trainings/jpeg_model/7_classes/v13/checkpoint.pth.tar', map_location='cpu') 
state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint

# This code is to load the model trained with multiple GPUs and run it with a single GPU/CPU
state_dict_modified = {}
for key, value in state_dict.items():
    if key.startswith('module.'):
        new_key = key[len('module.'):]  
        state_dict_modified[new_key] = value
    else:
        state_dict_modified[key] = value

model.load_state_dict(state_dict_modified)
model.eval()

# https://pytorch.org/vision/main/generated/torchvision.transforms.Compose.html
transform = Compose([
    CenterCrop(84),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def load_label_mapping(label_mapping_path):
    """
    Load label mapping from a file.
    Args:
        label_mapping_path (str): Path to the label mapping file.
    Returns:
        dict: A dictionary mapping numeric labels to their corresponding string labels.
    """
    label_mapping = {}
    with open(label_mapping_path, 'r') as f:
        for i, line in enumerate(f):
            label = line.strip()
            label_mapping[i] = label
    return label_mapping

label_mapping = load_label_mapping(label_mapping_path)


def preprocess_frames(frames, transform):
    """
    Preprocess frames using a specified transformation.
    Args:
        frames (list of numpy arrays): Input frames as a list of numpy arrays.
        transform: Transformation function to apply to each frame.
    Returns:
        torch.Tensor: Preprocessed frames stacked along the time dimension.
    """
    processed_frames = [transform(Image.fromarray(frame)) for frame in frames]
    frame_stack = torch.stack(processed_frames, dim=1)
    return frame_stack.unsqueeze(0)

# Initialize video
cap = cv2.VideoCapture(0)


frame_sequence = []
sequence_length = 16

while True:
    # Read a frame from the video capture
    ret, frame = cap.read()
    
    if not ret:
        break
    # https://docs.opencv.org/4.x/d8/d01/group__imgproc__color__conversions.html#gga4e0972be5de079fed4e3a10e24ef5ef0a9cef93380497571d867d18149c268ed1
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Add the current RGB frame to the frame sequence
    if len(frame_sequence) < sequence_length:
        frame_sequence.append(frame_rgb)
    else:
        frame_sequence.pop(0)
        frame_sequence.append(frame_rgb)
    
    frame_processed = preprocess_frames(frame_sequence, transform)
    
    with torch.no_grad():
        # Model Inference
        outputs = model(frame_processed)
    
    _, predicted = torch.max(outputs.data, 1)
    predicted_gesture = label_mapping[predicted.item()]
    
    cv2.putText(frame, predicted_gesture, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Webcam - Gesture Recognition', frame)
    
    # Check for user input to quit the application
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object
cap.release()

# Close all OpenCV windows
cv2.destroyAllWindows()

