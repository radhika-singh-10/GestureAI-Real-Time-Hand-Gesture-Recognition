# import cv2
# import torch
# from torchvision.transforms import Compose, CenterCrop, ToTensor, Normalize
# from model import GestureDetection
# from PIL import Image
# import numpy as np

# model = GestureDetection(num_classes=5)  
# checkpoint = torch.load('./trainings/jpeg_model/jester_conv_4_classes/checkpoint.pth.tar', map_location='cpu') 
# state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
# # model.load_state_dict(state_dict)



# state_dict_modified = {}
# for key, value in state_dict.items():
#     if key.startswith('module.'):
#         new_key = key[len('module.'):]  # Remove the 'module.' prefix
#         state_dict_modified[new_key] = value
#     else:
#         state_dict_modified[key] = value

# model.load_state_dict(state_dict_modified)
# model.eval()

# transform = Compose([
#     CenterCrop(84),
#     ToTensor(),
#     Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])

# label_mapping_path = './20bn-jester-v1/annotations/jester-v1-labels-quick-testing copy.csv'


# def load_label_mapping(label_mapping_path):
#     label_mapping = {}
#     with open(label_mapping_path, 'r') as f:
#         for i, line in enumerate(f):
#             label = line.strip()
#             label_mapping[i] = label
#     return label_mapping

# label_mapping = load_label_mapping(label_mapping_path)


# def preprocess_frames(frames, transform):
#     # Apply transformations to each frame in the sequence
#     processed_frames = [transform(Image.fromarray(frame)) for frame in frames]
#     # Stack frames along the new 'depth' dimension
#     frame_stack = torch.stack(processed_frames, dim=1)
#     return frame_stack.unsqueeze(0)  # Add batch dimension

# # Initialize the webcam
# cap = cv2.VideoCapture(0)
# # Collect a sequence of frames
# frame_sequence = []
# sequence_length = 16  # The 'depth' your Conv3D model expects

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break
    
#     # Convert frame to RGB
#     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
#     if len(frame_sequence) < sequence_length:
#         # Add the current frame to the sequence
#         frame_sequence.append(frame_rgb)
#         continue
#     else:
#         # If we already have a sequence, remove the oldest frame
#         frame_sequence.pop(0)
#         frame_sequence.append(frame_rgb)
    
#     # Preprocess the sequence of frames
#     frame_processed = preprocess_frames(frame_sequence, transform)
    
#     with torch.no_grad():
#         outputs = model(frame_processed)
    
#     # Convert the model output to a human-readable label
#     _, predicted = torch.max(outputs.data, 1)
#     # cv2.imshow('Predicted', predicted)
#     predicted_gesture = label_mapping[predicted.item()]   # Replace with your label mapping logic
    
#     # Display the resulting frame with predicted gesture
#     cv2.putText(frame, predicted_gesture, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#     cv2.imshow('Webcam - Gesture Recognition', frame)
    

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break


# cap.release()
# cv2.destroyAllWindows()

import cv2
import torch
from torchvision.transforms import Compose, CenterCrop, ToTensor, Normalize
from model import GestureDetection
from PIL import Image
import numpy as np
import subprocess

# Add NSCameraUseContinuityCameraDeviceType to Info.plist
subprocess.run(["defaults", "write", "Info.plist", "NSCameraUseContinuityCameraDeviceType", "-bool", "YES"])

model = GestureDetection(num_classes=7)  
checkpoint = torch.load('./trainings/jpeg_model/7_classes/v13/checkpoint.pth.tar', map_location='cpu') 
state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint

state_dict_modified = {}
for key, value in state_dict.items():
    if key.startswith('module.'):
        new_key = key[len('module.'):]  # Remove the 'module.' prefix
        state_dict_modified[new_key] = value
    else:
        state_dict_modified[key] = value

model.load_state_dict(state_dict_modified)
model.eval()

transform = Compose([
    CenterCrop(84),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

label_mapping_path = './20bn-jester-v1/annotations/jester-v1-labels-quick-testing copy.csv'


def load_label_mapping(label_mapping_path):
    label_mapping = {}
    with open(label_mapping_path, 'r') as f:
        for i, line in enumerate(f):
            label = line.strip()
            label_mapping[i] = label
    return label_mapping

label_mapping = load_label_mapping(label_mapping_path)


def preprocess_frames(frames, transform):
    processed_frames = [transform(Image.fromarray(frame)) for frame in frames]
    frame_stack = torch.stack(processed_frames, dim=1)
    return frame_stack.unsqueeze(0)

cap = cv2.VideoCapture(0)
frame_sequence = []
sequence_length = 16

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    if len(frame_sequence) < sequence_length:
        frame_sequence.append(frame_rgb)
        continue
    else:
        frame_sequence.pop(0)
        frame_sequence.append(frame_rgb)
    
    frame_processed = preprocess_frames(frame_sequence, transform)
    
    with torch.no_grad():
        outputs = model(frame_processed)
    
    _, predicted = torch.max(outputs.data, 1)
    predicted_gesture = label_mapping[predicted.item()]
    
    cv2.putText(frame, predicted_gesture, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Webcam - Gesture Recognition', frame)
    

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
