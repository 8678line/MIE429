"""
This program uses the trained model to analyze and shorten a video

How to use (using VS Code code):
1. Create a folder containing the conda environment, the model checkpoint, sewer_image_classifier.py, and visualizer.py
    a. Alternatively, use a different trained model
2. Copy the raw videos into the folder
3. Initialize the conda environment in VS Code
4. In main(), load the name of the video to be processed
5. Run the program

How this program works:
1. process_video() uses the model and predicts the deficiency likelihood at each frame, skipping by the 'step' value
2. visualize_predictions() plots the prediction value against video time
3. adjust_video_speed() creates a new video with dynamic speed based on the model predictions

Outputs:
- A plot of the model predictions vs video time
- A processed (shortened) version of the original video
- (Optional) Create a baseline processed video with a constant speed adjustment can compare with the result above
"""

import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from moviepy import VideoFileClip, concatenate_videoclips
from moviepy import vfx
from scipy.ndimage import gaussian_filter1d
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights
from sewer_image_classifier import ResNetModel, SewerMLClassifier

# Using the ResNet model
resnet_model = resnet18(pretrained=False)
model = ResNetModel(resnet_model)

# Transformation to process frames (assuming 224x224 input size)
weights = ResNet18_Weights.DEFAULT
preprocess = weights.transforms()

# Function to predict deficiency likelihood for a single frame
def predict(frame, model):
    # Check if CUDA is available, and select the device accordingly
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Convert frame to tensor and apply preprocessing
    frame = frame.transpose(2, 0, 1)  # From HxWxC to CxHxW
    tensor = preprocess(torch.from_numpy(frame)).to(device)
    
    # Ensure no gradients are calculated during inference
    with torch.no_grad():
        output = model(tensor.unsqueeze(0))  # Add batch dimension
        output = nn.functional.softmax(output, 1)  # Softmax for probabilities
    
    # Convert output to binary prediction (0 or 1)
    prediction = output[0, 1]
    return prediction.item()

# Function to process video and store predictions with downsampling
def process_video(video_path, model, step):
    cap = cv2.VideoCapture(video_path)
    predictions = []
    timestamps = []
    frame_count = 0

    # Process frames
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Downsample: Skip frames based on the frame_skip_interval
        if frame_count % step == 0:
            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000  # Get timestamp in seconds
            prediction = predict(frame, model)
            predictions.append(prediction)
            timestamps.append(timestamp)
        frame_count += 1

    cap.release()
    return timestamps, predictions

# Function to visualize the predictions
def visualize_predictions(timestamps, predictions, save_name):
    # Create timebar for predictions
    plt.figure(figsize=(10, 5))
    plt.plot(timestamps, predictions, color='blue', linewidth=1.5)
    plt.yticks([0, 1], ['0.00 (Low)', '1.00 (High)'])
    plt.xlabel("Time (seconds)")
    plt.title("Predicted Deficiency Probability")
    plt.savefig(save_name)

# Using MoviePy to adjust video speed based on deficiency likelihood (abrupt speed jumps)
def adjust_video_speed(input_video_path, output_video_path, timestamps, predictions, sigma, max_speed=8, min_speed=0.3):
    video = VideoFileClip(input_video_path)
    duration = video.duration  # Video duration (sec)

    # Smoothen the predictions and playback speed changes using gaussian filtering
    smoothed_predictions = gaussian_filter1d(predictions, sigma)

    # Ensure timestamps align with video duration
    timestamps = np.linspace(0, duration, len(smoothed_predictions))

    # Iterate through individual video segments and adjust the playback speeds
    # Stitch the final video together and save to the output video
    segments = []
    for i in range(len(timestamps) - 1):
        prediction = smoothed_predictions[i]
        segment = video.subclipped(timestamps[i], timestamps[i + 1])
        speed_factor = min_speed + (max_speed - min_speed) * np.exp(-3 * prediction)
        segment_new = segment.with_effects([vfx.MultiplySpeed(speed_factor)])
        segments.append(segment_new)

    result = concatenate_videoclips(segments)
    result.write_videofile(output_video_path, codec="libx264", fps=video.fps)

# Baseline: speed up the video by a constant factor
def adjust_video_speed_baseline(input_video_path, output_video_path, speed_factor=5):
    video = VideoFileClip(input_video_path)
    result = video.with_effects([vfx.MultiplySpeed(speed_factor)])
    result.write_videofile(output_video_path, codec="libx264", fps=video.fps)

def main():
    ##### Setting up the model #####
    # Load the checkpoint with the map_location to CPU
    ckpt = torch.load('C:/Users/Support/Downloads/test/best-model-epoch=2-val_loss=0.50.ckpt', map_location=torch.device('cpu'))

    # Create a new state dict with the updated keys
    state_dict = ckpt["state_dict"]
    new_state_dict = {}
    for key, value in state_dict.items():
        # Remove redundant prefix if required
        if key.startswith('model.model.'):
            new_key = key[len('model.'):]
        else:
            new_key = key
        new_state_dict[new_key] = value

    # Load the state dict into the model
    model.load_state_dict(new_state_dict)

    # Move the model to the GPU if available
    if torch.cuda.is_available():
        model.to("cuda")
    
    # Set model to evaluation mode
    model.eval()
    print("Model setup complete, analyzing video")

    ##### Testing the model on a new video #####
    video_name = None # Input video file name e.g. "video_1"
    input_video = None # Path to the input video e.g. f"[path to folder]/{input_video}.mp4
    step = 50 # Predict on every 'step' frames
    output_plot = None # Output predictions plot file name e.g. f"{video_name}_step_{step}.png"
    output_video = None # Path to processed video e.g. f"[path to folder]/processed_{video_name}.mp4"
    
    # 1. Predict the deficiency likelihood
    timestamps, predictions = process_video(input_video, model, step)
    # 2. Generate a graph of the prediction results
    visualize_predictions(timestamps, predictions, output_plot)
    print("Video analysis complete, processing video")

    # 3. Processing the video based on the predictions
    adjust_video_speed(input_video, output_video, timestamps, predictions, sigma=1.0, max_speed=8, min_speed=0.3)
    print("Video processed")

    # 4. (Optional): run the baseline video processor
    # output_video_baseline = None # Path to baseline processed video e.g. f"[path to folder]/processed_baseline_{video_name}.mp4"
    # speed_factor = 5
    # adjust_video_speed_baseline(input_video, output_video_baseline, speed_factor)

if __name__ == "__main__":
    main()
