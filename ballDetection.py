# Import necessary libraries
import cv2
import numpy as np
import os
import random
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.model_selection import train_test_split

# Step 1: Generate pseudo-annotations for video data
# The purpose of this step is to create an initial dataset that we can use to train the object detection model
# Here we are going to randomly create bounding boxes around regions in frames where we assume the ball is

def generate_pseudo_annotations(video_path, output_dir, annotation_count=10):
    # Open video file
    video = cv2.VideoCapture(video_path)
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    os.makedirs(output_dir, exist_ok=True)

    annotations = []

    # Iterate over the total number of frames
    for frame_idx in range(frame_count):
        ret, frame = video.read()
        if not ret:
            break

        # Randomly generate annotations
        if frame_idx % annotation_count == 0:
            # Assuming a simple scenario: random bounding box around a possible ball
            x_min = random.randint(0, width // 2)
            y_min = random.randint(0, height // 2)
            box_width = random.randint(20, 80)
            box_height = random.randint(20, 80)

            x_max = min(x_min + box_width, width)
            y_max = min(y_min + box_height, height)

            # Draw the bounding box for visualization
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            annotations.append({'frame': frame_idx, 'bbox': [x_min, y_min, x_max, y_max]})

        # Save frames for further visualization
        output_frame_path = os.path.join(output_dir, f"frame_{frame_idx}.jpg")
        cv2.imwrite(output_frame_path, frame)

    video.release()
    return annotations

# Example usage
pseudo_annotations = generate_pseudo_annotations("sports_video.mp4", "output_frames")

# Step 2: Build an Object Detection Model
# We'll use a simple Convolutional Neural Network (CNN) to showcase object detection

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(4, activation='linear')  # Outputs bounding box coordinates
])

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

# Step 3: Prepare Training Data (using generated pseudo-annotations)

def prepare_training_data(annotations, frame_dir):
    X = []
    y = []
    for annotation in annotations:
        frame_path = os.path.join(frame_dir, f"frame_{annotation['frame']}.jpg")
        frame = cv2.imread(frame_path)
        frame = cv2.resize(frame, (64, 64))
        X.append(frame)
        y.append(annotation['bbox'])

    X = np.array(X) / 255.0
    y = np.array(y)
    return X, y

# Example usage
X, y = prepare_training_data(pseudo_annotations, "output_frames")

# Split data into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))

# Step 4: Evaluation Procedure with Visualizations

def evaluate_model(model, X_val, y_val):
    predictions = model.predict(X_val)
    for i in range(len(predictions)):
        plt.figure()
        plt.imshow(X_val[i])
        pred_box = predictions[i]
        true_box = y_val[i]
        plt.gca().add_patch(plt.Rectangle((pred_box[0], pred_box[1]), pred_box[2]-pred_box[0], pred_box[3]-pred_box[1], linewidth=2, edgecolor='r', facecolor='none'))
        plt.gca().add_patch(plt.Rectangle((true_box[0], true_box[1]), true_box[2]-true_box[0], true_box[3]-true_box[1], linewidth=2, edgecolor='g', facecolor='none'))
        plt.show()

# Example usage
evaluate_model(model, X_val, y_val)

# Step 5: Iteratively Improve the Dataset
# The simplest approach here could be selecting frames with the worst predictions, improving annotations, and retraining.
# Here, you could use various heuristics to decide which frames need more accurate annotations.

# Dockerfile for the Project
# Create a Dockerfile to containerize the application
DOCKERFILE_CONTENT = '''
# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir numpy opencv-python-headless matplotlib tensorflow scikit-learn

# Run ballDetection.py when the container launches
CMD ["python", "ballDetection.py"]
'''

# Save Dockerfile
with open('Dockerfile', 'w') as f:
    f.write(DOCKERFILE_CONTENT)

# Instructions to build and run the Docker container
print("\nTo build the Docker container, run the following command:")
print("docker build -t ball_detection .")
print("\nTo run the Docker container, use:")
print("docker run -v $(pwd):/app ball_detection")

