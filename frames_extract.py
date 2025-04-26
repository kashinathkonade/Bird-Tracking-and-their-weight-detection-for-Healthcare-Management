# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 16:06:20 2023

@author: ThinkPad
"""

import cv2
import os

# Open the video file
video_path = "C:/Users/kashinath konade/Downloads/video_20230804_141634.mp4"
cap = cv2.VideoCapture(video_path)

# Get the frames per second (FPS) of the video
fps = int(cap.get(cv2.CAP_PROP_FPS))
# 
# Create an output directory to save the frames
output_dir = 'new_output_frames_141634'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Initialize variables
frame_count = 0
frame_rate = 5  # 2 frames per second

while True:
    ret, frame = cap.read()
    #cap.read() is used to read the next frame from the video. The returned values are stored in ret (a boolean indicating whether a frame was successfully read) and frame (the actual frame).
    if not ret:
        break

    # Save the frame every (FPS / 2) frames
    if frame_count % (int(fps / frame_rate)) == 0:
        frame_filename = os.path.join(output_dir, f'frame_{frame_count}.jpg')
        cv2.imwrite(frame_filename, frame)

    frame_count += 1

# Release the video capture object
cap.release()

print("Frames extracted successfully.")


