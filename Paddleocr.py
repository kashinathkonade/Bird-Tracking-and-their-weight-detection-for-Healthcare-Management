

import paddleocr
from paddleocr import PaddleOCR, draw_ocr
import cv2
from PIL import Image
import matplotlib.pyplot as plt


# Initialize PaddleOCR with English language support
ocr = PaddleOCR(use_angle_cls=True, lang='en')

# Path to the video file you want to process
video_path = r"C:\Users\kashinath konade\Desktop\Bird Detection and Tracking\IMG_2665.MOV"

# Open the video file
cap = cv2.VideoCapture(video_path)

# Get video properties
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = cap.get(5)

# Create VideoWriter object to save the output video
output_path = r"C:\Users\kashinath konade\Desktop\Bird Detection and Tracking\outputvideo.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

while True:
    # Read a frame from the video
    ret, frame = cap.read()

    if not ret:
        break

    # Convert the frame to RGB (required by PaddleOCR)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Use PaddleOCR to perform text detection and recognition
    result = ocr.ocr(frame_rgb, cls=True)

    # Draw the detected boxes on the frame
    im_show = draw_ocr(frame_rgb, result[0])

    # Convert the frame back to BGR for displaying and writing to video
    frame_bgr = cv2.cvtColor(im_show, cv2.COLOR_RGB2BGR)

    # Display the frame with detected text
    cv2.imshow('OCR Video', frame_bgr)
    plt.pause(0.01)

    # Write the frame to the output video
    out.write(frame_bgr)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()











import paddleocr
from paddleocr import PaddleOCR, draw_ocr
import cv2
from PIL import Image
import matplotlib.pyplot as plt

# Initialize PaddleOCR with English language support
ocr = PaddleOCR(use_angle_cls=True, lang='en')

# Path to the video file you want to process

video_path = r"C:\Users\kashinath konade\Desktop\Bird Detection and Tracking\IMG_2665.MOV"

# Open the video file
cap = cv2.VideoCapture(video_path)

# Get video properties
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = cap.get(5)

# Create VideoWriter object to save the output video

output_path = r"C:\Users\kashinath konade\Desktop\Bird Detection and Tracking\outputvideo.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

while True:
    # Read a frame from the video
    ret, frame = cap.read()

    if not ret:
        break

    # Convert the frame to RGB (required by PaddleOCR)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Use PaddleOCR to perform text detection and recognition
    result = ocr.ocr(frame_rgb, cls=True)

    # Check if there are OCR results
    if result and result[0] and 'text' in result[0]:
        # Draw the detected boxes on the frame
        im_show = draw_ocr(frame_rgb, result[0]['text'])

        # Convert the frame back to BGR for displaying and writing to video
        frame_bgr = cv2.cvtColor(im_show, cv2.COLOR_RGB2BGR)

        # Display the frame with detected text
        cv2.imshow('OCR Video', frame_bgr)
        plt.pause(0.01)

        # Write the frame to the output video
        out.write(frame_bgr)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()




