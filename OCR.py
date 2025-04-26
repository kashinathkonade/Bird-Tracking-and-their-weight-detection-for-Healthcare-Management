
import cv2
import pytesseract
import numpy as np
import supervision as sv
from google.colab.patches import cv2_imshow

# initiate annotators
box_annotator = sv.BoxAnnotator(thickness=4, text_thickness=4, text_scale=2)

def extract_weight(roi):
    print("Extract Weight Function Called")
    # Check the shape of the ROI
    print("ROI Shape:", roi.shape)
    # Convert the ROI to grayscale
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Apply denoising
    denoised_roi = cv2.fastNlMeansDenoising(gray_roi, None, h=10, searchWindowSize=21)  # Adjust parameters as needed

    # Apply smoothing
    smoothed_roi = cv2.GaussianBlur(denoised_roi, (5, 5), 0)  # Adjust kernel size as needed

    # Apply a threshold to extract the digits
    # Experiment with different values for block_size and constant
    thresh = cv2.adaptiveThreshold(denoised_roi, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 9, 1)


    # Print thresholded image for debugging
    cv2_imshow(thresh)
    # Print thresholded image for debugging
    cv2_imshow(roi)
    # Use pytesseract to recognize the digits
    weight_text = pytesseract.image_to_string(thresh, config='--psm 8 --oem 1 -c tessedit_char_whitelist=0123456789')

    print("OCR Output:", weight_text)

    try:
        weight = float(weight_text)
        return weight
    except ValueError:
        print("Error converting text to float:", weight_text)
        return None

#!pip install ultralytics
from ultralytics import YOLO
model = YOLO("yolov8m.pt")

# Open the video file
cap = cv2.VideoCapture('/content/drive/MyDrive/IMG_2665.MOV')  # Replace with your video file path
# Open the video file
ret, frame = cap.read()

# Check if the frame is successfully read
if not ret:
    print("Error reading frame.")
else:
    # Get video information (e.g., frame width, height)
    height, width = frame.shape[:2]

    # detect
    results = model(frame, imgsz=1280)[0]
    detections = sv.Detections.from_yolov8(results)
    # Filter detections for class ID 3
    weight_detections = detections[detections.class_id == 3]
    # Debug print to check the number of weight detections
    print("Number of weight detections:", len(weight_detections))

    # Iterate through weight detections
    for idx, detection in enumerate(weight_detections):
        bbox, confidence, class_id, _ = detection
        print(f"Detection {idx + 1} - Class ID: {class_id}, Confidence: {confidence}, Bounding Box: {bbox}")

        # Extract the bounding box coordinates
        x, y, w, h = map(int, bbox)
        print(f"Bounding Box Coordinates: x={x}, y={y}, w={w}, h={h}")
         # Dynamically adjust the ROI around the detected bounding box

        roi = frame[y:y+100, x:x+150]
        # Debug print to check if the roi is extracted
        print("ROI Shape:", roi.shape)


        # Process the frame to extract the weight
        weight_value = extract_weight(roi)
        # Debug print to check if the weight extraction function is called
        print("Weight Value:", weight_value)

        # Do something with the extracted weight value
        if weight_value is not None:
            print("Extracted Weight:", weight_value)

            # Annotate the frame with the weight value
            cv2.putText(frame, f'Weight: {weight_value}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame with annotations
    from PIL import Image
    from IPython.display import display

    # Convert the OpenCV image to a PIL image
    frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Display the PIL image
    display(frame_pil)

# Release the video capture object
cap.release()
