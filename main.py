from ultralytics import YOLO
import cv2
import numpy as np
import pandas as pd
import ast

import util
from sort.sort import *
from util import get_car, read_license_plate, write_csv

def draw_border(img, top_left, bottom_right, color=(0, 255, 0), thickness=10, line_length_x=200, line_length_y=200):
    x1, y1 = top_left
    x2, y2 = bottom_right

    cv2.line(img, (x1, y1), (x1, y1 + line_length_y), color, thickness)  # Top-left
    cv2.line(img, (x1, y1), (x1 + line_length_x, y1), color, thickness)

    cv2.line(img, (x1, y2), (x1, y2 - line_length_y), color, thickness)  # Bottom-left
    cv2.line(img, (x1, y2), (x1 + line_length_x, y2), color, thickness)

    cv2.line(img, (x2, y1), (x2 - line_length_x, y1), color, thickness)  # Top-right
    cv2.line(img, (x2, y1), (x2, y1 + line_length_y), color, thickness)

    cv2.line(img, (x2, y2), (x2, y2 - line_length_y), color, thickness)  # Bottom-right
    cv2.line(img, (x2, y2), (x2 - line_length_x, y2), color, thickness)

    return img

# Initialize results dictionary and SORT tracker
results = {}
mot_tracker = Sort()

# Load models
coco_model = YOLO('yolov8n.pt')
license_plate_detector = YOLO('license_plate_detector.pt')

# Load video
cap = cv2.VideoCapture('./sample.mp4')

# Specify vehicle classes for detection
vehicles = [2, 3, 5, 7]  # Example vehicle class IDs from COCO dataset

# Read frames
frame_nmr = -1
ret = True
while ret:
    frame_nmr += 1
    ret, frame = cap.read()
    if ret:
        results[frame_nmr] = {}
        
        # Detect vehicles
        detections = coco_model(frame)[0]
        detections_ = []
        
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            if int(class_id) in vehicles:
                detections_.append([x1, y1, x2, y2, score])

                # Draw bounding box for the detected vehicle
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)  # Green box

        # Track vehicles
        track_ids = mot_tracker.update(np.asarray(detections_))

        # Detect license plates
        license_plates = license_plate_detector(frame)[0]
        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate

            # Assign license plate to car
            xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)

            if car_id != -1:
                # Expand the bounding box by a factor (e.g., 1.2 times) to capture more area
                padding = 15  # Number of pixels to add to each side
                expanded_x1 = int(max(0, x1 - padding))
                expanded_y1 = int(max(0, y1 - padding))
                expanded_x2 = int(min(frame.shape[1], x2 + padding))
                expanded_y2 = int(min(frame.shape[0], y2 + padding))

                # Crop license plate with the expanded bounding box
                license_plate_crop = frame[expanded_y1:expanded_y2, expanded_x1:expanded_x2]

                # Process license plate
                license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)

                # Read license plate number
                license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)

                if license_plate_text is not None:
                    results[frame_nmr][car_id] = {
                        'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                        'license_plate': {
                            'bbox': [expanded_x1, expanded_y1, expanded_x2, expanded_y2],
                            'text': license_plate_text,
                            'bbox_score': score,
                            'text_score': license_plate_text_score
                        }
                    }

                # Draw bounding box for the expanded license plate
                cv2.rectangle(frame, (expanded_x1, expanded_y1), (expanded_x2, expanded_y2), (255, 0, 0), 2)  # Blue box
                
                # Display license plate text above the bounding box
                cv2.putText(frame, license_plate_text, (expanded_x1, expanded_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

        # Resize the frame for display
        resized_frame = cv2.resize(frame, (800, 600))

        # Draw borders around detected cars
        for car_id in np.unique(list(results[frame_nmr].keys())):
            if 'car' in results[frame_nmr][car_id]:
                car_bbox = results[frame_nmr][car_id]['car']['bbox']
                draw_border(frame, (int(car_bbox[0]), int(car_bbox[1])), (int(car_bbox[2]), int(car_bbox[3])), (0, 255, 0), 2)

        # Display the frame with detections
        cv2.imshow('Frame', resized_frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Write results to CSV file
write_csv(results, './test.csv')

# Release resources
cap.release()
cv2.destroyAllWindows()
