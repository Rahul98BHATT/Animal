import os
import time
import logging
import cv2 as cv
from ultralytics import YOLO

MODEL_DIR = 'best.pt'

logging.basicConfig(
    filename="log.log", 
    filemode='a', 
    level=logging.INFO, 
    format='%(asctime)s:%(levelname)s:%(name)s:%(message)s'
)

def main():
    # Load the YOLO model
    model = YOLO(MODEL_DIR)
    logging.info("Model loaded successfully")

    # Set up the webcam
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        logging.error("Error opening webcam.")
        return

    logging.info("Webcam opened successfully")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            logging.error("Failed to capture image")
            break

        # Predict the frame
        predict = model.predict(frame, conf=0.75)
        
        # Plot boxes
        plotted = predict[0].plot()

        # Display the frame
        cv.imshow('Webcam Frame', plotted)

        # Break the loop if 'q' is pressed
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()
    logging.info("Webcam released and windows closed")

if __name__ == '__main__':
    main()