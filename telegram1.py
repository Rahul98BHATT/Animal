import os
import time
import logging
import cv2 as cv
from ultralytics import YOLO
import pygame
import requests

MODEL_DIR = 'best.pt'
ALERT_SOUND = 'alert.wav'  # Path to the WAV file to be played when an animal is detected
TELEGRAM_BOT_TOKEN = '7377193087:AAF8ILEg0LjO8Ucx9omlYhXRAuR0cDto0FU'
CHAT_ID = '1765902365'

logging.basicConfig(
    filename="log.log", 
    filemode='a', 
    level=logging.INFO, 
    format='%(asctime)s:%(levelname)s:%(name)s:%(message)s'
)

def play_alert_sound():
    pygame.mixer.init()
    alert_sound = pygame.mixer.Sound(ALERT_SOUND)
    alert_sound.play()

def send_telegram_photo(photo_path, caption):
    url = f'https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto'
    with open(photo_path, 'rb') as photo:
        response = requests.post(url, data={'chat_id': CHAT_ID, 'caption': caption}, files={'photo': photo})
    if response.status_code == 200:
        logging.info("Photo sent to Telegram successfully")
    else:
        logging.error(f"Failed to send photo to Telegram: {response.status_code}, {response.text}")

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

        # Check if any boxes are detected
        if len(predict[0].boxes) > 0:
            play_alert_sound()
            logging.info("Animal detected and alert sound played")

            # Save the frame with the detection
            photo_path = 'detected_animal.jpg'
            cv.imwrite(photo_path, plotted)

            # Send the photo to Telegram with a caption
            caption = "Animal detected!"
            send_telegram_photo(photo_path, caption)

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
