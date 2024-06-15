import os, time, logging, tempfile
import logging
import cv2 as cv
from PIL import Image
import streamlit as st
from ultralytics import YOLO

MODEL_DIR = 'best.pt'

logging.basicConfig(
    filename="log.log", 
    filemode='a', 
    level=logging.INFO, 
    format='%(asctime)s:%(levelname)s:%(name)s:%(message)s'
)

def main():
    # load a model
    global model
    model = YOLO(MODEL_DIR)

    st.sidebar.header("**Animal Classes**")

    class_names = ['Buffalo', 'Elephant', 'Rhino', 'Zebra', "Cheetah", "Fox", "Jaguar", "Tiger", "Lion", "Panda"]

    for animal in class_names:
        st.sidebar.markdown(f"- *{animal.capitalize()}*")

    st.title("Real-time Animal Species Detection")
    st.write("The aim of this project is to develop an efficient computer vision model capable of real-time wildlife detection.")

    detection_mode = st.sidebar.selectbox("Select Detection Mode", ["Upload Image/Video", "Webcam"])
    
    if detection_mode == "Upload Image/Video":
        # Load image or video
        uploaded_file = st.file_uploader("Upload an image or video", type=['jpg', 'jpeg', 'png', 'mp4'])

        if uploaded_file:
            if uploaded_file.type.startswith('image'):
                inference_images(uploaded_file)
            
            if uploaded_file.type.startswith('video'):
                inference_video(uploaded_file)
    elif detection_mode == "Webcam":
        st.write("Starting Webcam for Real-time Detection")
        if st.button("Start Webcam"):
            inference_webcam()

def inference_images(uploaded_file):
    image = Image.open(uploaded_file)
    # predict the image
    predict = model.predict(image)

    # plot boxes
    boxes = predict[0].boxes
    plotted = predict[0].plot()[:, :, ::-1]

    if len(boxes) == 0:
        st.markdown("**No Detection**")

    # open the image.
    st.image(plotted, caption="Detected Image", width=600)
    logging.info("Detected Image")

def inference_video(uploaded_file):
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.write(uploaded_file.read())
    temp_file.close()

    cap = cv.VideoCapture(temp_file.name)
    frame_count = 0
    if not cap.isOpened():
        st.error("Error opening video file.")
 

    frame_placeholder = st.empty()
    stop_placeholder = st.button("Stop")

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame_count += 1
        if frame_count % 2 == 0:
            # predict the frame
            predict = model.predict(frame, conf=0.75)
            # plot boxes
            plotted = predict[0].plot()

            # Display the video
            frame_placeholder.image(plotted, channels="BGR", caption="Video Frame")
        
        # Clean up the temporary file
        if stop_placeholder:
            os.unlink(temp_file.name)
            break

    cap.release()  
    
def inference_webcam():
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        st.error("Error opening webcam.")
        return

    frame_placeholder = st.empty()
    stop_button = st.button("Stop Webcam")

    while True:
        ret, frame = cap.read()

        if not ret:
            st.error("Failed to capture image")
            break

        # predict the frame
        predict = model.predict(frame, conf=0.75)
        # plot boxes
        plotted = predict[0].plot()

        # Display the video
        frame_placeholder.image(plotted, channels="BGR", caption="Webcam Frame")

        if stop_button:
            break

    cap.release()

if __name__ == '__main__':
    main()
