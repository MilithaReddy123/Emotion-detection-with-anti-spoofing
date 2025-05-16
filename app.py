import streamlit as st
import cv2
import numpy as np
from deepface import DeepFace
from liveness import is_real_face
import threading
import time

st.set_page_config(page_title="Emotion Detector (Optimized)", layout="wide")
st.title("🧠 Real-time Emotion Detection with Anti-Spoofing")

FRAME_WIDTH = 480
FRAME_HEIGHT = 360
FRAME_WINDOW = st.image([])
status_placeholder = st.empty()

# Shared data
latest_frame = None
emotion_result = ("Neutral", 0)
live_result = False
lock = threading.Lock()
running = True

def emotion_liveness_worker():
    global latest_frame, emotion_result, live_result, running

    while running:
        time.sleep(2)  # Run analysis every 2 seconds to reduce CPU load
        if latest_frame is not None:
            with lock:
                frame = latest_frame.copy()

            live_result = is_real_face(frame)

            if live_result:
                try:
                    res = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False, detector_backend='opencv')
                    emotion = res[0]['dominant_emotion']
                    score = res[0]['emotion'][emotion]
                    emotion_result = (emotion, score)
                except Exception as e:
                    print("[ERROR] DeepFace failed:", e)
                    emotion_result = ("Unknown", 0)
            else:
                emotion_result = ("Fake Face", 0)

def main():
    global latest_frame, emotion_result, live_result, running

    cap = cv2.VideoCapture(0)
    threading.Thread(target=emotion_liveness_worker, daemon=True).start()

    while True:
        ret, frame = cap.read()
        if not ret:
            st.warning("Cannot access webcam.")
            break

        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

        with lock:
            latest_frame = frame.copy()

        emotion, score = emotion_result
        live_text = "Real Face ✅" if live_result else "Fake Face ❌"
        label = f"{emotion} ({score:.1f}%) | {live_text}"
        color = (0, 255, 0) if live_result else (0, 0, 255)

        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, color, 2)

        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        status_placeholder.markdown(f"**Status:** <span style='color: {'green' if live_result else 'red'}'>{live_text}</span>", unsafe_allow_html=True)

    running = False
    cap.release()

if __name__ == "__main__":
    main()
