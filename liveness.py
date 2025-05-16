import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

def eye_aspect_ratio(landmarks, left_indices, right_indices):
    def dist(p1, p2):
        return np.linalg.norm(np.array(p1) - np.array(p2))

    left_eye = [landmarks[i] for i in left_indices]
    right_eye = [landmarks[i] for i in right_indices]

    def calc_ear(eye):
        A = dist(eye[1], eye[5])
        B = dist(eye[2], eye[4])
        C = dist(eye[0], eye[3])
        return (A + B) / (2.0 * C)

    left_ear = calc_ear(left_eye)
    right_ear = calc_ear(right_eye)

    return (left_ear + right_ear) / 2.0

def is_real_face(frame):
    try:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            h, w, _ = frame.shape
            landmarks = [(int(p.x * w), int(p.y * h)) for p in face_landmarks.landmark]

            left_indices = [33, 160, 158, 133, 153, 144]
            right_indices = [362, 385, 387, 263, 373, 380]

            ear = eye_aspect_ratio(landmarks, left_indices, right_indices)
            print(f"[DEBUG] EAR: {ear}")
            return ear > 0.15  # Lowered threshold
        else:
            print("[DEBUG] No landmarks detected")
        return False
    except Exception as e:
        print(f"[ERROR] Liveness check failed: {e}")
        return False
