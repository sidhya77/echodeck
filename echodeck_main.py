import cv2
import mediapipe as mp
import torch
import numpy as np
from PIL import Image

# Load pretrained model
model = torch.load("asl_model.pt", map_location=torch.device('cpu'))
model.eval()

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

with mp_hands.Hands(min_detection_confidence=0.7, max_num_hands=2) as hands:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        if results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                label = "Left hand" if idx == 0 else "Right hand"
                cv2.putText(frame, f"{label}: mock gesture", 
                            (int(hand_landmarks.landmark[0].x * frame.shape[1]),
                             int(hand_landmarks.landmark[0].y * frame.shape[0])),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        cv2.imshow("EchoDeck", frame)
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
