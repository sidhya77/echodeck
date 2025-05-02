import cv2
import mediapipe as mp
import torch
import torch.nn.functional as F
import numpy as np

# Load ASL model
model = torch.load("asl_model.pt", map_location=torch.device("cpu"))
model.eval()

# Define label mapping (index to letter)
LABELS = [chr(i) for i in range(65, 91)]  # 'A' to 'Z'

# Map gestures to commands
LEFT_HAND_COMMANDS = {
    'A': 'Play Track 1',
    'B': 'Pause Track 1',
    'C': 'Next Track 1',
    'D': 'Crossfade Next 1',
    'W': 'Volume Up Track 1',
    'V': 'Volume Down Track 1',
    'O': 'Random Crossfade 1'
}
RIGHT_HAND_COMMANDS = {
    'A': 'Play Track 2',
    'B': 'Pause Track 2',
    'C': 'Next Track 2',
    'D': 'Crossfade Next 2',
    'W': 'Volume Up Track 2',
    'V': 'Volume Down Track 2',
    'O': 'Random Crossfade 2'
}

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7, max_num_hands=2)

# Load webcam
cap = cv2.VideoCapture(0)

track1_name = "Fashion Killa.mp3"
track2_name = "Ride.mp3"

def preprocess_landmarks(landmarks):
    coords = np.array([[lm.x, lm.y] for lm in landmarks], dtype=np.float32).flatten()
    return torch.tensor(coords).unsqueeze(0)  # shape: (1, 42)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks and result.multi_handedness:
        for hand_landmarks, handedness in zip(result.multi_hand_landmarks, result.multi_handedness):
            hand_label = handedness.classification[0].label  # 'Left' or 'Right'
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            input_tensor = preprocess_landmarks(hand_landmarks.landmark)
            with torch.no_grad():
                output = model(input_tensor)
                prediction_idx = torch.argmax(F.softmax(output, dim=1)).item()
                predicted_letter = LABELS[prediction_idx]

            if hand_label == 'Left':
                action = LEFT_HAND_COMMANDS.get(predicted_letter, '')
                color = (0, 255, 0)  # Green
                label = f"Left hand: {predicted_letter} = {action}"
            else:
                action = RIGHT_HAND_COMMANDS.get(predicted_letter, '')
                color = (255, 0, 0)  # Blue
                label = f"Right hand: {predicted_letter} = {action}"

            # Get coordinates of wrist to anchor label
            wrist = hand_landmarks.landmark[0]
            h, w, _ = frame.shape
            x, y = int(wrist.x * w), int(wrist.y * h)
            cv2.putText(frame, label, (x - 50, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Show track names
    cv2.putText(frame, f"Track 1: {track1_name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 102, 255), 2)
    cv2.putText(frame, f"Track 2: {track2_name}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    cv2.imshow("EchoDeck", frame)

    if cv2.waitKey(5) & 0xFF == 27:  # ESC key to exit
        break

cap.release()
cv2.destroyAllWindows()
