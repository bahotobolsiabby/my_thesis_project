import cv2
import json
import math
import time
from pathlib import Path

import mediapipe as mp


# Save gesture file in the same folder as this script
PROJECT_DIR = Path(__file__).resolve().parent
COMMAND_FILE = PROJECT_DIR / "gesture_command.json"

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils


def landmark_distance(p1, p2):
    """Compute 2D distance between two MediaPipe landmarks."""
    return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)


def write_json_safely(path, data):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except PermissionError:
        # If Webots is reading the file at the same time, skip this frame.
        pass


def get_hand_command(landmarks):
    """
    Converts MediaPipe hand landmarks into simple robot commands.

    Controls:
    - Pinch thumb + index finger = close gripper
    - Open thumb + index finger = open gripper
    - Move palm left/right = rotate robot base
    - Move palm up/down = raise/lower robot arm
    """

    thumb_tip = landmarks[4]
    index_tip = landmarks[8]

    # Landmark 9 is around the middle knuckle / palm area.
    # It is more stable than the fingertips for movement control.
    palm_center = landmarks[9]

    pinch_distance = landmark_distance(thumb_tip, index_tip)

    if pinch_distance < 0.05:
        gesture = "close_gripper"
    else:
        gesture = "open_gripper"

    command_data = {
        "detected": True,
        "gesture": gesture,
        "pinch_distance": pinch_distance,
        "hand_x": palm_center.x,
        "hand_y": palm_center.y,
        "timestamp": time.time()
    }

    return command_data


def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Webcam could not be opened.")
        return

    print(f"Writing MediaPipe commands to: {COMMAND_FILE}")
    print("Press Q to quit.")

    with mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    ) as hands:

        while True:
            success, frame = cap.read()

            if not success:
                print("Error: Could not read webcam frame.")
                break

            # Mirror view
            frame = cv2.flip(frame, 1)

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb_frame)

            command_data = {
                "detected": False,
                "gesture": "none",
                "pinch_distance": None,
                "hand_x": None,
                "hand_y": None,
                "timestamp": time.time()
            }

            if result.multi_hand_landmarks:
                hand_landmarks = result.multi_hand_landmarks[0]
                landmarks = hand_landmarks.landmark

                command_data = get_hand_command(landmarks)

                mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

                gesture = command_data["gesture"]
                pinch_distance = command_data["pinch_distance"]
                hand_x = command_data["hand_x"]
                hand_y = command_data["hand_y"]

                cv2.putText(
                    frame,
                    f"Gesture: {gesture}",
                    (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    f"Pinch distance: {pinch_distance:.3f}",
                    (30, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    f"hand_x: {hand_x:.3f}  hand_y: {hand_y:.3f}",
                    (30, 130),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

            write_json_safely(COMMAND_FILE, command_data)

            cv2.imshow("MediaPipe Hand Control", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
