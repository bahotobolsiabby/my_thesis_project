import cv2
import json
import math
import time
from pathlib import Path

import mediapipe as mp


# This file is read by Webots.
PROJECT_DIR = Path(__file__).resolve().parent
COMMAND_FILE = PROJECT_DIR / "gesture_command.json"

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# You can change this while the program is running by pressing 1-7
selected_joint = 1

# Press H to request home pose once
home_request = False


def landmark_distance(p1, p2):
    return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)


def compute_hand_size(landmarks):
    xs = [p.x for p in landmarks]
    ys = [p.y for p in landmarks]
    return max(max(xs) - min(xs), max(ys) - min(ys))


def write_json_safely(path, data):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except PermissionError:
        # Webots may read the file at the same time. Skip this frame.
        pass


def get_hand_command(landmarks):
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    palm_center = landmarks[9]

    pinch_distance = landmark_distance(thumb_tip, index_tip)
    hand_size = compute_hand_size(landmarks)

    if pinch_distance < 0.05:
        gesture = "close_gripper"
    else:
        gesture = "open_gripper"

    return {
        "detected": True,
        "gesture": gesture,
        "pinch_distance": pinch_distance,
        "hand_x": palm_center.x,
        "hand_y": palm_center.y,
        "hand_size": hand_size,
        "selected_joint": selected_joint,
        "home_request": home_request,
        "timestamp": time.time()
    }


def draw_control_help(frame, command_data):
    h, w, _ = frame.shape

    gesture = command_data.get("gesture", "none")
    hand_x = command_data.get("hand_x", None)
    hand_y = command_data.get("hand_y", None)
    hand_size = command_data.get("hand_size", None)

    cv2.putText(frame, f"Selected joint: J{selected_joint}", (30, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    cv2.putText(frame, "Press 1-7 to select joint | Move hand left/right to move selected joint",
                (30, h - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

    cv2.putText(frame, "Pinch = close gripper | Open fingers = open gripper | H = home | Q = quit",
                (30, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

    cv2.putText(frame, f"Gesture: {gesture}", (30, 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    if hand_x is not None and hand_y is not None:
        cv2.putText(frame, f"hand_x: {hand_x:.3f}  hand_y: {hand_y:.3f}",
                    (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    if hand_size is not None:
        cv2.putText(frame, f"hand_size/depth: {hand_size:.3f}",
                    (30, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.line(frame, (int(w * 0.5), 0), (int(w * 0.5), h), (0, 255, 255), 1)
    cv2.putText(frame, "center", (int(w * 0.5) + 5, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)


def main():
    global selected_joint, home_request

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Webcam could not be opened.")
        return

    print(f"Writing MediaPipe commands to: {COMMAND_FILE}")
    print("Controls:")
    print("  Press 1-7 = select Panda joint")
    print("  Move hand left/right = move selected joint")
    print("  Pinch = close gripper")
    print("  Open fingers = open gripper")
    print("  Press H = request home pose")
    print("  Press Q = quit")

    with mp_hands.Hands(max_num_hands=1,
                        min_detection_confidence=0.7,
                        min_tracking_confidence=0.7) as hands:

        while True:
            success, frame = cap.read()

            if not success:
                print("Error: Could not read webcam frame.")
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb_frame)

            command_data = {
                "detected": False,
                "gesture": "none",
                "pinch_distance": None,
                "hand_x": None,
                "hand_y": None,
                "hand_size": None,
                "selected_joint": selected_joint,
                "home_request": home_request,
                "timestamp": time.time()
            }

            if result.multi_hand_landmarks:
                hand_landmarks = result.multi_hand_landmarks[0]
                landmarks = hand_landmarks.landmark

                command_data = get_hand_command(landmarks)

                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            draw_control_help(frame, command_data)
            write_json_safely(COMMAND_FILE, command_data)

            # home_request is only sent for one frame
            home_request = False

            cv2.imshow("MediaPipe Joint Control", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            elif key in [ord(str(i)) for i in range(1, 8)]:
                selected_joint = int(chr(key))
                print(f"Selected joint changed to J{selected_joint}")

            elif key == ord("h"):
                home_request = True
                print("Home pose requested")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
