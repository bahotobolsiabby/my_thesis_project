"""
mediapipe_hand_control_parallel.py

Low-latency, parallelized version of the uploaded MediaPipe hand controller.

Threads:
1. CameraCaptureThread continuously grabs the newest webcam frame.
2. MediaPipeWorker processes only the newest available frame and drops stale ones.
3. The main thread displays the latest annotated frame and reads keyboard input.

This preserves the uploaded controller's behavior:
- Press 1-7 to select a Panda joint.
- Move the hand left/right to move the selected joint.
- Pinch to close the gripper.
- Press H for home.
- Press Q to quit.
"""

import cv2
import json
import math
import os
import threading
import time
from pathlib import Path

import mediapipe as mp


PROJECT_DIR = Path(__file__).resolve().parent
COMMAND_FILE = PROJECT_DIR / "gesture_command.json"

CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30

# MediaPipe model_complexity=0 reduces inference latency.
MEDIAPIPE_MODEL_COMPLEXITY = 0
MIN_DETECTION_CONFIDENCE = 0.70
MIN_TRACKING_CONFIDENCE = 0.70

# Keep the original laterally inverted/reflection display.
MIRROR_CAMERA = True

# A home request stays active briefly so Webots cannot miss it.
HOME_REQUEST_DURATION_SECONDS = 0.30

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils


def landmark_distance(p1, p2):
    return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)


def compute_hand_size(landmarks):
    xs = [point.x for point in landmarks]
    ys = [point.y for point in landmarks]
    return max(max(xs) - min(xs), max(ys) - min(ys))


def write_json_atomically(path, data):
    """
    Write to a temporary file and replace the command file atomically.
    This prevents Webots from reading a half-written JSON document.
    """
    temporary_path = path.with_suffix(path.suffix + ".tmp")

    try:
        with open(temporary_path, "w", encoding="utf-8") as file:
            json.dump(data, file, separators=(",", ":"))

        os.replace(temporary_path, path)
    except (PermissionError, OSError):
        # Skip one command frame rather than blocking the camera pipeline.
        try:
            if temporary_path.exists():
                temporary_path.unlink()
        except OSError:
            pass


class SharedState:
    def __init__(self):
        self.lock = threading.Lock()
        self.running = True

        self.latest_camera_frame = None
        self.latest_camera_frame_id = -1
        self.latest_capture_time = 0.0

        self.latest_display_frame = None
        self.latest_command = None

        self.selected_joint = 1
        self.home_request_until = 0.0

        self.capture_fps = 0.0
        self.inference_fps = 0.0

    def stop(self):
        with self.lock:
            self.running = False

    def is_running(self):
        with self.lock:
            return self.running

    def set_selected_joint(self, joint_number):
        with self.lock:
            self.selected_joint = joint_number

    def request_home(self):
        with self.lock:
            self.home_request_until = (
                time.monotonic() + HOME_REQUEST_DURATION_SECONDS
            )

    def get_control_state(self):
        with self.lock:
            return (
                self.selected_joint,
                time.monotonic() < self.home_request_until,
            )

    def publish_camera_frame(self, frame, frame_id, capture_time, fps):
        with self.lock:
            # Single-slot buffer: the newest frame replaces the old frame.
            self.latest_camera_frame = frame
            self.latest_camera_frame_id = frame_id
            self.latest_capture_time = capture_time
            self.capture_fps = fps

    def get_latest_camera_frame(self):
        with self.lock:
            if self.latest_camera_frame is None:
                return None, -1, 0.0

            return (
                self.latest_camera_frame.copy(),
                self.latest_camera_frame_id,
                self.latest_capture_time,
            )

    def publish_result(self, display_frame, command, inference_fps):
        with self.lock:
            self.latest_display_frame = display_frame
            self.latest_command = command
            self.inference_fps = inference_fps

    def get_display_snapshot(self):
        with self.lock:
            frame = (
                None
                if self.latest_display_frame is None
                else self.latest_display_frame.copy()
            )

            return (
                frame,
                self.selected_joint,
                self.capture_fps,
                self.inference_fps,
            )


def open_low_latency_camera():
    """
    Try DirectShow first on Windows. Fall back to the default OpenCV backend.
    """
    camera = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

    if not camera.isOpened():
        camera.release()
        camera = cv2.VideoCapture(CAMERA_INDEX)

    if not camera.isOpened():
        return None

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    camera.set(cv2.CAP_PROP_FPS, CAMERA_FPS)

    # Not every camera/backend supports this, but it is harmless when ignored.
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    return camera


class CameraCaptureThread(threading.Thread):
    def __init__(self, shared_state):
        super().__init__(name="CameraCaptureThread", daemon=True)
        self.shared_state = shared_state
        self.camera = None

    def run(self):
        self.camera = open_low_latency_camera()

        if self.camera is None:
            print("ERROR: Webcam could not be opened.")
            self.shared_state.stop()
            return

        frame_id = 0
        frames_this_second = 0
        fps_window_start = time.monotonic()
        capture_fps = 0.0

        while self.shared_state.is_running():
            success, frame = self.camera.read()

            if not success:
                time.sleep(0.002)
                continue

            capture_time = time.time()
            frame_id += 1
            frames_this_second += 1

            now = time.monotonic()
            elapsed = now - fps_window_start

            if elapsed >= 1.0:
                capture_fps = frames_this_second / elapsed
                frames_this_second = 0
                fps_window_start = now

            self.shared_state.publish_camera_frame(
                frame,
                frame_id,
                capture_time,
                capture_fps,
            )

        self.camera.release()


def make_hand_command(
    landmarks,
    selected_joint,
    home_request,
    frame_id,
    capture_time,
):
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    palm_center = landmarks[9]

    pinch_distance = landmark_distance(thumb_tip, index_tip)
    hand_size = compute_hand_size(landmarks)

    gesture = (
        "close_gripper"
        if pinch_distance < 0.05
        else "open_gripper"
    )

    processed_time = time.time()

    return {
        "detected": True,
        "gesture": gesture,
        "pinch_distance": pinch_distance,
        "hand_x": palm_center.x,
        "hand_y": palm_center.y,
        "hand_size": hand_size,
        "selected_joint": selected_joint,
        "home_request": home_request,
        "frame_id": frame_id,
        "capture_timestamp": capture_time,
        "timestamp": processed_time,
        "pipeline_latency_ms": max(
            0.0,
            (processed_time - capture_time) * 1000.0,
        ),
    }


def make_empty_command(
    selected_joint,
    home_request,
    frame_id,
    capture_time,
):
    processed_time = time.time()

    return {
        "detected": False,
        "gesture": "none",
        "pinch_distance": None,
        "hand_x": None,
        "hand_y": None,
        "hand_size": None,
        "selected_joint": selected_joint,
        "home_request": home_request,
        "frame_id": frame_id,
        "capture_timestamp": capture_time,
        "timestamp": processed_time,
        "pipeline_latency_ms": max(
            0.0,
            (processed_time - capture_time) * 1000.0,
        ),
    }


def draw_control_help(frame, command, selected_joint, capture_fps, inference_fps):
    height, width, _ = frame.shape

    cv2.putText(
        frame,
        f"Selected joint: J{selected_joint}",
        (25, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (0, 255, 0),
        2,
    )

    cv2.putText(
        frame,
        f"Capture: {capture_fps:.1f} FPS | "
        f"Inference: {inference_fps:.1f} FPS",
        (25, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        (0, 255, 255),
        2,
    )

    latency = command.get("pipeline_latency_ms")
    if latency is not None:
        cv2.putText(
            frame,
            f"Newest-frame latency: {latency:.1f} ms",
            (25, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (0, 255, 255),
            2,
        )

    cv2.putText(
        frame,
        f"Gesture: {command.get('gesture', 'none')}",
        (25, 130),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.70,
        (0, 255, 0),
        2,
    )

    cv2.putText(
        frame,
        "1-7 select joint | Move hand left/right",
        (25, height - 62),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 0),
        2,
    )

    cv2.putText(
        frame,
        "Pinch close | H home | Q quit",
        (25, height - 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 0),
        2,
    )

    cv2.line(
        frame,
        (width // 2, 0),
        (width // 2, height),
        (0, 255, 255),
        1,
    )


class MediaPipeWorker(threading.Thread):
    def __init__(self, shared_state):
        super().__init__(name="MediaPipeWorker", daemon=True)
        self.shared_state = shared_state

    def run(self):
        last_processed_frame_id = -1
        processed_this_second = 0
        fps_window_start = time.monotonic()
        inference_fps = 0.0

        with mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=MEDIAPIPE_MODEL_COMPLEXITY,
            min_detection_confidence=MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
        ) as hands:
            while self.shared_state.is_running():
                frame, frame_id, capture_time = (
                    self.shared_state.get_latest_camera_frame()
                )

                if frame is None or frame_id == last_processed_frame_id:
                    time.sleep(0.001)
                    continue

                # Do not process queued old frames. Always process the newest.
                last_processed_frame_id = frame_id

                working_frame = (
                    cv2.flip(frame, 1)
                    if MIRROR_CAMERA
                    else frame
                )

                rgb_frame = cv2.cvtColor(
                    working_frame,
                    cv2.COLOR_BGR2RGB,
                )
                rgb_frame.flags.writeable = False
                result = hands.process(rgb_frame)

                selected_joint, home_request = (
                    self.shared_state.get_control_state()
                )

                command = make_empty_command(
                    selected_joint,
                    home_request,
                    frame_id,
                    capture_time,
                )

                if result.multi_hand_landmarks:
                    hand_landmarks = result.multi_hand_landmarks[0]
                    command = make_hand_command(
                        hand_landmarks.landmark,
                        selected_joint,
                        home_request,
                        frame_id,
                        capture_time,
                    )

                    mp_draw.draw_landmarks(
                        working_frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                    )

                write_json_atomically(COMMAND_FILE, command)

                processed_this_second += 1
                now = time.monotonic()
                elapsed = now - fps_window_start

                if elapsed >= 1.0:
                    inference_fps = processed_this_second / elapsed
                    processed_this_second = 0
                    fps_window_start = now

                _, _, capture_fps, _ = (
                    self.shared_state.get_display_snapshot()
                )

                draw_control_help(
                    working_frame,
                    command,
                    selected_joint,
                    capture_fps,
                    inference_fps,
                )

                self.shared_state.publish_result(
                    working_frame,
                    command,
                    inference_fps,
                )


def main():
    shared_state = SharedState()

    capture_thread = CameraCaptureThread(shared_state)
    inference_thread = MediaPipeWorker(shared_state)

    capture_thread.start()
    inference_thread.start()

    print(f"Writing commands to: {COMMAND_FILE}")
    print("Parallel low-latency pipeline started.")
    print("The newest camera frame replaces stale buffered frames.")
    print("Controls: 1-7 select joint | H home | Q quit")

    try:
        while shared_state.is_running():
            (
                display_frame,
                selected_joint,
                capture_fps,
                inference_fps,
            ) = shared_state.get_display_snapshot()

            if display_frame is not None:
                cv2.imshow(
                    "MediaPipe Parallel Joint Control",
                    display_frame,
                )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                shared_state.stop()
                break

            if key in [ord(str(number)) for number in range(1, 8)]:
                selected_joint = int(chr(key))
                shared_state.set_selected_joint(selected_joint)
                print(f"Selected joint changed to J{selected_joint}")

            elif key == ord("h"):
                shared_state.request_home()
                print("Home pose requested")

            time.sleep(0.001)

    finally:
        shared_state.stop()
        capture_thread.join(timeout=2.0)
        inference_thread.join(timeout=2.0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
