import cv2
import os
import threading

import numpy as np
from ultralytics import YOLO
from dotenv import load_dotenv


# --- Configuration ---
load_dotenv()

ip = os.getenv("IP")
port = os.getenv("PORT")
stream_url = f"http://{ip}:{port}/video"

CONFIDENCE_THRESHOLD = 0.4
WINDOW_TITLE = "DroidCam Wi-Fi"
MODEL_PATH = "yolov8s.pt"

# --- Shared Thread Data ---
latest_frame = None
latest_results = []
frame_lock = threading.Lock()
results_lock = threading.Lock()


# --- Shortcuts ---
toggle = {"bounding_boxes": False}

# --- Color Utility ---
def get_class_color(class_id):
    hue = int((class_id * 137.508) % 180)  # Golden angle distribution across HSV hue range
    hsv_color = np.uint8([[[hue, 220, 255]]])
    bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)
    return tuple(int(x) for x in bgr_color[0][0])


# --- Threads ---
# Capture Thread
def capture_frames(cap: cv2.VideoCapture):
    global latest_frame
    frame_count = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            # print("Capture thread: failed to read frame.")
            break

        with frame_lock:
            latest_frame = frame


        frame_count += 1
        # if frame_count % 30 == 0:
        #     print(f"Capture thread: {frame_count} frames captured")

# Inference Thread
def run_inference(model: YOLO):
    global latest_results
    inference_count = 0

    while True:
        with frame_lock:
            frame = latest_frame

        if frame is None:
            continue

        results = list(model.track(frame, stream=True, verbose=False))

        with results_lock:
            latest_results = results

        inference_count += 1
        # if inference_count % 10 == 0:
        #     print(f"Inference thread: {inference_count} inferences run")


# --- Detection & Drawing ---
def draw_object_boxes(model: YOLO, frame):
    if not toggle["bounding_boxes"]:
        return
    
    with results_lock:
        results = latest_results

    # Iterate through all objects found within the frame
    for result in results:
        class_names = result.names

        # Apply bounding boxes and labels for each object
        for box in result.boxes:
            confidence = float(box.conf[0])

            if confidence < CONFIDENCE_THRESHOLD:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_id = int(box.cls[0])
            class_name = class_names[class_id]
            color = get_class_color(class_id)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            cv2.putText(
                frame,
                f"{class_name} {confidence:.2f}",
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )


# --- Main ---
def start_phone_camera():
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "fflags;nobuffer|flags;low_delay|framedrop;1"

    cap = cv2.VideoCapture(stream_url)
    model = YOLO(MODEL_PATH)

    # Perform threaded tasks
    capture_thread = threading.Thread(target=capture_frames, args=(cap,), daemon=True)
    inference_thread = threading.Thread(target=run_inference, args=(model,), daemon=True)

    capture_thread.start()
    inference_thread.start()

    # Apply settings to video capture
    cap.set(cv2.CAP_PROP_FPS, 24)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print("Could not open video stream.")
        return

    while True:
        with frame_lock:
            frame = latest_frame

        if frame is None:
            continue

        frame = frame.copy()

        # Apply bounding boxes and display frame
        draw_object_boxes(model, frame)
        cv2.imshow(WINDOW_TITLE, frame)

        # Check for keyboard input
        key = cv2.waitKey(1) & 0xFF

        if key == ord("b"):
            toggle["bounding_boxes"] = not toggle["bounding_boxes"]

        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()



start_phone_camera()