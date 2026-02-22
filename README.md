# DroidCam Object Detection

A real-time object detection app that uses your phone as a wireless camera via DroidCam, with live bounding boxes and labels powered by YOLOv8.

## Demo

- Streams live video from your phone over Wi-Fi
- Detects and labels objects in real time using YOLOv8
- Smooth playback via multithreaded capture and inference
- Toggle bounding boxes on/off with a keyboard shortcut

## Requirements

- Python 3.8+
- [DroidCam](https://www.dev47apps.com/) installed on your phone
- Your phone and PC on the same Wi-Fi network

Install dependencies:

```bash
pip install opencv-python ultralytics python-dotenv numpy
```

## Setup

**1. Create a `.env` file** in the project root with your phone's IP and port as shown in the DroidCam app:

```
IP=''
PORT=''
```

**2. Download the YOLOv8 model** — it will be downloaded automatically on first run, no manual steps needed.

**3. Open the DroidCam app** on your phone and make sure it is running in the foreground.

## Usage

```bash
python phone_camera.py
```

| Key | Action          |
|-----|-----------------|
| `b` | Toggle bounding boxes and labels |
| `q` | Quit            |

## How It Works

The app runs three concurrent threads to keep the video smooth even while the model is computing:

- **Capture thread** — reads frames from the phone stream as fast as possible
- **Inference thread** — continuously runs YOLOv8 on the latest frame
- **Main thread** — displays the latest frame with the most recent detections overlaid

Without threading, the display would freeze for several hundred milliseconds each time the model ran. With threading, the capture and display are never blocked by inference.

## Project Structure

```
├── phone_camera.py       # Entry point
├── .env          # IP and port config (not committed to version control)
├── .gitignore
└── README.md
```

## Troubleshooting

**`Could not open video stream`**
- Make sure DroidCam is open and running on your phone
- Verify your phone and PC are on the same Wi-Fi network
- Open `http://{IP}:{PORT}/video` in your browser — if it works there, try removing `cv2.CAP_FFMPEG` from the `VideoCapture` call
- Check that your firewall allows inbound connections on the DroidCam port

**Video is buffering**
- Set `cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)` after opening the stream to minimize internal frame queuing
- Make sure you are close to your Wi-Fi router

## Possible Extensions

- Swap `yolov8s.pt` for `yolov8n-pose.pt` to detect body poses
- Swap for `yolov8n-seg.pt` for instance segmentation
- Add a people counter or motion-triggered alert
- Log detections to a file for later analysis