from flask import Flask, jsonify, request, Response
from flask_cors import CORS
import requests
import datetime
import threading
import subprocess
import os
import signal
import time

app = Flask(__name__)
CORS(app)

detections_log = []
alerts_log = []
recordings_log = []
cameras = {
    "front_porch": {
        "name": "Front Porch",
        "input": "rpi",
        "status": "active"
    }
}

OLLAMA_URL = "http://localhost:11434/api/generate"
ALERT_LABELS = ["person", "car", "truck", "knife", "backpack"]
ANALYSIS_COOLDOWN = 30
last_analysis_time = None

# Recording state
RECORDINGS_DIR = os.path.expanduser("~/argus/recordings")
os.makedirs(RECORDINGS_DIR, exist_ok=True)
recording_process = None
recording_start_time = None
RECORDING_DURATION = 30  # seconds per clip


def analyze_with_ollama(detection):
    global last_analysis_time
    now = datetime.datetime.now()

    if last_analysis_time and (now - last_analysis_time).seconds < ANALYSIS_COOLDOWN:
        return

    last_analysis_time = now
    prompt = (
        f"You are Argus, an AI security camera system. "
        f"A '{detection['label']}' was detected with {detection['confidence']*100:.0f}% confidence "
        f"at camera '{detection.get('camera', 'front_porch')}' at {detection['timestamp']}. "
        f"Briefly assess if this is normal or suspicious and recommend action. "
        f"Keep response under 50 words."
    )

    try:
        resp = requests.post(OLLAMA_URL, json={
            "model": "tinyllama",
            "prompt": prompt,
            "stream": False
        }, timeout=30)
        analysis = resp.json().get("response", "Analysis unavailable")
    except Exception as e:
        analysis = f"Analysis failed: {str(e)}"

    alert = {
        "type": "ai_analysis",
        "detection": detection,
        "analysis": analysis,
        "timestamp": now.isoformat(),
    }
    alerts_log.append(alert)
    if len(alerts_log) > 200:
        alerts_log.pop(0)


def generate_mjpeg():
    """Stream MJPEG from Pi camera using libcamera"""
    process = subprocess.Popen(
        [
            "libcamera-vid",
            "--codec", "mjpeg",
            "--width", "640",
            "--height", "480",
            "--framerate", "15",
            "--timeout", "0",
            "--nopreview",
            "-o", "-"
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL
    )

    boundary = b"--frame\r\n"
    buf = b""

    try:
        while True:
            chunk = process.stdout.read(4096)
            if not chunk:
                break
            buf += chunk

            # Find JPEG frames (start with FFD8, end with FFD9)
            while True:
                start = buf.find(b'\xff\xd8')
                end = buf.find(b'\xff\xd9', start + 2) if start != -1 else -1

                if start == -1 or end == -1:
                    break

                frame = buf[start:end + 2]
                buf = buf[end + 2:]

                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" +
                    frame +
                    b"\r\n"
                )
    finally:
        process.terminate()


@app.route("/api/stream")
def video_stream():
    """Live MJPEG video stream from Front Porch camera"""
    return Response(
        generate_mjpeg(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/api/cameras", methods=["GET"])
def get_cameras():
    return jsonify(cameras)


@app.route("/api/detections", methods=["GET"])
def get_detections():
    limit = request.args.get("limit", 50, type=int)
    camera = request.args.get("camera", None)
    if camera:
        filtered = [d for d in detections_log if d.get("camera") == camera]
        return jsonify(filtered[-limit:])
    return jsonify(detections_log[-limit:])


@app.route("/api/detections", methods=["POST"])
def add_detection():
    data = request.json
    data["timestamp"] = datetime.datetime.now().isoformat()
    if "camera" not in data:
        data["camera"] = "front_porch"
    detections_log.append(data)

    if len(detections_log) > 500:
        detections_log.pop(0)

    if data.get("label") in ALERT_LABELS:
        alert = {
            "type": "detection",
            "label": data["label"],
            "confidence": data["confidence"],
            "camera": data.get("camera", "front_porch"),
            "timestamp": data["timestamp"],
        }
        alerts_log.append(alert)
        if len(alerts_log) > 200:
            alerts_log.pop(0)

        threading.Thread(target=analyze_with_ollama, args=(data,), daemon=True).start()

    return jsonify({"status": "ok"})


@app.route("/api/alerts", methods=["GET"])
def get_alerts():
    limit = request.args.get("limit", 20, type=int)
    return jsonify(alerts_log[-limit:])


@app.route("/api/alerts/clear", methods=["POST"])
def clear_alerts():
    alerts_log.clear()
    return jsonify({"status": "cleared"})


@app.route("/api/analyze", methods=["POST"])
def analyze():
    data = request.json
    prompt = data.get("prompt", f"Analyze this detection: {data}")

    try:
        resp = requests.post(OLLAMA_URL, json={
            "model": "tinyllama",
            "prompt": prompt,
            "stream": False
        }, timeout=30)
        return jsonify(resp.json())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/record", methods=["POST"])
def trigger_recording():
    """Trigger a video recording clip when person detected."""
    global recording_process, recording_start_time

    data = request.json or {}
    trigger = data.get("trigger", "manual")
    confidence = data.get("confidence", 0)
    frame = data.get("frame", 0)

    # If already recording, just extend by noting the trigger
    if recording_process and recording_process.poll() is None:
        return jsonify({
            "status": "already_recording",
            "started": recording_start_time,
        })

    # Start a new recording
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"argus_{trigger}_{timestamp}.h264"
    filepath = os.path.join(RECORDINGS_DIR, filename)

    try:
        recording_process = subprocess.Popen(
            [
                "libcamera-vid",
                "--width", "1280",
                "--height", "720",
                "--framerate", "30",
                "--timeout", str(RECORDING_DURATION * 1000),
                "--nopreview",
                "-o", filepath,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        recording_start_time = datetime.datetime.now().isoformat()

        rec_entry = {
            "filename": filename,
            "trigger": trigger,
            "confidence": confidence,
            "frame": frame,
            "started": recording_start_time,
            "duration": RECORDING_DURATION,
        }
        recordings_log.append(rec_entry)
        if len(recordings_log) > 100:
            recordings_log.pop(0)

        # Also add an alert for recording
        alerts_log.append({
            "type": "recording",
            "label": "person",
            "confidence": confidence,
            "camera": "front_porch",
            "timestamp": recording_start_time,
            "analysis": f"Recording triggered by {trigger} detection (confidence: {confidence}). Saving {RECORDING_DURATION}s clip.",
        })

        print(f"[RECORD] Started recording: {filename}")
        return jsonify({"status": "recording_started", "filename": filename})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/record/stop", methods=["POST"])
def stop_recording():
    """Stop the current recording early."""
    global recording_process
    if recording_process and recording_process.poll() is None:
        recording_process.terminate()
        recording_process = None
        return jsonify({"status": "stopped"})
    return jsonify({"status": "not_recording"})


@app.route("/api/recordings", methods=["GET"])
def get_recordings():
    """List recent recordings."""
    limit = request.args.get("limit", 20, type=int)
    return jsonify(recordings_log[-limit:])


@app.route("/api/status", methods=["GET"])
def status():
    is_recording = recording_process is not None and recording_process.poll() is None
    return jsonify({
        "status": "running",
        "detections_count": len(detections_log),
        "alerts_count": len(alerts_log),
        "recordings_count": len(recordings_log),
        "is_recording": is_recording,
        "alert_labels": ALERT_LABELS,
        "cameras": cameras,
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
