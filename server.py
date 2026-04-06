from flask import Flask, jsonify, request, Response
from flask_cors import CORS
from prometheus_client import (
    Counter, Gauge, Histogram, Summary, Info,
    generate_latest, CONTENT_TYPE_LATEST, CollectorRegistry, REGISTRY,
    start_http_server as prom_start_http_server,
)
import requests
import datetime
import threading
import subprocess
import os
import signal
import time
import psutil

app = Flask(__name__)
CORS(app)

# ─── Prometheus Metrics ─────────────────────────────────────────────
DETECTION_COUNTER = Counter(
    'argus_detections_total',
    'Total object detections',
    ['label', 'camera']
)
ALERT_COUNTER = Counter(
    'argus_alerts_total',
    'Total alerts generated',
    ['type', 'camera']
)
DETECTION_CONFIDENCE = Histogram(
    'argus_detection_confidence',
    'Detection confidence distribution',
    ['label'],
    buckets=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]
)
ACTIVE_CAMERAS = Gauge(
    'argus_active_cameras',
    'Number of active cameras'
)
IS_RECORDING = Gauge(
    'argus_is_recording',
    'Whether the system is currently recording (1=yes)'
)
RECORDINGS_TOTAL = Counter(
    'argus_recordings_total',
    'Total recordings triggered',
    ['trigger']
)
OLLAMA_LATENCY = Histogram(
    'argus_ollama_analysis_seconds',
    'Ollama analysis response time',
    buckets=[0.5, 1, 2, 5, 10, 20, 30]
)
OLLAMA_ERRORS = Counter(
    'argus_ollama_errors_total',
    'Total Ollama analysis failures'
)
FRAME_POST_COUNTER = Counter(
    'argus_frames_received_total',
    'Total frames received from detection pipeline'
)
# Federated learning metrics (ready for FL module)
FL_ROUND = Gauge('argus_fl_current_round', 'Current federated learning round')
FL_LOCAL_LOSS = Gauge('argus_fl_local_loss', 'Local training loss from last FL round')
FL_MODEL_VERSION = Gauge('argus_fl_model_version', 'Current model version/generation')
FL_PEERS = Gauge('argus_fl_connected_peers', 'Number of connected FL peers')
# WiFi CSI metrics (ready for CSI module)
WIFI_CSI_MOTION = Gauge('argus_wifi_csi_motion_detected', 'WiFi CSI motion detected (1=yes)')
WIFI_CSI_AMPLITUDE = Gauge('argus_wifi_csi_mean_amplitude', 'Mean CSI subcarrier amplitude')
WIFI_CSI_PERSONS = Gauge('argus_wifi_csi_estimated_persons', 'Estimated person count from WiFi CSI')
# System metrics
SYSTEM_CPU = Gauge('argus_system_cpu_percent', 'System CPU usage percent')
SYSTEM_MEMORY = Gauge('argus_system_memory_percent', 'System memory usage percent')
SYSTEM_TEMP = Gauge('argus_system_cpu_temp_celsius', 'CPU temperature in Celsius')
NODE_INFO = Info('argus_node', 'Argus node information')


def collect_system_metrics():
    """Background thread to collect system metrics every 5 seconds."""
    while True:
        try:
            SYSTEM_CPU.set(psutil.cpu_percent(interval=1))
            SYSTEM_MEMORY.set(psutil.virtual_memory().percent)
            # CPU temp — works on Pi, may not on laptop
            try:
                temps = psutil.sensors_temperatures()
                if temps:
                    for name, entries in temps.items():
                        if entries:
                            SYSTEM_TEMP.set(entries[0].current)
                            break
            except (AttributeError, KeyError):
                pass
        except Exception:
            pass
        time.sleep(5)


# Start system metrics collection
_sys_thread = threading.Thread(target=collect_system_metrics, daemon=True)
_sys_thread.start()

import socket
NODE_INFO.info({
    'hostname': socket.gethostname(),
    'version': '0.3.0',
    'role': 'edge_node',
})

detections_log = []
alerts_log = []
recordings_log = []

# Shared frame buffer for live stream (fed by detection pipeline)
latest_frame = None
frame_lock = threading.Lock()
frame_event = threading.Event()
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
        with OLLAMA_LATENCY.time():
            resp = requests.post(OLLAMA_URL, json={
                "model": "tinyllama",
                "prompt": prompt,
                "stream": False
            }, timeout=30)
        analysis = resp.json().get("response", "Analysis unavailable")
    except Exception as e:
        OLLAMA_ERRORS.inc()
        analysis = f"Analysis failed: {str(e)}"

    alert = {
        "type": "ai_analysis",
        "detection": detection,
        "analysis": analysis,
        "timestamp": now.isoformat(),
    }
    alerts_log.append(alert)
    ALERT_COUNTER.labels(type="ai_analysis", camera=detection.get("camera", "front_porch")).inc()
    if len(alerts_log) > 200:
        alerts_log.pop(0)


@app.route("/metrics")
def metrics():
    """Prometheus metrics endpoint."""
    ACTIVE_CAMERAS.set(sum(1 for c in cameras.values() if c.get("status") == "active"))
    is_rec = 1 if (recording_process and recording_process.poll() is None) else 0
    IS_RECORDING.set(is_rec)
    return Response(generate_latest(REGISTRY), mimetype=CONTENT_TYPE_LATEST)


@app.route("/api/frame", methods=["POST"])
def post_frame():
    """Receive a JPEG frame from the detection pipeline."""
    global latest_frame
    FRAME_POST_COUNTER.inc()
    if request.content_type and "image/jpeg" in request.content_type:
        frame_data = request.data
    else:
        frame_data = request.data
    if not frame_data:
        return jsonify({"error": "no frame data"}), 400
    with frame_lock:
        latest_frame = frame_data
    frame_event.set()
    return jsonify({"status": "ok"})


def generate_mjpeg():
    """Yield MJPEG frames from the shared buffer (fed by detection pipeline)."""
    while True:
        frame_event.wait(timeout=2.0)
        with frame_lock:
            frame = latest_frame
        if frame is None:
            continue
        frame_event.clear()
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" +
            frame +
            b"\r\n"
        )


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

    # Prometheus: track detection
    camera = data.get("camera", "front_porch")
    label = data.get("label", "unknown")
    DETECTION_COUNTER.labels(label=label, camera=camera).inc()
    if "confidence" in data:
        DETECTION_CONFIDENCE.labels(label=label).observe(data["confidence"])

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
        ALERT_COUNTER.labels(type="detection", camera=camera).inc()
        if len(alerts_log) > 200:
            alerts_log.pop(0)

        threading.Thread(target=analyze_with_ollama, args=(data,), daemon=True).start()

    # Feed to federated learning
    if _fl_client:
        feed_detection(data)

    # Feed to sensor fusion
    if data.get("label") == "person":
        update_camera_state(
            detected=True,
            label=data.get("label", ""),
            confidence=data.get("confidence", 0.0),
        )

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
                "rpicam-vid",
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

        RECORDINGS_TOTAL.labels(trigger=trigger).inc()
        IS_RECORDING.set(1)
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
        "metrics_endpoint": "/metrics",
        "fl_round": FL_ROUND._value.get(),
        "fl_peers": FL_PEERS._value.get(),
    })


# ─── Federated Learning Integration ─────────────────────────────────
from federated import init_federated, feed_detection, get_fl_status

_fl_client = None


def start_fl():
    """Initialize federated learning subsystem."""
    global _fl_client
    try:
        _fl_client = init_federated(prometheus_metrics={
            "fl_round": FL_ROUND,
            "fl_loss": FL_LOCAL_LOSS,
            "fl_version": FL_MODEL_VERSION,
            "fl_peers": FL_PEERS,
        })
        print("[FL] Federated learning initialized")
    except Exception as e:
        print(f"[FL] Init failed (non-fatal): {e}")


@app.route("/api/fl/status", methods=["GET"])
def fl_status():
    return jsonify(get_fl_status())


# ─── WiFi CSI Integration ────────────────────────────────────────────
from wifi_csi import init_wifi_csi, get_csi_status, get_recent_detections as get_csi_detections
from sensor_fusion import (
    get_fused_status, get_fusion_history,
    update_wifi_state, update_camera_state,
)

_csi_engine = None


def start_wifi_csi():
    """Initialize WiFi CSI subsystem."""
    global _csi_engine
    try:
        _csi_engine = init_wifi_csi(prometheus_metrics={
            "motion": WIFI_CSI_MOTION,
            "amplitude": WIFI_CSI_AMPLITUDE,
            "persons": WIFI_CSI_PERSONS,
        })
        print("[CSI] WiFi CSI engine initialized")

        # Start a thread to bridge CSI events to sensor fusion
        def _csi_to_fusion():
            while True:
                try:
                    if _csi_engine:
                        update_wifi_state(
                            motion=_csi_engine.motion_detected,
                            confidence=0.7 if _csi_engine.motion_detected else 0.0,
                            activity=_csi_engine.current_activity,
                            persons=_csi_engine.person_count,
                        )
                except Exception:
                    pass
                time.sleep(0.5)

        threading.Thread(target=_csi_to_fusion, daemon=True).start()
    except Exception as e:
        print(f"[CSI] Init failed (non-fatal): {e}")


@app.route("/api/wifi/status", methods=["GET"])
def wifi_status():
    return jsonify(get_csi_status())


@app.route("/api/wifi/detections", methods=["GET"])
def wifi_detections():
    limit = request.args.get("limit", 20, type=int)
    return jsonify(get_csi_detections(limit))


@app.route("/api/fusion/status", methods=["GET"])
def fusion_status():
    return jsonify(get_fused_status())


@app.route("/api/fusion/history", methods=["GET"])
def fusion_history():
    limit = request.args.get("limit", 50, type=int)
    return jsonify(get_fusion_history(limit))


if __name__ == "__main__":
    start_fl()
    start_wifi_csi()
    app.run(host="0.0.0.0", port=5000)
