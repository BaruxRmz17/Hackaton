from flask import Flask, Response, request, jsonify, render_template
from flask_socketio import SocketIO, emit
import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist
from playsound import playsound
import time
import sqlite3
from datetime import datetime
import base64

app = Flask(__name__)
socketio = SocketIO(app)

# Funciones de cálculo
def calculate_ear(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def calculate_head_tilt(landmarks):
    nose = landmarks[30]
    left_eye = landmarks[36]
    right_eye = landmarks[45]
    eye_center = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)
    delta_x = nose[0] - eye_center[0]
    delta_y = nose[1] - eye_center[1]
    angle = np.degrees(np.arctan2(delta_y, delta_x))
    return angle

# Configuración de la base de datos
def init_db():
    conn = sqlite3.connect('driver_monitoring.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS drivers (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT,
                    phone TEXT,
                    license_plate TEXT,
                    photo TEXT,
                    timestamp TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS emotional_responses (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    driver_id INTEGER,
                    question TEXT,
                    answer TEXT,
                    FOREIGN KEY(driver_id) REFERENCES drivers(id))''')
    c.execute('''CREATE TABLE IF NOT EXISTS fatigue_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    driver_id INTEGER,
                    timestamp TEXT,
                    ear REAL,
                    tilt REAL,
                    alert TEXT,
                    FOREIGN KEY(driver_id) REFERENCES drivers(id))''')
    conn.commit()
    conn.close()

# Guardar datos del chofer
def save_driver(name, phone, license_plate, photo, emotions):
    conn = sqlite3.connect('driver_monitoring.db')
    c = conn.cursor()
    timestamp = datetime.now().isoformat()
    c.execute("INSERT INTO drivers (name, phone, license_plate, photo, timestamp) VALUES (?, ?, ?, ?, ?)",
              (name, phone, license_plate, photo, timestamp))
    driver_id = c.lastrowid
    for question, answer in emotions.items():
        c.execute("INSERT INTO emotional_responses (driver_id, question, answer) VALUES (?, ?, ?)",
                  (driver_id, question, answer))
    conn.commit()
    conn.close()
    return driver_id

# Guardar eventos de fatiga
def save_fatigue_event(driver_id, ear, tilt, alert):
    conn = sqlite3.connect('driver_monitoring.db')
    c = conn.cursor()
    timestamp = datetime.now().isoformat()
    c.execute("INSERT INTO fatigue_events (driver_id, timestamp, ear, tilt, alert) VALUES (?, ?, ?, ?, ?)",
              (driver_id, timestamp, ear, tilt, alert))
    conn.commit()
    conn.close()

# Configuración inicial
EAR_THRESHOLD = 0.25
EAR_CONSEC_FRAMES = 20
TILT_THRESHOLD = 30
TILT_CONSEC_FRAMES = 20
COUNTER_EAR = 0
COUNTER_TILT = 0
last_ear_check = time.time()

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
LEFT_EYE = list(range(36, 42))
RIGHT_EYE = list(range(42, 48))

cap = cv2.VideoCapture(0)
current_driver_id = None

# Capturar foto
@app.route('/capture_photo', methods=['GET'])
def capture_photo():
    ret, frame = cap.read()
    if ret:
        ret, buffer = cv2.imencode('.jpg', frame)
        photo = base64.b64encode(buffer).decode('utf-8')
        return jsonify({'photo': photo})
    return jsonify({'error': 'No se pudo capturar la foto'}), 500

# Registro del chofer
@app.route('/register', methods=['POST'])
def register_driver():
    data = request.json
    name = data.get('name')
    phone = data.get('phone')
    license_plate = data.get('license_plate')
    photo = data.get('photo')
    emotions = data.get('emotions', {})
    global current_driver_id
    current_driver_id = save_driver(name, phone, license_plate, photo, emotions)
    return jsonify({'message': 'Chofer registrado', 'driver_id': current_driver_id})

# Video feed para monitoreo
def generate_frames():
    global COUNTER_EAR, COUNTER_TILT, last_ear_check, current_driver_id
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 480))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        alert_message = None
        ear_value = None
        tilt_value = None

        for face in faces:
            landmarks = predictor(gray, face)
            landmarks = np.array([[p.x, p.y] for p in landmarks.parts()])

            left_eye = landmarks[LEFT_EYE]
            right_eye = landmarks[RIGHT_EYE]
            left_ear = calculate_ear(left_eye)
            right_ear = calculate_ear(right_eye)
            ear = (left_ear + right_ear) / 2.0
            tilt_angle = calculate_head_tilt(landmarks)

            cv2.polylines(frame, [left_eye], True, (0, 255, 0), 1)
            cv2.polylines(frame, [right_eye], True, (0, 255, 0), 1)

            if ear < EAR_THRESHOLD:
                COUNTER_EAR += 1
                if COUNTER_EAR >= EAR_CONSEC_FRAMES:
                    current_time = time.time()
                    if current_time - last_ear_check > 1:
                        playsound("alert.wav", block=False)
                        last_ear_check = current_time
                    alert_message = "OJOS CERRADOS"
                    cv2.putText(frame, "ALERTA: OJOS CERRADOS", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                COUNTER_EAR = 0

            if abs(tilt_angle) > TILT_THRESHOLD:
                COUNTER_TILT += 1
                if COUNTER_TILT >= TILT_CONSEC_FRAMES and COUNTER_EAR < EAR_CONSEC_FRAMES:
                    alert_message = "CABEZA INCLINADA"
                    cv2.putText(frame, "ALERTA: CABEZA INCLINADA", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                COUNTER_TILT = 0

            ear_value = ear
            tilt_value = tilt_angle

            cv2.putText(frame, f"EAR: {ear:.2f}", (300, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Tilt: {tilt_angle:.2f}", (300, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            if current_driver_id and (ear_value is not None or tilt_value is not None):
                save_fatigue_event(current_driver_id, ear_value, tilt_value, alert_message)

        socketio.emit('update', {
            'ear': round(ear_value, 2) if ear_value else 0,
            'tilt': round(tilt_value, 2) if tilt_value else 0,
            'alert': alert_message
        })
        socketio.sleep(0.1)

        ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return "Backend running. Connect from React frontend."

if __name__ == '__main__':
    init_db()
    socketio.run(app, host='0.0.0.0', port=8080)