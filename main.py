# main.py
import cv2  
import dlib
import numpy as np
from scipy.spatial import distance as dist
from playsound import playsound
import time
from datetime import datetime

from supabase_config import SUPABASE_URL, SUPABASE_KEY
from supabase import create_client, Client

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Funciones de cálculo
def calculate_ear(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def calculate_head_tilt(landmarks):
    eye_center = ((landmarks[36][0] + landmarks[45][0]) // 2, (landmarks[36][1] + landmarks[45][1]) // 2)
    chin = landmarks[8]  # Punto de la barbilla
    delta_x = chin[0] - eye_center[0]
    delta_y = chin[1] - eye_center[1]
    angle = np.degrees(np.arctan2(delta_y, delta_x))
    return angle

# Calibración del EAR
def calibrate_ear(cap, detector, predictor, duration=5):
    ear_open, ear_closed = [], []
    start_time = time.time()
    print("Mira la cámara con ojos abiertos por 5 segundos...")
    while time.time() - start_time < duration:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        faces = detector(gray)
        for face in faces:
            landmarks = predictor(gray, face)
            landmarks = np.array([[p.x, p.y] for p in landmarks.parts()])
            left_eye = landmarks[LEFT_EYE]
            right_eye = landmarks[RIGHT_EYE]
            ear = (calculate_ear(left_eye) + calculate_ear(right_eye)) / 2.0
            ear_open.append(ear)
        cv2.imshow("Calibración", frame)
        cv2.waitKey(1)
    
    print("Ahora cierra los ojos por 5 segundos...")
    start_time = time.time()
    while time.time() - start_time < duration:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        faces = detector(gray)
        for face in faces:
            landmarks = predictor(gray, face)
            landmarks = np.array([[p.x, p.y] for p in landmarks.parts()])
            left_eye = landmarks[LEFT_EYE]
            right_eye = landmarks[RIGHT_EYE]
            ear = (calculate_ear(left_eye) + calculate_ear(right_eye)) / 2.0
            ear_closed.append(ear)
        cv2.imshow("Calibración", frame)
        cv2.waitKey(1)
    
    cv2.destroyWindow("Calibración")
    avg_open = np.mean(ear_open)
    avg_closed = np.mean(ear_closed)
    return (avg_open + avg_closed) / 2  # Umbral intermedio

# Configuración inicial
EAR_CONSEC_FRAMES = 20
TILT_THRESHOLD = 70  # Inclinación extrema
TILT_CONSEC_FRAMES = 20
COUNTER_EAR = 0
COUNTER_TILT = 0
ALARM_COUNT = 0
TOTAL_EYE_CLOSED = 0
FATIGUE_EVENT_COUNT = 0
ALARM_ACTIVE = False
ear_history = []
tilt_history = []

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
LEFT_EYE = list(range(36, 42))
RIGHT_EYE = list(range(42, 48))

# Iniciar captura y calibrar EAR
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
EAR_THRESHOLD = calibrate_ear(cap, detector, predictor)
print(f"EAR_THRESHOLD calibrado: {EAR_THRESHOLD}")

DRIVER_ID = "Conductor_001"
SESSION_START = datetime.now()
supabase.table("drivers").insert({
    "driver_id": DRIVER_ID,
    "session_start": SESSION_START.isoformat()
}).execute()

last_ear_check = time.time()
last_frame_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)  # Normalizar iluminación
    gray = cv2.GaussianBlur(gray, (5, 5), 0)  # Reducir ruido
    faces = detector(gray)

    current_time = time.time()
    frame_duration = current_time - last_frame_time
    last_frame_time = current_time

    for face in faces:
        landmarks = predictor(gray, face)
        landmarks = np.array([[p.x, p.y] for p in landmarks.parts()])

        left_eye = landmarks[LEFT_EYE]
        right_eye = landmarks[RIGHT_EYE]
        ear = (calculate_ear(left_eye) + calculate_ear(right_eye)) / 2.0
        tilt_angle = calculate_head_tilt(landmarks)

        # Promedio móvil para suavizar datos
        ear_history.append(ear)
        tilt_history.append(tilt_angle)
        if len(ear_history) > 5:
            ear_history.pop(0)
            tilt_history.pop(0)
        avg_ear = np.mean(ear_history)
        avg_tilt = np.mean(tilt_history)

        cv2.polylines(frame, [left_eye], True, (0, 255, 0), 1)
        cv2.polylines(frame, [right_eye], True, (0, 255, 0), 1)

        # Detección de fatiga ocular
        if avg_ear < EAR_THRESHOLD:
            if COUNTER_EAR == 0:
                eye_close_start = current_time
            COUNTER_EAR += 1
            TOTAL_EYE_CLOSED += frame_duration
            if COUNTER_EAR >= EAR_CONSEC_FRAMES and (current_time - eye_close_start) > 0.5:  # Filtrar parpadeos
                if current_time - last_ear_check >= 1:
                    playsound("alert.wav", block=False)
                    ALARM_COUNT += 1
                    FATIGUE_EVENT_COUNT += 1
                    last_ear_check = current_time
                    ALARM_ACTIVE = True
                    cv2.putText(frame, "ALERTA: OJOS CERRADOS", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    supabase.table("fatigue_events").insert({
                        "driver_id": DRIVER_ID,
                        "event_time": datetime.now().isoformat(),
                        "alert_type": "somnolencia",
                        "eye_closed_seconds": TOTAL_EYE_CLOSED,
                        "alarm_triggered": True
                    }).execute()
        else:
            COUNTER_EAR = 0
            if ALARM_ACTIVE:
                ALARM_ACTIVE = False
                last_ear_check = time.time() - 1

        # Detección de inclinación extrema
        if abs(avg_tilt) > TILT_THRESHOLD and avg_ear < 0.3:  # Validación cruzada con EAR
            if COUNTER_TILT == 0:
                tilt_start = current_time
            COUNTER_TILT += 1
            if COUNTER_TILT >= TILT_CONSEC_FRAMES and (current_time - tilt_start) > 1:  # Sostenida por 1s
                FATIGUE_EVENT_COUNT += 1
                cv2.putText(frame, "ALERTA: CABEZA MUY INCLINADA", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                supabase.table("fatigue_events").insert({
                    "driver_id": DRIVER_ID,
                    "event_time": datetime.now().isoformat(),
                    "alert_type": "cabeza muy inclinada",
                    "eye_closed_seconds": 0.0,
                    "alarm_triggered": False
                }).execute()
        else:
            COUNTER_TILT = 0

        cv2.putText(frame, f"EAR: {avg_ear:.2f}", (300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Tilt: {avg_tilt:.2f}", (300, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Sistema de Deteccion de Fatiga", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

SESSION_END = datetime.now()
supabase.table("drivers").update({
    "session_end": SESSION_END.isoformat()
}).eq("driver_id", DRIVER_ID).execute()

supabase.table("session_summary").insert({
    "driver_id": DRIVER_ID,
    "session_start": SESSION_START.isoformat(),
    "total_alarms": ALARM_COUNT,
    "total_eye_closed": TOTAL_EYE_CLOSED,
    "fatigue_events": FATIGUE_EVENT_COUNT
}).execute()

cap.release()
cv2.destroyAllWindows()