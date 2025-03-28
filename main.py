# main.py
import cv2  
import dlib
import numpy as np
from scipy.spatial import distance as dist
from playsound import playsound
import time
from datetime import datetime

# Crear cliente de Supabase
from supabase_config import SUPABASE_URL, SUPABASE_KEY
from supabase import create_client, Client

# Crear cliente de Supabase
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Funciones de cálculo
def calculate_ear(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def calculate_head_tilt(landmarks):
    nose = landmarks[30]  # Punta de la nariz
    left_eye = landmarks[36]  # Esquina externa del ojo izquierdo
    right_eye = landmarks[45]  # Esquina externa del ojo derecho
    eye_center = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)
    delta_x = nose[0] - eye_center[0]
    delta_y = nose[1] - eye_center[1]
    angle = np.degrees(np.arctan2(delta_y, delta_x))
    return angle

# Configuración inicial
EAR_THRESHOLD = 0.25
EAR_CONSEC_FRAMES = 20
TILT_THRESHOLD = 70  # Aumentado a 70 grados para inclinación extrema (casi pecho)
TILT_CONSEC_FRAMES = 20
COUNTER_EAR = 0
COUNTER_TILT = 0
ALARM_COUNT = 0
TOTAL_EYE_CLOSED = 0
FATIGUE_EVENT_COUNT = 0
ALARM_ACTIVE = False

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
LEFT_EYE = list(range(36, 42))
RIGHT_EYE = list(range(42, 48))

DRIVER_ID = "Conductor_001"
SESSION_START = datetime.now()
supabase.table("drivers").insert({
    "driver_id": DRIVER_ID,
    "session_start": SESSION_START.isoformat()
}).execute()

cap = cv2.VideoCapture(0)
last_ear_check = time.time()
last_frame_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    current_time = time.time()
    frame_duration = current_time - last_frame_time
    last_frame_time = current_time

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
            TOTAL_EYE_CLOSED += frame_duration
            if COUNTER_EAR >= EAR_CONSEC_FRAMES:
                current_time = time.time()
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

        # Ajuste para inclinación extrema
        if abs(tilt_angle) > TILT_THRESHOLD:  # Solo se activa con inclinación mayor a 70 grados
            COUNTER_TILT += 1
            if COUNTER_TILT >= TILT_CONSEC_FRAMES and COUNTER_EAR < EAR_CONSEC_FRAMES:
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

        cv2.putText(frame, f"EAR: {ear:.2f}", (300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Tilt: {tilt_angle:.2f}", (300, 60),
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