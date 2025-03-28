import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist
from playsound import playsound
import time
from datetime import datetime

# DeepFace para la detección de emociones
from deepface import DeepFace

# Supabase
from supabase_config import SUPABASE_URL, SUPABASE_KEY
from supabase import create_client, Client

# -------------------- CONFIGURACIÓN DE OJOS (FATIGA) --------------------
EAR_THRESHOLD = 0.23
EYE_CLOSED_WARNING_SEC = 3.0
EAR_BUFFER_SIZE = 5

eyes_closed_start_time = None
eyes_closed_warning_shown = False
eyes_closed_alarm_triggered = False

TOTAL_EYE_CLOSED = 0.0
ALARM_COUNT = 0
FATIGUE_EVENT_COUNT = 0
ear_buffer = []

# -------------------- CONFIGURACIÓN EMOCIONES --------------------
ANALYZE_EVERY_N_FRAMES = 10  
FACE_MARGIN = 25              
current_emotion_text = "Desconocida"

DETECTOR_BACKEND = "mtcnn"  # Cambia a 'retinaface' si lo deseas

# Mapeo DeepFace -> español (opcional)
emotion_map = {
    "angry": "enojado",
    "disgust": "irritado",
    "fear": "miedo",
    "happy": "feliz",
    "sad": "triste",
    "surprise": "sorprendido",
    "neutral": "neutral"
}

# ------------------- CONFIGURACIÓN SUPABASE -------------------
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
DRIVER_ID = "Conductor_001"
SESSION_START = datetime.now()

supabase.table("drivers").insert({
    "driver_id": DRIVER_ID,
    "session_start": SESSION_START.isoformat()
}).execute()

# ------------------- INICIAR CÁMARA Y DLIB -------------------
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Landmarks índices
LEFT_EYE = list(range(36, 42))
RIGHT_EYE = list(range(42, 48))
LEFT_EYEBROW = list(range(17, 22))   # 17..21
RIGHT_EYEBROW = list(range(22, 27))  # 22..26
MOUTH_OUTER = list(range(48, 55))    # 48..54

# ----------------------------------------------------------------------------
#                          FUNCIONES AUXILIARES
# ----------------------------------------------------------------------------
def calculate_ear(eye_points):
    A = dist.euclidean(eye_points[1], eye_points[5])  # vertical
    B = dist.euclidean(eye_points[2], eye_points[4])  # vertical
    C = dist.euclidean(eye_points[0], eye_points[3])  # horizontal
    return (A + B) / (2.0 * C)

def moving_average(values, window_size):
    if not values:
        return 0.0
    if len(values) < window_size:
        return np.mean(values)
    else:
        return np.mean(values[-window_size:])

def eyebrow_to_eye_distance(eye_points, brow_points):
    """
    Distancia entre el centro del ojo y el centro de la ceja.
    """
    eye_center = np.mean(eye_points, axis=0)
    brow_center = np.mean(brow_points, axis=0)
    return dist.euclidean(eye_center, brow_center)

def mouth_aspect_ratio(mouth_points):
    """
    Aproximación de MAR (Mouth Aspect Ratio),
    horizontal = dist(48,54), vertical = dist(50,53).
    """
    horizontal = dist.euclidean(mouth_points[0], mouth_points[-1])
    vertical = dist.euclidean(mouth_points[2], mouth_points[5])
    if horizontal == 0:
        return 0
    return vertical / horizontal

def mouth_corners_down(mouth_points):
    """
    True si las comisuras (48, 54) están más abajo que el centro (50, 51).
    """
    left_corner = mouth_points[0]   # (48)
    right_corner = mouth_points[-1] # (54)
    mid_top = mouth_points[2]       # (50)
    mid_bottom = mouth_points[3]    # (51)
    avg_mid_y = (mid_top[1] + mid_bottom[1]) / 2.0
    return (left_corner[1] > avg_mid_y and right_corner[1] > avg_mid_y)

# ----------------------------------------------------------------------------
#                           BUCLE PRINCIPAL
# ----------------------------------------------------------------------------
frame_count = 0
last_frame_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error al acceder a la cámara")
        break

    frame_count += 1
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    current_time = time.time()
    frame_duration = current_time - last_frame_time
    last_frame_time = current_time

    face_detected = False

    for face in faces:
        face_detected = True
        shape_dlib = predictor(gray, face)
        landmarks = np.array([[p.x, p.y] for p in shape_dlib.parts()])

        # OJOS
        left_eye_pts = landmarks[LEFT_EYE]
        right_eye_pts = landmarks[RIGHT_EYE]
        ear = (calculate_ear(left_eye_pts) + calculate_ear(right_eye_pts)) / 2.0

        ear_buffer.append(ear)
        smoothed_ear = moving_average(ear_buffer, EAR_BUFFER_SIZE)

        # CEJAS
        left_brow_pts = landmarks[LEFT_EYEBROW]
        right_brow_pts = landmarks[RIGHT_EYEBROW]
        dist_left_brow = eyebrow_to_eye_distance(left_eye_pts, left_brow_pts)
        dist_right_brow = eyebrow_to_eye_distance(right_eye_pts, right_brow_pts)
        avg_brow_dist = (dist_left_brow + dist_right_brow) / 2.0
        brow_is_low = (avg_brow_dist < 15)

        # BOCA
        mouth_points = landmarks[MOUTH_OUTER]  # 48..54
        mar = mouth_aspect_ratio(mouth_points)
        corners_down = mouth_corners_down(mouth_points)

        # Dibujar ojos (opcional)
        cv2.polylines(frame, [left_eye_pts], True, (0, 255, 0), 1)
        cv2.polylines(frame, [right_eye_pts], True, (0, 255, 0), 1)

        # ----------------- DETECCIÓN OJOS CERRADOS -----------------
        if smoothed_ear < EAR_THRESHOLD and not (brow_is_low and smoothed_ear > EAR_THRESHOLD * 0.9):
            TOTAL_EYE_CLOSED += frame_duration

            if eyes_closed_start_time is None:
                eyes_closed_start_time = current_time
                eyes_closed_warning_shown = False
                eyes_closed_alarm_triggered = False

            closed_duration = current_time - eyes_closed_start_time

            if not eyes_closed_warning_shown:
                cv2.putText(frame, "ADVERTENCIA: OJOS CERRADOS",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 255, 255), 2)
                eyes_closed_warning_shown = True

            if closed_duration >= EYE_CLOSED_WARNING_SEC and not eyes_closed_alarm_triggered:
                playsound("alert.wav", block=False)
                ALARM_COUNT += 1
                FATIGUE_EVENT_COUNT += 1
                eyes_closed_alarm_triggered = True

                supabase.table("fatigue_events").insert({
                    "driver_id": DRIVER_ID,
                    "event_time": datetime.now().isoformat(),
                    "alert_type": "somnolencia",
                    "eye_closed_seconds": TOTAL_EYE_CLOSED,
                    "alarm_triggered": True
                }).execute()

                cv2.putText(frame, "ALERTA: OJOS CERRADOS (ALARMA)",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 0, 255), 2)
            else:
                if closed_duration < EYE_CLOSED_WARNING_SEC:
                    cv2.putText(frame, "ADVERTENCIA: OJOS CERRADOS",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (0, 255, 255), 2)
        else:
            eyes_closed_start_time = None
            eyes_closed_warning_shown = False
            eyes_closed_alarm_triggered = False

        # ------------------- DETECCIÓN DE EMOCIONES (DeepFace) -------------------
        if frame_count % ANALYZE_EVERY_N_FRAMES == 0:
            x1, y1 = face.left(), face.top()
            x2, y2 = face.right(), face.bottom()

            # Añadir margen
            x1m = max(0, x1 - FACE_MARGIN)
            y1m = max(0, y1 - FACE_MARGIN)
            x2m = min(frame.shape[1], x2 + FACE_MARGIN)
            y2m = min(frame.shape[0], y2 + FACE_MARGIN)

            roi_face = frame[y1m:y2m, x1m:x2m]

            try:
                analysis = DeepFace.analyze(
                    roi_face,
                    actions=['emotion'],
                    enforce_detection=False,
                    detector_backend=DETECTOR_BACKEND
                )
                if isinstance(analysis, list):
                    analysis = analysis[0]

                emotions_dict = analysis['emotion']
                # Emoción dominante original
                dom_emo = analysis['dominant_emotion']

                # -- Ajuste heurístico --
                # 1) Si cejas muy bajas => subir 'angry'
                if brow_is_low and 'angry' in emotions_dict:
                    emotions_dict['angry'] *= 1.4  # 40% extra

                # 2) Si boca muy cerrada (mar<0.2) y cejas bajas => subir 'angry' más
                if mar < 0.2 and brow_is_low and 'angry' in emotions_dict:
                    emotions_dict['angry'] *= 1.4

                # 3) Si corners_down => subir 'sad'
                if corners_down and 'sad' in emotions_dict:
                    emotions_dict['sad'] *= 1.6  # 60% extra

                # 4) Descontar 'angry' si corners_down para no confundir con "triste"
                if corners_down and 'angry' in emotions_dict:
                    emotions_dict['angry'] *= 0.8  # reduce 20%

                # 5) Otras reglas si deseas para 'fear' (boca abierta)
                if mar > 0.4 and 'fear' in emotions_dict:
                    emotions_dict['fear'] *= 1.3

                # Recalcular la dominante tras los ajustes
                sorted_emotions = sorted(emotions_dict.items(), key=lambda x: x[1], reverse=True)
                primary_emo, primary_val = sorted_emotions[0]
                secondary_emo, secondary_val = sorted_emotions[1] if len(sorted_emotions) > 1 else (None, 0)

                # Si 'angry','sad','disgust' > 20% => priorizar
                if emotions_dict.get('angry', 0) > 20:
                    final_emo = 'angry'
                    final_val = emotions_dict['angry']
                elif emotions_dict.get('sad', 0) > 20:
                    final_emo = 'sad'
                    final_val = emotions_dict['sad']
                elif emotions_dict.get('disgust', 0) > 20:
                    final_emo = 'disgust'
                    final_val = emotions_dict['disgust']
                else:
                    # Revisar segunda emoción si está cerca
                    if (primary_val - secondary_val) < 5 and secondary_emo in ['angry','sad','disgust']:
                        final_emo = secondary_emo
                        final_val = secondary_val
                    else:
                        final_emo = primary_emo
                        final_val = primary_val

                emotion_es = emotion_map.get(final_emo, final_emo)
                current_emotion_text = f"{emotion_es} ({final_val:.1f}%)"

                # Insertar en Supabase
                supabase.table("emotions").insert({
                    "driver_id": DRIVER_ID,
                    "event_time": datetime.now().isoformat(),
                    "emotion": emotion_es
                }).execute()

            except Exception:
                pass

    # Si no hay rostro
    if not face_detected:
        cv2.putText(frame, "No se detecta rostro",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 255), 2)

    # EAR en pantalla
    if ear_buffer:
        avg_ear = moving_average(ear_buffer, EAR_BUFFER_SIZE)
        cv2.putText(frame,
                    f"EAR: {avg_ear:.2f}",
                    (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2)

    # Emoción actual
    cv2.putText(frame,
                f"Emocion: {current_emotion_text}",
                (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2)

    cv2.imshow("SafeDrive - Emociones Ajustadas", frame)

    # Salir con 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --------------------------- FIN DE SESIÓN ---------------------------
SESSION_END = datetime.now()
supabase.table("drivers").update({
    "session_end": SESSION_END.isoformat()
}).eq("driver_id", DRIVER_ID).execute()

supabase.table("session_summary").insert({
    "driver_id": DRIVER_ID,
    "session_start": SESSION_START.isoformat(),
    "session_end": SESSION_END.isoformat(),
    "total_alarms": ALARM_COUNT,
    "total_eye_closed": TOTAL_EYE_CLOSED,
    "fatigue_events": FATIGUE_EVENT_COUNT
}).execute()

cap.release()
cv2.destroyAllWindows()