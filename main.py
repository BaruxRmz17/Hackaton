import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist
from playsound import playsound
import time
from datetime import datetime

# DeepFace para reconocimiento de emociones
# Instalar con: pip install deepface
from deepface import DeepFace

# Importar credenciales de Supabase
from supabase_config import SUPABASE_URL, SUPABASE_KEY
from supabase import create_client, Client

# ----------------------------------------------------------------------------
#                           CONFIGURACIÓN / CONSTANTES
# ----------------------------------------------------------------------------

# Umbral de EAR para considerar ojos cerrados
EAR_THRESHOLD = 0.25  
# Tiempo que se permite con ojos cerrados antes de sonar la alarma (segundos)
EYE_CLOSED_WARNING_SEC = 3.0  

# Buffer para suavizar EAR (promedio móvil simple)
EAR_BUFFER_SIZE = 5  

# Variables para FATIGA / SOMNOLENCIA
eyes_closed_start_time = None  # Cuándo se detectó que los ojos se cerraron
eyes_closed_warning_shown = False
eyes_closed_alarm_triggered = False

# Contadores generales
TOTAL_EYE_CLOSED = 0.0  # Acumulado de tiempo con ojos cerrados
ALARM_COUNT = 0
FATIGUE_EVENT_COUNT = 0

# Buffer para suavizar EAR
ear_buffer = []

# Crear cliente de Supabase
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# IDs y tiempos de sesión
DRIVER_ID = "Conductor_001"
SESSION_START = datetime.now()

# Insertar inicio de sesión en supabase (tabla drivers)
supabase.table("drivers").insert({
    "driver_id": DRIVER_ID,
    "session_start": SESSION_START.isoformat()
}).execute()

# Inicializar captura de video (cámara)
cap = cv2.VideoCapture(0)  # Ajusta el índice de cámara si es necesario

# Detector de rostros y predictor de landmarks de Dlib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Índices para los ojos en los 68 puntos del predictor
LEFT_EYE = list(range(36, 42))
RIGHT_EYE = list(range(42, 48))

# ----------------------------------------------------------------------------
#                           FUNCIONES AUXILIARES
# ----------------------------------------------------------------------------

def calculate_ear(eye_points):
    """
    Calcula el Eye Aspect Ratio (EAR) dado un array con 6 puntos del ojo.
    """
    A = dist.euclidean(eye_points[1], eye_points[5])  # distancia vertical 1-5
    B = dist.euclidean(eye_points[2], eye_points[4])  # distancia vertical 2-4
    C = dist.euclidean(eye_points[0], eye_points[3])  # distancia horizontal 0-3
    return (A + B) / (2.0 * C)

def moving_average(values, window_size):
    """
    Devuelve el promedio móvil de una lista de valores.
    """
    import numpy as np
    if len(values) == 0:
        return 0.0
    if len(values) < window_size:
        return np.mean(values)
    else:
        return np.mean(values[-window_size:])

# Mapeo opcional de emociones de DeepFace a español
emotion_map = {
    "angry": "enojado",
    "disgust": "disgust",     # No siempre se reporta
    "fear": "miedo",
    "happy": "feliz",
    "sad": "triste",
    "surprise": "sorprendido",
    "neutral": "neutral"
}

# ----------------------------------------------------------------------------
#                           BUCLE PRINCIPAL
# ----------------------------------------------------------------------------

last_frame_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(frame_gray)

    current_time = time.time()
    frame_duration = current_time - last_frame_time
    last_frame_time = current_time

    face_detected = False

    # ----------------------- DETECCIÓN DE ROSTRO y LANDMARKS ------------------------
    for face in faces:
        face_detected = True
        # Predecir landmarks
        shape_dlib = predictor(frame_gray, face)
        landmarks = np.array([[p.x, p.y] for p in shape_dlib.parts()])

        # Ojos
        left_eye_points = landmarks[LEFT_EYE]
        right_eye_points = landmarks[RIGHT_EYE]
        left_ear = calculate_ear(left_eye_points)
        right_ear = calculate_ear(right_eye_points)
        ear = (left_ear + right_ear) / 2.0

        # Suavizar EAR con promedio móvil
        ear_buffer.append(ear)
        smoothed_ear = moving_average(ear_buffer, EAR_BUFFER_SIZE)

        # Dibujar ojos en la imagen (opcional)
        cv2.polylines(frame, [left_eye_points], True, (0, 255, 0), 1)
        cv2.polylines(frame, [right_eye_points], True, (0, 255, 0), 1)

        # ---------------------------------------------------------------------------
        # DETECCIÓN DE OJOS CERRADOS (SOMNOLENCIA)
        # ---------------------------------------------------------------------------
        if smoothed_ear < EAR_THRESHOLD:
            # Acumular el tiempo de ojos cerrados
            TOTAL_EYE_CLOSED += frame_duration

            if eyes_closed_start_time is None:
                # Se acaban de cerrar los ojos
                eyes_closed_start_time = current_time
                eyes_closed_warning_shown = False
                eyes_closed_alarm_triggered = False

            closed_duration = current_time - eyes_closed_start_time

            # Mostrar ADVERTENCIA apenas detectamos ojos cerrados
            if not eyes_closed_warning_shown:
                cv2.putText(frame, "ADVERTENCIA: OJOS CERRADOS",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 255, 255), 2)
                eyes_closed_warning_shown = True

            # Si supera los EYE_CLOSED_WARNING_SEC, sonar alarma
            if closed_duration >= EYE_CLOSED_WARNING_SEC and not eyes_closed_alarm_triggered:
                playsound("alert.wav", block=False)
                ALARM_COUNT += 1
                FATIGUE_EVENT_COUNT += 1
                eyes_closed_alarm_triggered = True

                # Insertar evento en Supabase (fatigue_events)
                supabase.table("fatigue_events").insert({
                    "driver_id": DRIVER_ID,
                    "event_time": datetime.now().isoformat(),
                    "alert_type": "somnolencia",
                    "eye_closed_seconds": TOTAL_EYE_CLOSED,
                    "alarm_triggered": True
                }).execute()

                cv2.putText(frame, "ALERTA: OJOS CERRADOS (ALARMA)",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 0, 255), 2)
            else:
                # Mientras no llegue al tiempo de alarma, mantenemos la advertencia
                if closed_duration < EYE_CLOSED_WARNING_SEC:
                    cv2.putText(frame, "ADVERTENCIA: OJOS CERRADOS",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (0, 255, 255), 2)
        else:
            # Si EAR regresa a > EAR_THRESHOLD, ojos abiertos
            eyes_closed_start_time = None
            eyes_closed_warning_shown = False
            eyes_closed_alarm_triggered = False

        # ---------------------------------------------------------------------------
        # DETECCIÓN DE EMOCIONES con DeepFace
        # ---------------------------------------------------------------------------
        # Para un mejor resultado, recorta la cara detectada (ROI)
        x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
        roi_face = frame[y1:y2, x1:x2]

        try:
            # Con enforce_detection=False evitamos error si la cara no se detecta bien
            analysis = DeepFace.analyze(
                img_path = roi_face,
                actions = ['emotion'],
                enforce_detection=False
            )
            # La respuesta suele tener la estructura:
            # {
            #   'emotion': {
            #       'angry': 0.0, 'disgust': 0.0, 'fear': 0.0,
            #       'happy': 99.99, 'sad': 0.01, 'surprise': 0.0, 'neutral': 0.0
            #   },
            #   'dominant_emotion': 'happy',
            #   ...
            # }

            # Emoción principal
            dominant_emotion = analysis['dominant_emotion']
            # Probabilidad de esa emoción (opcional, si quieres mostrarlo)
            score = analysis['emotion'][dominant_emotion]

            # Mapeo a español (opcional)
            emotion_es = emotion_map.get(dominant_emotion, dominant_emotion)

            # Mostrar en pantalla la emoción principal
            cv2.putText(
                frame,
                f"Emocion: {emotion_es} ({score:.2f}%)",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2
            )

            # Insertar en la tabla "emotions" de Supabase
            supabase.table("emotions").insert({
                "driver_id": DRIVER_ID,
                "event_time": datetime.now().isoformat(),
                "emotion": emotion_es
            }).execute()

        except Exception as e:
            # Si DeepFace falla en la detección, continuamos
            # (Por ejemplo, si la cara está muy ladeada, muy oscura, etc.)
            pass

    # Si no se detecta ningún rostro
    if not face_detected:
        cv2.putText(frame, "No se detecta rostro",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 255), 2)

    # Mostrar EAR de depuración en pantalla (opcional)
    if len(ear_buffer) > 0:
        cv2.putText(frame, f"EAR (promedio): {moving_average(ear_buffer, EAR_BUFFER_SIZE):.2f}",
                    (300, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2)

    # Visualización
    cv2.imshow("Sistema de Deteccion de Fatiga - SafeDrive", frame)

    # Salir con tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ----------------------------------------------------------------------------
#                 FINALIZACION DE SESION Y REGISTRO EN SUPABASE
# ----------------------------------------------------------------------------
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