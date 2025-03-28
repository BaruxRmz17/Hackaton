# main.py
import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist
from playsound import playsound
import time
from datetime import datetime

# Supabase
from supabase_config import SUPABASE_URL, SUPABASE_KEY
from supabase import create_client, Client

# ----------------------------------------------------------------------------
#                           CONFIGURACIÓN / CONSTANTES
# ----------------------------------------------------------------------------

# Umbral de EAR para considerar ojos cerrados
EAR_THRESHOLD = 0.25  
# Tiempo que se permite con ojos cerrados antes de sonar la alarma (segundos)
EYE_CLOSED_WARNING_SEC = 3.0  

# Umbral de tamaño de buffer para suavizar EAR (promedio móvil)
EAR_BUFFER_SIZE = 5  

# Umbral de ángulos (en grados) para considerar la cabeza inclinada
# (por ejemplo, si pitch o yaw o roll exceden +/- 20°, se considera inclinada)
HEAD_TILT_THRESHOLD = 20.0
# Tiempo de advertencia y alarma para inclinación
HEAD_TILT_WARNING_SEC = 2.0
HEAD_TILT_ALARM_SEC = 5.0

# Variables para FATIGA / SOMNOLENCIA
eyes_closed_start_time = None  # Cuándo se detectó que los ojos se cerraron
eyes_closed_warning_shown = False
eyes_closed_alarm_triggered = False

# Variables para INCLINACIÓN DE CABEZA
head_tilt_start_time = None
head_tilt_warning_shown = False
head_tilt_alarm_triggered = False

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

# Insertar inicio de sesión en supabase
supabase.table("drivers").insert({
    "driver_id": DRIVER_ID,
    "session_start": SESSION_START.isoformat()
}).execute()

# Inicializar captura de video (cámara)
cap = cv2.VideoCapture(0)  # Asegúrate de que sea la cámara correcta

# Detector y predictor de Dlib
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
    if len(values) < window_size:
        return np.mean(values)
    else:
        return np.mean(values[-window_size:])

def get_head_pose(shape, frame_size):
    """
    Calcula la pose de la cabeza (pitch, yaw, roll) usando solvePnP.
    Devuelve (pitch, yaw, roll) en grados.
    shape: np.array de 68x2 con landmarks faciales
    frame_size: tupla (width, height) del frame para estimar la cámara intrínseca
    """
    # Puntos 2D relevantes de la cara (en la imagen)
    image_points = np.array([
        shape[30],  # Nariz
        shape[8],   # Barbilla
        shape[36],  # Esquina izq. ojo izq.
        shape[45],  # Esquina der. ojo der.
        shape[48],  # Esquina izq. boca
        shape[54]   # Esquina der. boca
    ], dtype="double")

    # Puntos 3D del modelo de la cabeza (arbitrarios pero proporcionales)
    model_points = np.array([
        (0.0, 0.0, 0.0),         # Nariz
        (0.0, -330.0, -65.0),    # Barbilla
        (-225.0, 170.0, -135.0), # Ojo izq.
        (225.0, 170.0, -135.0),  # Ojo der.
        (-150.0, -150.0, -125.0),# Boca izq.
        (150.0, -150.0, -125.0)  # Boca der.
    ])

    # Parámetros intrínsecos de cámara aproximados
    focal_length = frame_size[0]
    center = (frame_size[0] / 2, frame_size[1] / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype="double")

    dist_coeffs = np.zeros((4, 1))  # Suponemos sin distorsión

    success, rotation_vector, _ = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not success:
        return 0.0, 0.0, 0.0

    # Convertir a ángulos (pitch, yaw, roll)
    rmat, _ = cv2.Rodrigues(rotation_vector)
    # Extraer ángulos en grados
    # Referencia: https://learnopencv.com/rotation-matrix-to-euler-angles/
    sy = np.sqrt(rmat[0, 0]*rmat[0, 0] + rmat[1, 0]*rmat[1, 0])
    singular = sy < 1e-6

    if not singular:
        pitch = np.arctan2(-rmat[2, 0], sy)
        yaw = np.arctan2(rmat[1, 0], rmat[0, 0])
        roll = np.arctan2(rmat[2, 1], rmat[2, 2])
    else:
        pitch = np.arctan2(-rmat[2, 0], sy)
        yaw = np.arctan2(-rmat[0, 1], rmat[1, 1])
        roll = 0

    pitch = np.degrees(pitch)
    yaw = np.degrees(yaw)
    roll = np.degrees(roll)

    return pitch, yaw, roll

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

    frame_height, frame_width = frame.shape[:2]

    # Por si no se detecta ninguna cara, reseteamos algunas alertas visuales
    face_detected = False

    for face in faces:
        face_detected = True
        # Obtener landmarks
        shape_dlib = predictor(frame_gray, face)
        landmarks = np.array([[p.x, p.y] for p in shape_dlib.parts()])

        # Ojos
        left_eye_points = landmarks[LEFT_EYE]
        right_eye_points = landmarks[RIGHT_EYE]
        left_ear = calculate_ear(left_eye_points)
        right_ear = calculate_ear(right_eye_points)
        ear = (left_ear + right_ear) / 2.0

        # Suavizado del EAR
        ear_buffer.append(ear)
        smoothed_ear = moving_average(ear_buffer, EAR_BUFFER_SIZE)

        # Head Pose
        pitch, yaw, roll = get_head_pose(landmarks, (frame_width, frame_height))

        # Dibujar ojos en pantalla
        cv2.polylines(frame, [left_eye_points], True, (0, 255, 0), 1)
        cv2.polylines(frame, [right_eye_points], True, (0, 255, 0), 1)

        # --------------------------------------------------------------------
        # DETECCIÓN DE OJOS CERRADOS CON TIEMPO
        # --------------------------------------------------------------------
        if smoothed_ear < EAR_THRESHOLD:
            # Acumular el tiempo de ojos cerrados
            TOTAL_EYE_CLOSED += frame_duration

            if eyes_closed_start_time is None:
                # Se acaban de cerrar los ojos
                eyes_closed_start_time = current_time
                eyes_closed_warning_shown = False
                eyes_closed_alarm_triggered = False

            closed_duration = current_time - eyes_closed_start_time

            # 1) Pasado cierto tiempo, mostrar ADVERTENCIA en pantalla
            if closed_duration >= 0.0 and not eyes_closed_warning_shown:
                # Se muestra ADVERTENCIA apenas cierra ojos, pero puedes darle
                # un margen si lo deseas (por ej. 1 seg)
                cv2.putText(frame, "ADVERTENCIA: OJOS CERRADOS",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 255, 255), 2)
                eyes_closed_warning_shown = True

            # 2) Si supera los 3 segundos (EYE_CLOSED_WARNING_SEC), suena alarma
            if closed_duration >= EYE_CLOSED_WARNING_SEC and not eyes_closed_alarm_triggered:
                playsound("alert.wav", block=False)
                ALARM_COUNT += 1
                FATIGUE_EVENT_COUNT += 1
                eyes_closed_alarm_triggered = True

                # Insertar evento en Supabase
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
                # Si está cerrado, pero aún no se supera el umbral de alarma,
                # solo mostramos la advertencia
                if closed_duration < EYE_CLOSED_WARNING_SEC:
                    cv2.putText(frame, "ADVERTENCIA: OJOS CERRADOS",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (0, 255, 255), 2)

        else:
            # Si EAR regresa por encima del umbral, ojos abiertos
            eyes_closed_start_time = None
            eyes_closed_warning_shown = False
            eyes_closed_alarm_triggered = False

        # --------------------------------------------------------------------
        # DETECCIÓN DE CABEZA INCLINADA (PITCH, YAW, ROLL)
        # Se considera inclinada si se pasa un umbral en cualquiera de los ángulos
        # --------------------------------------------------------------------
        def head_is_tilted(p, y, r):
            """
            Retorna True si alguno de los ángulos excede HEAD_TILT_THRESHOLD.
            """
            if abs(p) > HEAD_TILT_THRESHOLD:
                return True
            if abs(y) > HEAD_TILT_THRESHOLD:
                return True
            if abs(r) > HEAD_TILT_THRESHOLD:
                return True
            return False

        if head_is_tilted(pitch, yaw, roll):
            if head_tilt_start_time is None:
                # Se detectó inclinación por primera vez
                head_tilt_start_time = current_time
                head_tilt_warning_shown = False
                head_tilt_alarm_triggered = False

            tilt_duration = current_time - head_tilt_start_time

            # Mostrar ADVERTENCIA luego de HEAD_TILT_WARNING_SEC
            if tilt_duration >= HEAD_TILT_WARNING_SEC and not head_tilt_warning_shown:
                cv2.putText(frame, "ADVERTENCIA: CABEZA INCLINADA",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 255, 255), 2)
                head_tilt_warning_shown = True

            # Después de HEAD_TILT_ALARM_SEC, dispara alarma
            if tilt_duration >= HEAD_TILT_ALARM_SEC and not head_tilt_alarm_triggered:
                playsound("alert.wav", block=False)
                ALARM_COUNT += 1
                FATIGUE_EVENT_COUNT += 1
                head_tilt_alarm_triggered = True

                supabase.table("fatigue_events").insert({
                    "driver_id": DRIVER_ID,
                    "event_time": datetime.now().isoformat(),
                    "alert_type": "cabeza muy inclinada",
                    "eye_closed_seconds": 0.0,
                    "alarm_triggered": True
                }).execute()

                cv2.putText(frame, "ALERTA: CABEZA MUY INCLINADA (ALARMA)",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 0, 255), 2)

            else:
                # Si ya se mostró la advertencia, pero no se ha llegado
                # al tiempo de alarma
                if tilt_duration < HEAD_TILT_ALARM_SEC:
                    if head_tilt_warning_shown and not head_tilt_alarm_triggered:
                        cv2.putText(frame, "ADVERTENCIA: CABEZA INCLINADA",
                                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.7, (0, 255, 255), 2)
        else:
            # Regresó a postura normal
            head_tilt_start_time = None
            head_tilt_warning_shown = False
            head_tilt_alarm_triggered = False

        # --------------------------------------------------------------------
        # MOSTRAR INFO DE DEBUG EN PANTALLA
        # --------------------------------------------------------------------
        cv2.putText(frame, f"EAR (promedio): {smoothed_ear:.2f}",
                    (300, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Pitch: {pitch:.2f} Yaw: {yaw:.2f} Roll: {roll:.2f}",
                    (300, 60), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2)

    # Si no se detecta rostro, podrías mostrar un mensaje o manejarlo diferente
    if not face_detected:
        cv2.putText(frame, "No se detecta rostro",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 255), 2)

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