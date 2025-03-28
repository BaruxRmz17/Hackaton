import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist
from playsound import playsound
import time

# Función para calcular el Eye Aspect Ratio (EAR)
def calculate_ear(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Función para calcular la inclinación de la cabeza
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
EAR_THRESHOLD = 0.25  # Umbral para ojos cerrados
EAR_CONSEC_FRAMES = 20  # Frames consecutivos para alerta por ojos
TILT_THRESHOLD = 30  # Umbral para inclinación de la cabeza (en grados)
TILT_CONSEC_FRAMES = 20  # Frames consecutivos para alerta por inclinación
COUNTER_EAR = 0  # Contador para ojos cerrados
COUNTER_TILT = 0  # Contador para inclinación de la cabeza

# Cargar el detector y predictor de dlib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Índices de los puntos faciales
LEFT_EYE = list(range(36, 42))  # Ojo izquierdo
RIGHT_EYE = list(range(42, 48))  # Ojo derecho

# Iniciar captura de video
cap = cv2.VideoCapture(0)
last_ear_check = time.time()  # Tiempo de la última vez que se verificaron los ojos

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)
        landmarks = np.array([[p.x, p.y] for p in landmarks.parts()])

        # Detección de ojos cerrados
        left_eye = landmarks[LEFT_EYE]
        right_eye = landmarks[RIGHT_EYE]
        left_ear = calculate_ear(left_eye)
        right_ear = calculate_ear(right_eye)
        ear = (left_ear + right_ear) / 2.0

        # Detección de inclinación de la cabeza
        tilt_angle = calculate_head_tilt(landmarks)

        # Dibujar contornos de los ojos (opcional)
        cv2.polylines(frame, [left_eye], True, (0, 255, 0), 1)
        cv2.polylines(frame, [right_eye], True, (0, 255, 0), 1)

        # Lógica de detección de fatiga (ojos cerrados)
        if ear < EAR_THRESHOLD:
            COUNTER_EAR += 1
            if COUNTER_EAR >= EAR_CONSEC_FRAMES:
                # Si no ha sonado la alarma o si los ojos están cerrados nuevamente
                current_time = time.time()
                if current_time - last_ear_check > 1:  # Solo suena cada 1 segundo
                    playsound("alert.wav", block=False)  # Sonido de alerta
                    last_ear_check = current_time  # Actualiza el tiempo de la última vez que sonó la alarma
                    cv2.putText(frame, "ALERTA: OJOS CERRADOS", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            COUNTER_EAR = 0

        # Lógica de detección de inclinación
        if abs(tilt_angle) > TILT_THRESHOLD:
            COUNTER_TILT += 1
            if COUNTER_TILT >= TILT_CONSEC_FRAMES:
                if COUNTER_EAR < EAR_CONSEC_FRAMES:  # Solo activar si no hay alerta de ojos cerrados
                    cv2.putText(frame, "ALERTA: CABEZA INCLINADA", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            COUNTER_TILT = 0

        # Mostrar EAR y ángulo en pantalla
        cv2.putText(frame, f"EAR: {ear:.2f}", (300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Tilt: {tilt_angle:.2f}", (300, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Sistema de Deteccion de Fatiga", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
