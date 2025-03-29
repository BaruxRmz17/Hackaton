import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist
from playsound import playsound
import time
from datetime import datetime

# DeepFace para emociones
from deepface import DeepFace

# Supabase
from supabase_config import SUPABASE_URL, SUPABASE_KEY
from supabase import create_client, Client
from postgrest.exceptions import APIError

# ---------------------------------------------------------
#            PARÁMETROS DE DETECCIÓN DE wsssssssssssssssssssssssssss
# ---------------------------------------------------------
EAR_THRESHOLD = 0.23
EYE_CLOSED_WARNING_SEC = 3.0
EAR_BUFFER_SIZE = 5

# ---------------------------------------------------------
#            PARÁMETROS DE DETECCIÓN DE EMOCIONES
# ---------------------------------------------------------
ANALYZE_EVERY_N_FRAMES = 10
FACE_MARGIN = 25

# Heurísticas para subir/bajar probabilidades
# (hemos reducido la influencia de la boca en enojo o tristeza)
INCREMENT_ANGRY = 1.4        # +40% enojo si cejas bajas
INCREMENT_SAD = 1.6          # +60% triste si comisuras caídas
DECREMENT_ANGRY_IF_SAD = 0.7 # -30% enojo si comisuras caídas
INCREMENT_FEAR = 1.3         # +30% miedo si boca muy abierta

# Umbrales de cejas y boca
EYEBROW_LOW_THRESHOLD = 15    # cejas bajas => posible enojo
MOUTH_OPEN_MIN = 0.4          # si MAR > 0.4 => boca abierta => fear
# Eliminamos la dependencia “boca cerrada = enojado” para mayor flexibilidad

# ---------------------------------------------------------
#       VARIABLES DE ESTADO DE FATIGA Y EMOCIÓN
# ---------------------------------------------------------
eyes_closed_start_time = None
eyes_closed_warning_shown = False
eyes_closed_alarm_triggered = False
TOTAL_EYE_CLOSED = 0.0
ALARM_COUNT = 0
FATIGUE_EVENT_COUNT = 0
ear_buffer = []

current_emotion_text = "Desconocida"

# Mapeo DeepFace -> Español
emotion_map = {
    "angry": "enojado",
    "disgust": "irritado",
    "fear": "miedo",
    "happy": "feliz",
    "sad": "triste",
    "surprise": "sorprendido",
    "neutral": "neutral"
}

# ---------------------------------------------------------
#           CONFIGURACIÓN DE SUPABASE
# ---------------------------------------------------------
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# -------------- FUNCIONES DE TABLAS --------------
def listar_conductores():
    resp = supabase.table("drivers").select("driver_id, driver_name").execute()
    return resp.data

def crear_conductor(nombre, email):
    try:
        insercion = supabase.table("drivers").insert({
            "driver_name": nombre,
            "driver_email": email
        }).execute()
        return insercion.data[0]["driver_id"]
    except APIError as e:
        print("Error creando conductor:", e)
        return None

def seleccionar_o_crear_conductor():
    while True:
        print("\n¿Deseas (E)legir un conductor existente o (N)uevo conductor?")
        opcion = input("Opción (E/N): ").strip().lower()
        if opcion == 'e':
            # Listar
            lista = listar_conductores()
            if not lista:
                print("No hay conductores registrados. Crea uno nuevo.")
                continue
            print("\nConductores existentes:")
            for c in lista:
                print(f"  ID: {c['driver_id']} - Nombre: {c['driver_name']}")
            try:
                elegido = int(input("Introduce el driver_id: ").strip())
                ids_validos = [c['driver_id'] for c in lista]
                if elegido in ids_validos:
                    return elegido
                else:
                    print("ID inválido. Intenta de nuevo.")
            except ValueError:
                print("Valor inválido. Intenta de nuevo.")
        elif opcion == 'n':
            nombre = input("Introduce el nombre del conductor: ").strip()
            email = input("Introduce el email (puedes dejarlo vacío): ").strip()
            driver_id = crear_conductor(nombre, email)
            if driver_id is not None:
                print(f"Conductor creado con driver_id={driver_id}")
                return driver_id
            else:
                print("No se pudo crear. Intenta de nuevo.")
        else:
            print("Opción no válida.")

def crear_sesion(driver_id):
    start_time = datetime.now()
    resp = supabase.table("driver_sessions").insert({
        "driver_id": driver_id,
        "session_start": start_time.isoformat()
    }).execute()
    session_data = resp.data[0]
    session_id = session_data["session_id"]
    return session_id, start_time

def cerrar_sesion(session_id, start_time):
    end_time = datetime.now()
    supabase.table("driver_sessions").update({
        "session_end": end_time.isoformat()
    }).eq("session_id", session_id).execute()
    dur = end_time - start_time
    print(f"\nSesión {session_id} finalizada. Duración: {dur}.")

def insertar_fatiga_event(session_id, eye_closed_seconds):
    supabase.table("fatigue_events").insert({
        "session_id": session_id,
        "event_time": datetime.now().isoformat(),
        "alert_type": "somnolencia",
        "eye_closed_seconds": eye_closed_seconds,
        "alarm_triggered": True
    }).execute()

def insertar_emocion(session_id, emotion_text):
    supabase.table("emotions").insert({
        "session_id": session_id,
        "event_time": datetime.now().isoformat(),
        "emotion": emotion_text
    }).execute()

# ---------------------------------------------------------
#      FUNCIONES DE DETECCIÓN (EAR, CEJAS, BOCA)
# ---------------------------------------------------------
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
    return np.mean(values[-window_size:])

def eyebrow_to_eye_distance(eye_points, brow_points):
    eye_center = np.mean(eye_points, axis=0)
    brow_center = np.mean(brow_points, axis=0)
    return dist.euclidean(eye_center, brow_center)

def mouth_aspect_ratio(mouth_points):
    horizontal = dist.euclidean(mouth_points[0], mouth_points[-1])
    vertical = dist.euclidean(mouth_points[2], mouth_points[5])
    if horizontal == 0:
        return 0
    return vertical / horizontal

def mouth_corners_down(mouth_points):
    left_corner = mouth_points[0]
    right_corner = mouth_points[-1]
    mid_top = mouth_points[2]
    mid_bottom = mouth_points[3]
    avg_mid_y = (mid_top[1] + mid_bottom[1]) / 2.0
    return (left_corner[1] > avg_mid_y and right_corner[1] > avg_mid_y)

# ---------------------------------------------------------
#              BUCLE PRINCIPAL DE EJECUCIÓN
# ---------------------------------------------------------
def main():
    # 1) Elegir/crear conductor
    driver_id = seleccionar_o_crear_conductor()
    # Obtener el nombre para mostrar en pantalla
    conductor_resp = supabase.table("drivers").select("driver_id, driver_name").eq("driver_id", driver_id).execute()
    conductor_name = conductor_resp.data[0]["driver_name"]

    # 2) Crear sesión
    session_id, session_start = crear_sesion(driver_id)
    print(f"\nIniciando sesión {session_id} para el conductor '{conductor_name}'")

    # 3) Configurar la detección
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    LEFT_EYE = list(range(36, 42))
    RIGHT_EYE = list(range(42, 48))
    LEFT_EYEBROW = list(range(17, 22))
    RIGHT_EYEBROW = list(range(22, 27))
    MOUTH_OUTER = list(range(48, 55))

    global eyes_closed_start_time, eyes_closed_warning_shown
    global eyes_closed_alarm_triggered, TOTAL_EYE_CLOSED, ALARM_COUNT
    global FATIGUE_EVENT_COUNT, ear_buffer, current_emotion_text

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

            # OJOS (EAR)
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
            brow_is_low = (avg_brow_dist < EYEBROW_LOW_THRESHOLD)

            # BOCA
            mouth_points = landmarks[MOUTH_OUTER]
            mar = mouth_aspect_ratio(mouth_points)
            corners_down = mouth_corners_down(mouth_points)

            # Dibujar ojos (opcional)
            cv2.polylines(frame, [left_eye_pts], True, (0, 255, 0), 1)
            cv2.polylines(frame, [right_eye_pts], True, (0, 255, 0), 1)

            # ------------ DETECCIÓN OJOS CERRADOS -------------
            if smoothed_ear < EAR_THRESHOLD and not (brow_is_low and smoothed_ear > EAR_THRESHOLD * 0.9):
                TOTAL_EYE_CLOSED += frame_duration
                if eyes_closed_start_time is None:
                    eyes_closed_start_time = current_time
                    eyes_closed_warning_shown = False
                    eyes_closed_alarm_triggered = False

                closed_duration = current_time - eyes_closed_start_time

                if not eyes_closed_warning_shown:
                    cv2.putText(frame, "ADVERTENCIA: OJOS CERRADOS", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    eyes_closed_warning_shown = True

                if closed_duration >= EYE_CLOSED_WARNING_SEC and not eyes_closed_alarm_triggered:
                    playsound("alert.wav", block=False)
                    ALARM_COUNT += 1
                    FATIGUE_EVENT_COUNT += 1
                    eyes_closed_alarm_triggered = True

                    insertar_fatiga_event(session_id, TOTAL_EYE_CLOSED)

                    cv2.putText(frame, "ALERTA: OJOS CERRADOS (ALARMA)",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (0, 0, 255), 2)
                else:
                    if closed_duration < EYE_CLOSED_WARNING_SEC:
                        cv2.putText(frame, "ADVERTENCIA: OJOS CERRADOS", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            else:
                eyes_closed_start_time = None
                eyes_closed_warning_shown = False
                eyes_closed_alarm_triggered = False

            # ------------ DETECCIÓN EMOCIONES -------------
            if frame_count % ANALYZE_EVERY_N_FRAMES == 0:
                x1, y1 = face.left(), face.top()
                x2, y2 = face.right(), face.bottom()

                x1m = max(0, x1 - FACE_MARGIN)
                y1m = max(0, y1 - FACE_MARGIN)
                x2m = min(frame.shape[1], x2 + FACE_MARGIN)
                y2m = min(frame.shape[0], y2 + FACE_MARGIN)

                roi_face = frame[y1m:y2m, x1m:x2m]

                try:
                    analysis = DeepFace.analyze(
                        roi_face,
                        actions=['emotion'],
                        enforce_detection=False
                    )
                    if isinstance(analysis, list):
                        analysis = analysis[0]

                    emotions_dict = analysis['emotion']  # e.g. {'angry': X, 'sad': Y, ...}

                    # Heurísticas suaves
                    if brow_is_low and 'angry' in emotions_dict:
                        emotions_dict['angry'] *= INCREMENT_ANGRY

                    if corners_down and 'sad' in emotions_dict:
                        emotions_dict['sad'] *= INCREMENT_SAD

                    if corners_down and 'angry' in emotions_dict:
                        emotions_dict['angry'] *= DECREMENT_ANGRY_IF_SAD

                    if mar > MOUTH_OPEN_MIN and 'fear' in emotions_dict:
                        emotions_dict['fear'] *= INCREMENT_FEAR

                    # Recalcular la dominante
                    sorted_emos = sorted(emotions_dict.items(), key=lambda x: x[1], reverse=True)
                    primary_emo, primary_val = sorted_emos[0]
                    secondary_emo, secondary_val = sorted_emos[1] if len(sorted_emos) > 1 else (None, 0)

                    # Reglas finales
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
                        if (primary_val - secondary_val) < 5 and secondary_emo in ['angry','sad','disgust']:
                            final_emo = secondary_emo
                            final_val = secondary_val
                        else:
                            final_emo = primary_emo
                            final_val = primary_val

                    # Mapeo a español
                    final_es = emotion_map.get(final_emo, final_emo)
                    current_emotion_text = f"{final_es} ({final_val:.1f}%)"

                    insertar_emocion(session_id, final_es)

                    # Mostrar distribución completa
                    offset_y = 200
                    font_scale = 0.5
                    for emo_eng, val in sorted_emos:
                        emo_es = emotion_map.get(emo_eng, emo_eng)
                        text_line = f"{emo_es}: {val:.1f}%"
                        cv2.putText(frame, text_line, (10, offset_y),
                                    cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                                    (200, 200, 200), 1)
                        offset_y += 15

                except Exception:
                    pass

        # Si no se detecta rostro
        if not face_detected:
            cv2.putText(frame, "No se detecta rostro", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Mostrar EAR
        if ear_buffer:
            avg_ear = moving_average(ear_buffer, EAR_BUFFER_SIZE)
            cv2.putText(frame, f"EAR: {avg_ear:.2f}", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Emoción final
        cv2.putText(frame, f"Emocion final: {current_emotion_text}", (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # Mostrar info de conductor y sesión
        cv2.putText(frame, f"Conductor: {conductor_name}", (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Sesión ID debajo del nombre
        cv2.putText(frame, f"Session ID: {session_id}", (10, 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.imshow("SafeDrive - Fatiga & Emociones", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Cerrar sesión
    cerrar_sesion(session_id, session_start)

if __name__ == "__main__":
    main()