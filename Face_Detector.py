import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist
from playsound import playsound
import time
import datetime
import json

class SistemaRegistroFatiga:
    def __init__(self):
        # Configuración de detección facial
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        
        # Configuración de parámetros
        self.EAR_THRESHOLD = 0.25
        self.EAR_CONSEC_FRAMES = 20
        self.TILT_THRESHOLD = 30
        self.TILT_CONSEC_FRAMES = 20
        
        # Índices de puntos faciales
        self.LEFT_EYE = list(range(36, 42))
        self.RIGHT_EYE = list(range(42, 48))
        
        # Datos de usuarios
        self.usuarios = {}
        self.usuario_actual = None
        self.load_data()

    def load_data(self):
        try:
            with open('usuarios.json', 'r') as f:
                self.usuarios = json.load(f)
        except FileNotFoundError:
            self.usuarios = {}

    def save_data(self):
        with open('usuarios.json', 'w') as f:
            json.dump(self.usuarios, f)

    def calculate_ear(self, eye):
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        C = dist.euclidean(eye[0], eye[3])
        return (A + B) / (2.0 * C)

    def calculate_head_tilt(self, landmarks):
        nose = landmarks[30]
        left_eye = landmarks[36]
        right_eye = landmarks[45]
        eye_center = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)
        delta_x = nose[0] - eye_center[0]
        delta_y = nose[1] - eye_center[1]
        return np.degrees(np.arctan2(delta_y, delta_x))

    def detectar_rostro(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            return False
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
            
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            cv2.imshow('Verificación Facial', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        rostro_detectado = len(faces) > 0
        cap.release()
        cv2.destroyAllWindows()
        return rostro_detectado

    def registrar_usuario(self):
        nombre = input("Ingresa tu nombre para registrarte: ")
        self.usuarios[nombre] = {
            'registrado': True,
            'veces_dormido': 0,
            'minutos_dormido': 0,
            'ultima_entrada': None,
            'descanso_recomendado': 0
        }
        self.save_data()
        print(f"\033[92mUsuario {nombre} registrado!\033[0m")

    def verificar_usuario(self):
        nombre = input("Ingresa tu nombre: ")
        if self.detectar_rostro():
            if nombre in self.usuarios:
                self.usuario_actual = nombre
                hora = datetime.datetime.now().strftime("%H:%M:%S")
                self.usuarios[nombre]['ultima_entrada'] = hora
                print(f"\033[92m¡Bienvenido {nombre}!\033[0m")
                print(f"Hora de entrada: {hora}")
                self.monitorear_fatiga()
            else:
                print(f"\033[91mUsuario no registrado\033[0m")
                if input("¿Deseas registrarte? (s/n): ").lower() == 's':
                    self.registrar_usuario()
        else:
            print(f"\033[91mNo se detectó rostro\033[0m")

    def monitorear_fatiga(self):
        cap = cv2.VideoCapture(0)
        counter_ear = 0
        counter_tilt = 0
        last_ear_check = time.time()
        inicio_sueno = None

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.detector(gray)

            for face in faces:
                landmarks = self.predictor(gray, face)
                landmarks = np.array([[p.x, p.y] for p in landmarks.parts()])

                # Calcular EAR
                left_ear = self.calculate_ear(landmarks[self.LEFT_EYE])
                right_ear = self.calculate_ear(landmarks[self.RIGHT_EYE])
                ear = (left_ear + right_ear) / 2.0

                # Calcular inclinación
                tilt_angle = self.calculate_head_tilt(landmarks)

                # Dibujar contornos
                cv2.polylines(frame, [landmarks[self.LEFT_EYE]], True, (0, 255, 0), 1)
                cv2.polylines(frame, [landmarks[self.RIGHT_EYE]], True, (0, 255, 0), 1)

                # Detección de ojos cerrados
                if ear < self.EAR_THRESHOLD:
                    counter_ear += 1
                    if counter_ear >= self.EAR_CONSEC_FRAMES:
                        if inicio_sueno is None:
                            inicio_sueno = time.time()
                        current_time = time.time()
                        if current_time - last_ear_check > 1:
                            playsound("alert.wav", block=False)
                            last_ear_check = current_time
                            cv2.putText(frame, "ALERTA: OJOS CERRADOS", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    if inicio_sueno is not None:
                        tiempo_dormido = (time.time() - inicio_sueno) / 60  # En minutos
                        self.usuarios[self.usuario_actual]['veces_dormido'] += 1
                        self.usuarios[self.usuario_actual]['minutos_dormido'] += tiempo_dormido
                        self.usuarios[self.usuario_actual]['descanso_recomendado'] += tiempo_dormido * 2
                        self.save_data()
                        inicio_sueno = None
                    counter_ear = 0

                # Detección de inclinación
                if abs(tilt_angle) > self.TILT_THRESHOLD:
                    counter_tilt += 1
                    if counter_tilt >= self.TILT_CONSEC_FRAMES and counter_ear < self.EAR_CONSEC_FRAMES:
                        cv2.putText(frame, "ALERTA: CABEZA INCLINADA", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    counter_tilt = 0

                # Mostrar estadísticas
                cv2.putText(frame, f"EAR: {ear:.2f}", (300, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, f"Tilt: {tilt_angle:.2f}", (300, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, f"Veces dormido: {self.usuarios[self.usuario_actual]['veces_dormido']}", 
                            (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, f"Min dormido: {self.usuarios[self.usuario_actual]['minutos_dormido']:.1f}", 
                            (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, f"Descanso rec: {self.usuarios[self.usuario_actual]['descanso_recomendado']:.1f}m", 
                            (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow("Monitoreo de Fatiga", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    def run(self):
        while True:
            print("\n=== Sistema de Registro y Monitoreo ===")
            print("1. Verificar ingreso")
            print("2. Salir")
            opcion = input("Selecciona una opción: ")
            
            if opcion == '1':
                self.verificar_usuario()
            elif opcion == '2':
                print("¡Hasta luego!")
                break

if __name__ == "__main__":
    sistema = SistemaRegistroFatiga()
    sistema.run()