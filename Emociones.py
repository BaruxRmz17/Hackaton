import cv2
from fer import FER
import numpy as np

# Inicializar el detector de emociones
detector = FER(mtcnn=True)

# Iniciar la captura de video desde la cámara (0 es la cámara predeterminada)
cap = cv2.VideoCapture(0)

# Verificar si la cámara se abrió correctamente
if not cap.isOpened():
    print("Error: No se pudo abrir la cámara")
    exit()

# Bucle principal para procesar el video en tiempo real
while True:
    # Leer frame por frame
    ret, frame = cap.read()
    
    if not ret:
        print("Error: No se pudo leer el frame")
        break

    # Detectar emociones en el frame
    result = detector.detect_emotions(frame)
    
    # Procesar los resultados si se detectan rostros
    if result:
        for face in result:
            # Obtener coordenadas del rostro
            x, y, w, h = face['box']
            
            # Dibujar un rectángulo alrededor del rostro
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Obtener las emociones detectadas
            emotions = face['emotions']
            dominant_emotion = max(emotions, key=emotions.get)
            
            # Mostrar la emoción dominante sobre el rectángulo
            cv2.putText(frame, dominant_emotion, 
                        (x, y-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.9, 
                        (0, 255, 0), 
                        2)
            
            # Opcional: Mostrar todas las emociones con sus probabilidades
            emotion_text = ""
            for emotion, score in emotions.items():
                emotion_text += f"{emotion}: {score:.2f}\n"
            
            # Dividir el texto en líneas y mostrarlo al lado del rostro
            for i, line in enumerate(emotion_text.split('\n')):
                if line:
                    cv2.putText(frame, line, 
                                (x+w+10, y+i*20), 
                                cv2.FONT_HERSHEY_SIMPLEX, 
                                0.5, 
                                (255, 255, 255), 
                                1)

    # Mostrar el frame procesado
    cv2.imshow('Detección de Emociones en Tiempo Real', frame)
    
    # Salir del bucle si se presiona 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la captura y cerrar ventanas
cap.release()
cv2.destroyAllWindows()