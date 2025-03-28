import cv2
from deepface import DeepFace

def detectar_emocion():
    cap = cv2.VideoCapture(0)  # Abrir la cámara

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error al acceder a la cámara")
            break

        cv2.imshow('Detectando emociones...', frame)  # Mostrar la imagen en vivo

        # Analizar emoción cada 30 frames para no sobrecargar el sistema
        try:
            resultado = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            emocion = resultado[0]['dominant_emotion']
            print(f"Emoción detectada: {emocion}")
        except Exception as e:
            print(f"Error en la detección: {e}")

        # Salir con la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Ejecutar detección de emociones
detectar_emocion()
