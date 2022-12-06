import cv2
import dlib

video_capture = cv2.VideoCapture(0)

detector_face_hog = dlib.get_frontal_face_detector()


while True:
    # Captura frame pro frame
    ok, frame = video_capture.read()

    imagem_cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    deteccoes = detector_face_hog(imagem_cinza, 1)

    # Desenha o retângulo
    for face in deteccoes:
        l, t, r, b = face.left(), face.top(), face.right(), face.bottom()
        cv2.rectangle(frame, (l, t), (r, b), (0, 0, 255), 2)
    
    cv2.imshow('Video', frame)
    
    # Finaliza o processo da webcam ao apertar a tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#Libera memória no final
video_capture.release()
cv2.destroyAllWindows()
