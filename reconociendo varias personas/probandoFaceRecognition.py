# Importación de bibliotecas
import cv2
import face_recognition

# Cargar la imagen de referencia
image = cv2.imread("imagenes/paul.jpg")
face_loc = face_recognition.face_locations(image)[0]
#print("face_loc:", face_loc)
face_image_encodings = face_recognition.face_encodings(image, known_face_locations=[face_loc])[0]
#print("fcae_image_encodings:", face_image_encodings)

'''cv2.rectangle(image, (face_loc[3], face_loc[0]), (face_loc[1],face_loc[2]),(0, 255, 0))
cv2.imshow("image",image)
cv2.waitKey(0)
cv2.destroyAllWindows()'''

# video streming
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# Bucle principal para procesar el video en tiempo real
while True:
     # Capturar un fotograma del video
    ret, frame = cap.read()
    if ret == False:break
     # Voltear el fotograma horizontalmente para que coincida con la imagen de referencia
    frame = cv2.flip(frame, 1)
    # Detección de rostros en el fotograma actual
    face_locations = face_recognition.face_locations(frame)
    # Comparación de cada rostro detectado con el rostro de referencia
    if face_locations != []:
        for face_location in face_locations:
            face_frame_encodings = face_recognition.face_encodings(frame, known_face_locations=[face_location])[0]
            result = face_recognition.compare_faces([face_frame_encodings], face_image_encodings)
            #print("Result:", result)
            # Asignación de etiquetas y colores según el resultado de la comparación
            if result[0] == True:
                text = "Ant-man"
                color = (0,0,255) # Rojo para coincidencia
            else:
                text = "Desconocido"
                color = (255,0,0) # Azul para no coincidencia
            # Dibujar rectángulos y texto en el fotograma
            cv2.rectangle(frame, (face_location[3], face_location[0]),(face_location[1], face_location[2]), color,2)
            cv2.rectangle(frame, (face_location[3], face_location[2]),(face_location[1], face_location[2]+30), color,-1)
            cv2.putText(frame, text, (face_location[3], face_location[2] + 20), 2, 0.7, (255, 255, 255), 1)
    # Mostrar el resultado en una ventana de visualización
    cv2.imshow("Chequeador inteligente", frame)
    # Salir del bucle al presionar la tecla 'Esc' (27)
    k = cv2.waitKey(1)
    if k == 27 & 0xFF:
        break
# Liberar la captura de video y cerrar las ventanas al finalizar
cap.release()
cv2.destroyAllWindows()