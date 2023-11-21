# Importar bibliotecas necesarias
import cv2
import os
import face_recognition

# Codificar los rostros extaridos
imageFacesPath = "C:/Users/1hgue/OneDrive/Escritorio/ITCH/8vo Semestre/Proyecto del servicio/reconociendo varias personas/faces"
# Listas para almacenar las codificaciones de los rostros y sus nombres
facesEncodings = []
facesNames = []
# Iterar sobre cada archivo en el directorio de rostros
for file_name in os.listdir(imageFacesPath):
    # Leer la imagen del rostro y convertirla al formato RGB
    image = cv2.imread(imageFacesPath + "/" + file_name)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Codificar el rostro usando face_recognition
    f_coding = face_recognition.face_encodings(image, known_face_locations=[(0, 150, 150 , 0)])[0]
    # Agregar la codificación y el nombre a las listas respectivas
    facesEncodings.append(f_coding)
    facesNames.append(file_name.split(".")[0])
'''print(facesEncodings)
print(facesNames)'''

####################################################
# Leyendo video
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Detector Facial
faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
# Bucle principal para procesar el video en tiempo real
while True:
    # Capturar un fotograma del video
    ret, frame = cap.read()
    if ret == False:
        break
    # Voltear el fotograma horizontalmente
    frame = cv2.flip(frame, 1)
    orig = frame.copy()
    # Detectar rostros en el fotograma
    faces = faceClassif.detectMultiScale(frame, 1.3, 5)
    # Iterar sobre cada rostro detectado en el fotograma
    for (x, y, w, h) in faces:
        # Recortar el rostro de la imagen original y convertirlo al formato RGB
        face = orig[y:y + h, x:x + w]
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        # Obtener la codificación del rostro actual usando face_recognition
        actual_face_encoding = face_recognition.face_encodings(face, known_face_locations=[(0, w, h, 0)])[0]
        # Comparar la codificación del rostro actual con las codificaciones almacenadas
        result = face_recognition.compare_faces(facesEncodings, actual_face_encoding)
        #print(result)
        # Determinar el nombre y color a mostrar según el resultado de la comparación
        if True in result:
            index = result.index(True)
            name = facesNames[index]
            color = (125, 220, 0) # Verde para rostro conocido
        else:
            name = "Desconocido"
            color = (50, 50, 255) # Rojo para rostro desconocido
        # Dibujar rectángulos y texto en el fotograma
        cv2.rectangle(frame, (x, y),(x + w, y + h), color,2)
        cv2.rectangle(frame, (x, y + h),(x + w, y + h + 30), color,-1)
        cv2.putText(frame, name, (x, y + h + 25), 2, 1, (255, 255, 255), 2, cv2.LINE_AA)
    # Mostrar el resultado en una ventana de visualización
    cv2.imshow("Frame", frame)
    # Salir del bucle al presionar la tecla 'Esc' (27)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

'''    cv2.imshow("Image", image)
    cv2.waitKey(0)
cv2.destroyAllWindows()'''
