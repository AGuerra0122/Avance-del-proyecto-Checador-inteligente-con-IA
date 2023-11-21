# Importar bibliotecas necesarias
import cv2
import os
# Ruta del directorio que contiene las imágenes a procesar
imagesPath = "C:/Users/1hgue/OneDrive/Escritorio/ITCH/8vo Semestre/Proyecto del servicio/reconociendo varias personas/Images"
# Verificar si la carpeta 'faces' ya existe, y si no, crearla
if not os.path.exists("faces"):
    os.makedirs("faces")
    print("Nueva carpeta: faces")

# Inicializar el clasificador de rostros (Haar Cascade)
faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
# Inicializar el contador para asignar nombres únicos a las imágenes de rostros
count = 0
# Iterar sobre cada imagen en el directorio especificado
for imageName in os.listdir(imagesPath):
    print(imageName)
    # Leer la imagen desde la ruta completa
    image = cv2.imread(imagesPath + "/" + imageName)
    # Detectar rostros en la imagen
    faces = faceClassif.detectMultiScale(image, 1.3, 5)
    # Iterar sobre cada rostro detectado en la imagen
    for (x, y, w, h) in faces:
        #cv2.rectangle(image, (x, y), (x + w , y + h), (0,255,0), 2)
        # Recortar el rostro de la imagen original
        face = image[y:y + h, x:x +w]
        # Redimensionar el rostro a 100x100 píxeles
        face = cv2.resize(face, (100, 100))
        # Guardar el rostro recortado en el directorio 'faces'
        cv2.imwrite("faces/" + str(count) + ".jpg", face)
        # Incrementar el contador para el siguiente rostro
        count += 1
        #cv2.imshow("face", face)
        #cv2.waitKey(0)
    #cv2.imshow("Image", image)
    #cv2.waitKey(0)
#cv2.destroyAllWindows()