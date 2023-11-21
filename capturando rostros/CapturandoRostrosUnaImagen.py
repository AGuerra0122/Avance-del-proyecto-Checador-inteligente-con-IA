import cv2 # se importa la libreria de opencv
faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml') # En resumen, esta línea de código carga un clasificador de cascada
                                                                                                 # Haar pre-entrenado para detectar rostros frontales, lo que permite
                                                                                                 # la detección automática de rostros en una imagen o secuencia de video

image = cv2.imread('gente.jpg') # Se almacena una imagen dentro de una variable
imageAux = image.copy() # Se hace una copia de la imagen de entrada
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # Se pasa la imagen de bgr a escala de grises y se almacena en gray

faces = faceClassif.detectMultiScale(gray, 1.1, 5) # Se realiza la deteccion de rostros y se almacenan las coodenadas de inicio ademas del ancho y alto
count = 0 # Variable de control para asignar un numero distintivo a cada rostro detectado

for (x,y,w,h) in faces: # Con este ciclo for se iran identificando y almacenando los rostros detectados tomando los valores del arreglo de arreglos faces
    cv2.rectangle(image, (x,y),(x+w,y+h),(128,0,255),2) # Se dibuja un rectangulo sobre el rostro detectado en la imagen, teniendo las coordenadas de inicio y fin es decir las coordenadas
                                                        # de dos esquinas opuestas
    rostro = imageAux[y:y+h,x:x+w] # De la imagen auxiliar (no tiene los rectangulos dibujados) se recorta la zona que tiene el rostro detectado y lo guarda en rostro
    rostro = cv2.resize(rostro,(150,150),interpolation=cv2.INTER_CUBIC) # se redimenciona la imagen a un estandar de 150x150 px
    cv2.imwrite('rostro_{}.jpg'.format(count),rostro) # Se almacena el rostro detectado con la leyenda "rostro_#.jpg" el numero que tomara el lugar del # es valor de la variable count
    count = count + 1

    cv2.imshow('image',image) # Se muestra la imagen original con los rostros detectados en cada iteracion del ciclo for
    cv2.imshow('rostro',rostro) # Se muestra el recorte del rostro detectado en la iteracion
    cv2.waitKey(0)

cv2.destroyAllWindows()