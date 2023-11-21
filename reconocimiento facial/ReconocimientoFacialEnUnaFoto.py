import cv2
import numpy as np

faceClassif = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') # Se crea el clasificador a partir del archivo haarcascade_frontalface_default

image = cv2.imread('gente.jpg') # Se crea un objeto para almacenar la imagen a trabajar
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # Se crea un nuevo objeto el cual almacenara la imagen en escala de grices

faces = faceClassif.detectMultiScale( gray,scaleFactor = 1.1,minNeighbors=5,minSize=(30,30),maxSize=(200,200))#Se aplica el clasificador a la imagen
        #se escribe primero la variable donde se guardo el clasificador, seguido de esto la funcion detectMultiscale, esta funcion recibe:
                        # La imagen a trabajar(en escala de grices), el factor de escala, este escalado se hace para detectar rostros de distintos tamaños,
                        # la cantidad de veces que se debe detectar un mismo rostro y las dimenciones 
                        # minimas y maximas de los rostros, esta funcion regresa un arreglo que indica el punto donde inicia un rostro y sus dimenciones, ancho y alto
for (x,y,w,h) in faces:# a partirt de la variable faces se recolectan los valores del punto de origen del rostro y su ancho y alto para guardarlos en las variables x,y,w,h
    cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)# Con la funcion rectangle se dibujaron recuadros, enmarcando los rostros encontrados
                                                    # esto se logra indicandole la imagen donde se dibujara el rectangulo, el punto de inicio y final, 
                                                    # el color y el grosor de la linea

cv2.imshow('image',image) # se muestra en pantalla la imagen con los rostros ya detectados
cv2.waitKey(0) # Espera a que se precione una tecla
cv2.destroyAllWindows() # Destrulle todas las ventanas