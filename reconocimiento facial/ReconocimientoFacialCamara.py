import cv2
import numpy as np

cap = cv2.VideoCapture(0) # Se inicia el proceso para la captura de video, el parametro a especificar es la camara que se usara

faceClassif = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')# Se crea el clasificador a partir del archivo haarcascade_frontalface_default

while True: # Ciclo infinito para realizar la deteccion de rostros hasta que se precione la q
    ret,frame= cap.read() # la funcion read() regresa dos valores, el primero es un valor booleano 1 o 0 si se leyo o no la imagen y la imagen que la camara ve en ese momento
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Se mueve el formato de color de la imagen de BGR a escala de grices y se guarda en un nuevo objeto  
    faces = faceClassif.detectMultiScale( gray,1.05,15)#Se aplica el clasificador a la imagen
        #se escribe primero la variable donde se guardo el clasificador, seguido de esto la funcion detectMultiscale, esta funcion recibe:
                        # La imagen a trabajar(en escala de grices), el factor de escala, este escalado se hace para detectar rostros de distintos tamaños y
                        # la cantidad de veces que se debe detectar un mismo rostro esta funcion regresa un arreglo que indica el punto donde inicia un
                        # rostro y sus dimenciones, ancho y alto

    for (x,y,w,h) in faces:# a partirt de la variable faces se recolectan los valores del punto de origen del rostro y su ancho y alto para guardarlos en las variables x,y,w,h
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)# Con la funcion rectangle se dibujaron recuadros, enmarcando los rostros encontrados
                                                    # esto se logra indicandole la imagen donde se dibujara el rectangulo, el punto de inicio y final, 
                                                    # el color y el grosor de la linea

    cv2.imshow('frame',frame) # se muestra en pantalla la imagen con los rostros ya detectados
    if cv2.waitKey(1) & 0xFF == ord('q'): # En caso de se que precione la tecla q se terminara la deteccion facial
        break
cap.release() # Lafuncion release() cierra el archivo de video o termina la captura de video
cv2.destroyAllWindows() # Destrulle todas las ventanas