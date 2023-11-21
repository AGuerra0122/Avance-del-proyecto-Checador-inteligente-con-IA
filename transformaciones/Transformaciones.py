import cv2
import numpy as np
import imutils

image = cv2.imread('rengoku.jpg') # se lee y almacena en un objeto la imagen a trabajar, la imagen se indica por medio del nombre y de ser necesario la ruta de almacenamiento
ancho = image.shape[1] # Haciendo uso de la funcion shape se obtiene el ancho de la imagen al mandar 1 como parametro
alto = image.shape[0] # Haciendo uso de la funcion shape se obtiene el alto de la imagen al mandar 0 como parametro

#Traslacion         x         y
'''M = np.float32([[1,0,ancho/2],[0,1,alto/2]]) # Para realizar la traslacion es necesario contar con un vector que determine hacia donde se movera la imagen
                                      # Esto se define el la tercera variable de cada vector de la matriz

imageOut = cv2.warpAffine(image,M,(ancho*2,alto*2)) # Para realizar la traslacion se utilizo la funcion warpAffine la cual recibe la imagen a trabajar
                                                    # la matriz que define la traslacion y el tamaño de la ventata que contiene la imagen'''

# Rotación   
#N = cv2.getRotationMatrix2D((ancho//2,alto//2),-90,2) # Se crea una matriz afin a la rotacion que deceamos realizar
                                                      # Esta recibira el centro de rotacion para la imagen, el angulo que rotara la imagen y el grado de escalado para la imagen
#imageOut = cv2.warpAffine(image,N,(ancho*3,alto*3)) # Para realizar la traslacion se utilizo la funcion warpAffine la cual recibe la imagen a trabajar
                                                    # la matriz que define la rotacion y el tamaño de la ventata que contiene la imagen

# Escalando una imagen 
#imageOut = cv2.resize(image,(500,300),interpolation=cv2.INTER_CUBIC) # Funcion dada por cv2 para escalar un imagen esta funcion recibe
                                                                     # la imagen a modificar, ancho, alto y el metodo de interpolacion
                                                                     # no es necesario que exista una relacion entre el ancho y alto, 
                                                                     # esto nos permite deformar la imagen

# Escalando una imagen con imutils
#imageOut = imutils.resize(image, width=120) # Funcion dada por imutils para escalar imagenes esta funcion recibe
#imageOut = imutils.resize(image, height=330)# la imagen y un parametro, ya sea ancho u alto y la funcion se encarga 
                                            # de asignar el volor desconocido conservando la relacion de aspecto

print('Image.shape=',image.shape)
imageOut = image[45:150,65:170] # Si se considera a la imagen como una matriz es posible recortar una seccion especifica de la misma
                                # Definiendo las coordenada del punto de inicio y fin

cv2.imshow('Imagen de entrada', image) # se muestra la imagen original en pantalla
cv2.imshow('Imagen de salida',imageOut) # se muestra la imagen de salida en pantalla

cv2.waitKey(0)
cv2.destroyAllWindows()
