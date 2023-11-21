import cv2 # Se importa el modulo de opencv
import os # Se importa el modulo os para usar funciones del sistema operativo
import numpy as np # Se importa el modulo numpy y se renombra como np, este modulo sirve para hacer opereciones matematicas y de algebra lineal

dataPath = 'C:/Users/1hgue/OneDrive/Escritorio/ITCH/8vo Semestre/Proyecto del servicio/reconociendo personas/data'# Se establece la direccion que se tomara como base de datos
peopleList = os.listdir(dataPath) # Se crea un arreglo con los nombres de las carpetas de cada sujeto
print('Lista de personas: ',peopleList) # Se imprime la lista de personas que se usaran para el entrenador

labels = [] # Se crea el arreglo labels
facesData = [] # Se crea el arreglo facesData
label = 0 # Se crea la variable de control label

for nameDir in peopleList: # Se realiza un ciclo for en el que se asignaran los nombres de la lista de personas a nameDir una por una
    personPath = dataPath + '/' + nameDir # Se establece la direccion de la persona a partir de la direcciion de la base de datos y el nombre de la persona
    print('Leyendo las imagenes') # Se imprime el mensaje de que se estan leyendo las imagenes

    for fileName in os.listdir(personPath): # Se realiza un ciclo for en el que se tomara el nombre de cada imagen contenida en la carpeta de cada sujeto
        print('Rostros: ', nameDir + '/' + fileName) # Se imprime el nombre de rostro que se esta usando
        labels.append(label) # Se agrega el valor de label al final del arreglo labels
        facesData.append(cv2.imread(personPath + '/' + fileName,0)) # Se agregan las imagenes en orden al arreglo facesData
        image = cv2.imread(personPath + '/' + fileName,0) # Se almacena la imagen que se esta guardando en el arreglo anterior en otra variable para mostrarla
        cv2.imshow('image',image) # Se muestra la imagen agregada al arreglo
        cv2.waitKey(10) # se hace una pausa de 10 milisegundos 
    label = label + 1 # Se aumenta en uno el valor de la variable label 

#print('labels= ', labels) # Se imprime el arreglo de etiquetas
#print('Número de etiquetas 0: ',np.count_nonzero(np.array(labels)==0)) # Se cuentan las etiquetas con el valor de 0
#print('Número de etiquetas 1: ',np.count_nonzero(np.array(labels)==1)) # Se cuentan las etiquetas con el valor de 1

#face_recognizer = cv2.face.EigenFaceRecognizer_create() # En esta línea de código se crea un objeto de tipo EigenFaceRecognizer
                                                        # utilizando la función EigenFaceRecognizer_create() del módulo
                                                        # cv2.face de la librería OpenCV. Este objeto se utiliza para
                                                        # entrenar y reconocer rostros utilizando el método de Eigenfaces,
                                                        # el cual se basa en la reducción de dimensionalidad de las imágenes
                                                        # de los rostros para generar una representación más compacta y así
                                                        # poder realizar la tarea de reconocimiento con mayor eficiencia.
#face_recognizer = cv2.face.FisherFaceRecognizer_create() #  Fisherfaces es un algoritmo de reconocimiento de caras que utiliza
                                                          # el análisis discriminante lineal (LDA) para extraer características
                                                          # discriminantes de las imágenes de entrenamiento y, luego, utiliza estas
                                                          # características para clasificar nuevas imágenes de prueba. La idea detrás 
                                                          # de Fisherfaces es maximizar la relación entre la varianza entre clases y la 
                                                          # varianza dentro de las clases en las características extraídas.
face_recognizer = cv2.face.LBPHFaceRecognizer_create() # crea un objeto de la clase LBPHFaceRecognizer, la cual entrena un modelo de
                                                        # reconocimiento facial utilizando el algoritmo LBPH. Este algoritmo extrae características
                                                        # locales de una imagen de rostro y las utiliza para identificar a una persona. Aunque puede 
                                                        # verse afectado por factores externos como la iluminación y la oclusión, es ampliamente 
                                                        # utilizado debido a su simplicidad y eficacia en el reconocimiento facial.

# ********* Entrenando el reconocedor de rostros *********
print("Entrenando...") # Se indica que se esta realizando el entrenamiento
face_recognizer.train(facesData, np.array(labels)) # Se realiza el entrenamiento cargando los arreglos con las imagenes y las etiquetas que corresponeden a dichas imagenes 
                                                   # Durante el entrenamiento, el modelo aprende a reconocer los patrones faciales de las personas representadas en las 
                                                   # imágenes de los rostros proporcionados. Cada imagen de rostro es etiquetada con una etiqueta correspondiente a la 
                                                   # identidad de la persona en la imagen, y el modelo utiliza esta información para aprender a asociar las 
                                                   # características de cada imagen de rostro con la identidad de la persona que representa.

# ********* Almacenando el modelo obtenido *********
#face_recognizer.write('modeloEigenFace.xml') # Se almacena el modelo entrenado con los rostros para EigenFaces
#face_recognizer.write('modeloFisherFace.xml') # Se almacena el modelo entrenado con los rostros para FisherFaces
face_recognizer.write('modeloLBPHFace.xml') # Se almacena el modelo entrenado con los rostros para LBPHFaces
print("Modelo almacenado...") # Se muestra un mensaje que indica que el modelo esta listo