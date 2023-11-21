import cv2 # Se importa el modulo de opencv
import os # Se importa el modulo os para usar funciones del sistema operativo
import imutils # Se importa el modulo imutils para hacer modificaciones a las imagenes en conjunto con opencv

personName = 'Jesus Palomino' # Nombre para la carpeta que contendra las imagenes del sujeto
dataPath = 'C:/Users/1hgue/OneDrive/Escritorio/ITCH/8vo Semestre/Proyecto del servicio/reconociendo personas/data' # Se establece la direccion donde se guardara la base de datos
personPath = dataPath + '/' + personName # Se establece la direccion de la carpeta del sujeto
print(personPath) # Se imprime la ruta de la carpeta del sujeto
if not os.path.exists(personPath): # Si la carpeta no existe sera creada
    print('Carpeta Creada: ',personPath) # Se imprime el mansaje que indica que la carpeta se ha creado
    os.makedirs(personPath) # Se crea la carpeta para el sujeto para el que se almacenaran los rostros

cap = cv2.VideoCapture('JesusPalominoEchartea.mp4')
#cap = cv2.VideoCapture(0,cv2.CAP_DSHOW) # Se inicia la captura en vivo con la camara

faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml') # Se realiza la deteccion de rostros frontales
count = 0 # Se establece el numero con el que iniciara la base de datos

while True: # Ciclo infinito para el almacenamiento de rostros
    ret, frame = cap.read() # Se almacena un valor booleano que indica si la camara esta funcionando, y la imagen que tiene la camara enfrente en ese momento
    if ret == False: break # Si la camara no esta funcionando el programa acabara
    frame = imutils.resize(frame, width=640) # Se redimenciona la imagen a un ancho maximo de 640px
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Se pasa la imagen capturada a escala de grices y se almacena parte
    auxFrame = frame.copy() # Se crea una copia intacta de la imagen para tomar recortes de ella
    
    faces = faceClassif.detectMultiScale( gray,scaleFactor = 1.2,minNeighbors=10) # Se realiza la deteccionde rostros sobre la imagen en escala de grises

    for (x, y, w, h) in faces: # Se realiza un ciclo for en el cual se inicia por asignar las coordenadas del punto inicial del rostro ademas del ancho y alto
        cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2) # Se dibuja un rectangulo sobre la imagen que tiene la camara
        rostro = auxFrame[y:y+h,x:x+w] # Se almacena el recorte del rostro detectado en una variable
        rostro = cv2.resize(rostro,(150,150),interpolation=cv2.INTER_CUBIC) # Se redimensiona el rostro detectado
        cv2.imwrite(personPath + '/rostro_{}.jpg'.format(count),rostro) # Se almacena cada recorte en la carpeta con el nombre del sujeto con el nombre "rostro_#.jpg" con un numero distintivo definido por la variable count
        count = count + 1 # Se aumenta en uno el contador para indentificar las imagenes
    cv2.imshow('frame',frame) # Se muestra la imagen de la camara con los rectangulos dibujados en los rostros encontrados

    k = cv2.waitKey(1) # Se almacena la tecla presionada con un milisegundo de espera
    if k == 27 or count >= 2500: # Si se presiona el espacio o se llega a la imagen numero 400 se acabara el programa
        break

cap.release()
cv2.destroyAllWindows()