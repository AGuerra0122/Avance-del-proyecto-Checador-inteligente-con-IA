import cv2 # Se importa el modulo de opencv
import os # Se importa le modulo os para acceder y crear directorios

if not os.path.exists('Rostros en video'):  # En caso de no existir la carpeta Rostros en video se crea dicha carpeta
    print('Carpeta creada: Rostros en video')# Se imprime un mensaje que indica que se ha creado la carpeta
    os.makedirs('Rostros en video') # Se crea la carpeta "Rostros en videos"

cap = cv2.VideoCapture(0,cv2.CAP_DSHOW) # Se inicializa la captura en vivo con la camara

faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml') # Se carga el clasificador de haarcascades para detectar rostros frontales

count = 0 # Se instancia y define la variable que servira para diferenciar a los rostros detectados
while True: # Se crea un ciclo infinito
    ret,frame = cap.read() # Se almacena un booleano sobre si se esta leyendo una imagen y la imagen como tal
    frame = cv2.flip(frame,1) # Se invierte sobre el eje vertical la imagen leida
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Se pasa la imagen leida a escala de grises
    AuxFrame = frame.copy() # Se crea una copia intacta de la imagen leida

    faces = faceClassif.detectMultiScale( gray,scaleFactor = 1.1,minNeighbors=8) # Se realiza la deteccion de rostros en la imagen leida
    
    k = cv2.waitKey(1) () # Se guardara el valor de la tecla precionada en un espacio de 1 milisegundo
    if k == 27: # en caso de presionarse el espacio se terminara el programa
        break

    for (x,y,w,h) in faces: # Se realiza un for en el que se asignaran los las coordenadas de inicio de cada rostro ademas del ancho y alto a las variables x, y, w & z
        cv2.rectangle(frame,(x,y),(x+w,y+h),(128,0,255),2) # Se dibuja un rectangulo sobre cada rostro detectado en la imagen
        rostro = AuxFrame[y:y+h,x:x+w] # Se crea un recorte del rostro detectado desde la imagen auxiliar
        rostro = cv2.resize(rostro,(150,150),interpolation=cv2.INTER_CUBIC) # Se redimenciona el recorte a 150x150px
        if k == ord('s'): # Si la tecla presionada fue la 's' se iniciara el almacenamiento del rostro
            cv2.imwrite('Rostros en video/rostro_{}.jpg'.format(count),rostro) # Se crea una imagen dentro de la carpeta Rostros en imagenes con el nombre rostro y un numero distintivo
            cv2.imshow('rostro',rostro) # Se muestra el rostro detectado 
            count = count + 1 # Se aumenta en uno el contador 
    cv2.rectangle(frame,(10,5),(450,25),(255,255,355)) # Se crea un rectangulo para enmarcar un mensaje
    cv2.putText(frame,'Presione s, para almacenar los Rostros en imagenes',(10,20),2,0.5,(128,0,255),1,cv2.LINE_AA) # Se coloca un mensaje para decidir si se guardaran los rostros detectados
    cv2.imshow('frame',frame) # Se muestra la imagen con el rostro enmarcado y el mensaje para almacenar
cap.release()
cv2.destroyAllWindows()
