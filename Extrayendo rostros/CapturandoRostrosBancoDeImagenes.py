import cv2 # Se importa el modulo de opencv
import os # Se importa le modulo os para acceder y crear directorios

imagesPath = "C:/Users/1hgue/OneDrive/Escritorio/ITCH/8vo Semestre/Proyecto del servicio/Extrayendo rostros/imagenes" # Se almacena la ruta de la carpeta que contiene las 
                                                                                                                      # imagenes de donde se tomaran los rostros
imagesPathList = os.listdir(imagesPath) # Se crea un arreglo para almacenar los nombres de las imagenes encontradas
#print('imagesPathList=',imagesPathList) # Se imprime el arreglo de los nombres de las imagenes

if not os.path.exists('Rostros en imagenes'): # En caso de no existir la carpeta Rostros en imagenes se crea dicha carpeta
    print('Carpeta creada: Rostros en imagenes') # Se imprime un mensaje que indica que se ha creado la carpeta
    os.makedirs('Rostros en imagenes') # Se crea la carpeta "Rostros en imagenes"

faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml') # Se carga el clasificador de haarcascades para detectar rostros frontales

count = 0 # Se instancia y define la variable que servira para diferenciar a los rostros detectados
for imageName in imagesPathList: # Se realiza un ciclo for que a la vez asigna el nombre de cada imagen a una variable
    #print('imageName=',imageName) Se imprime el nombre de la imagen a trabajar
    image = cv2.imread(imagesPath+'/'+imageName) # Se guarda desde la carpeta de la base de datos la imagen a trabajar  en la variable image
    print(image)
    imageAux = image.copy() # Se crea una copia intacta de la imagen
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # Se transforma la imagen a escala de grices y se guarda en la variable gray
 
    faces = faceClassif.detectMultiScale( gray,scaleFactor = 1.3,minNeighbors=10,minSize=(20,20),maxSize=(2000,2000) ) # Se realiza la deteccion de rostros dentro de la imagen 
    
    for (x,y,w,h) in faces: # Se realiza un for en el que se asignaran los las coordenadas de inicio de cada rostro ademas del ancho y alto a las variables x, y, w & z
        cv2.rectangle(image,(x,y),(x+w,y+h),(128,0,255),2) # Se dibuja un rectangulo sobre cada rostro detectado en la imagen
    cv2.rectangle(image,(10,5),(450,25),(255,255,355)) # Se crea un rectangulo para enmarcar un mensaje
    cv2.putText(image,'Presione s, para almacenar los Rostros en imagenes',(10,20),2,0.5,(128,0,255),1,cv2.LINE_AA) # Se coloca un mensaje para decidir si se guardaran los rostros detectados
    cv2.imshow('image',image) # Se muestra la imagen con los rostros enmarcados y el mensaje para almacenar
    k = cv2.waitKey(0) # Se almacena la ultima tecla presionada en la variable k
    if k == ord('s'): # Si la tecla presionada fue la 's' se iniciara el almacenamiento del rostro
        for(x,y,w,h) in faces: # Se realiza un ciclo en el que se tomaran las coordenadas de inicio de los rostros, el ancho y alto de los rostros 
            rostro = imageAux[y:y+h,x:x+w] # Se crea un recorte del rostro detectado desde la imagen auxiliar
            rostro = cv2.resize(rostro,(150,150),interpolation=cv2.INTER_CUBIC) # Se redimenciona el recorte a 150x150px
            cv2.imshow('rostro',rostro) # Se muestra el rostro detectado 
            #cv2.waitkey(0)
            cv2.imwrite('Rostros en imagenes/rostro_{}.jpg'.format(count),rostro) # Se crea una imagen dentro de la carpeta Rostros en imagenes con el nombre rostro y un numero distintivo
            count = count + 1 # Se aumenta en uno el contador 
    elif k == 27: # En caso de presionarse el espacio se brincara el proceso de almacenaje de rostros
        break

cv2.destroyAllWindows()
