import cv2 # se importa la libreria cv2

image = cv2.imread('figurasColores2.png') # se carga la imagen y se asigna a la variable image
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # se combierte la imagen a escala de grices y se guarda en la variable gray
canny = cv2.Canny(gray, 10, 150) # Se hace la deteccion de bordes para binarizar la imagen
#cv2.imshow('canny',canny)  # Con la funcion imshow se abre con una ventana con la imagen a mostrar y el titulo que se le haya asignado a la ventana
canny = cv2.dilate(canny, None, iterations=1) # se dilata la imagen binarizada para resaltar los contornos de las figuras
#cv2.imshow('dilate',canny) # Con la funcion imshow se abre con una ventana con la imagen a mostrar y el titulo que se le haya asignado a la ventana
canny = cv2.erode(canny, None, iterations=1) # se erociona la imagen dilatada para definir o dicho de otro modo enfocar los contornos de las figuras
#cv2.imshow('erode',canny) # Con la funcion imshow se abre con una ventana con la imagen a mostrar y el titulo que se le haya asignado a la ventana
#_, th = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY) # contrario a tomar solo el contorno en esta linea se rellena la figura completamente 
#_,cnts,_ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)# OpenCV 3
cnts,_ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)# OpenCV 4 en esta linea se llama a la funcion find contours la cual 
																			# busca y almacena los contornos de las imagenes y las almacena en forma
																			# de tupla RECIBE IMAGEN BINARIZADA, MODO Y METODO
#cv2.drawContours(image, cnts, 0 , (0,255,0), 2) # la funcion drawContours permite dibujar sobre la imagen original los contornos encontrados y almacenados en la tupla
												# RECIBE IMAGEN, CONTORNOS, INDICE EN LA TUPLA EN CASO DE SER MAS DE UN CONTORNO, COLOR DEL CONTORNO, GROSOR DEL CONTORNO
for c in cnts: # ciclo for para analizar y diferenciar las figuras en base a los contornos almacenados en cnts
	epsilon = 0.008*cv2.arcLength(c,True) # esta consiste en la precicion de la aproximacion que se dara para determinar la diferencia entre figuras
										  # haciendo uso de la funcion arcLenght(curva, true si es que la curva es cerrada) se establece un valor
										  # que a su vez se multiplica por un valor de porcentaje
	print(epsilon)
	approx = cv2.approxPolyDP(c,epsilon,True) # se realiza un calculo de a que figura se aproxima cada figura de acuerdo al nuemro de lados
											  # usando la funcion approxPolyDP(curva, precicion de la aproximacion, true si la curva es cerrada)
											  # se almacena el resultado en la variable approx, este consiste en guardar cada vertice que la curva de la figura proporcione
	x,y,w,h = cv2.boundingRect(approx) # Esta funcion se encarga de tomar las coordenadas de un punto de referencia para asi medir y guardar el ancho y alto de la figura
	print (len(approx)) # Se imprime la longitud de approx, misma que se refiere a el numero de lados de la figura
	if len(approx)==3: # si el numero de lados es 3 la figura es un triangulo
		cv2.putText(image,'Triangulo', (x,y-5),1,1,(0,255,0),1) # Con la funcion putText se coloca el texto deceado sobre la imagen, en este caso
																# se coloca el numbre de la figura, la fucnion recibe:
																# nombre del objeto que tiene la imagen, texto, pocision donde iniciara el texto
																# numero de fuente, tamaño de la fuente, color en BGR, grosor y tipo de linea
		print("Triangulo")

	if len(approx)==4: # Si el numero de lados es 4 la figura es un cuadrado o rectangulo
		aspect_ratio = float(w)/h # Se calcula el radio de aspecto de la figura
		print('aspect_ratio= ', aspect_ratio)
		if aspect_ratio == 1: # En caso de que el radio sea 1 se tratara de un cuadrado
			cv2.putText(image,'Cuadrado', (x,y-5),1,1,(0,255,0),1)
			print("Cuadrado")
		else: # En caso de ser cualquier otro valor se tratara de un rectangulo
			cv2.putText(image,'Rectangulo', (x,y-5),1,1,(0,255,0),1)
			print("Rectangulo")		

	if len(approx)==5: # Si el numero de lados es 5 se tratara de un pentagono
		cv2.putText(image,'Pentagono', (x,y-5),1,1,(0,255,0),1)
		print("Pentagono")

	if len(approx)==6: # Si el numero de lados es 6 se tratara de un hexagono
		cv2.putText(image,'Hexagono', (x,y-5),1,1,(0,255,0),1)
		print("Hexagono")

	if len(approx)>15: # Finalmente para toda figura de mas de 15 lados se considerara que es un circulo
		cv2.putText(image,'Circulo', (x,y-5),1,1,(0,255,0),1)
		print("circulo")

	cv2.drawContours(image, [approx], 0, (0,255,0),2) # Se dibujan los contornos de las figuras, tomando las coordenadas de los vertices de las figuras para dibujarlos
cv2.imshow('image',image) # Con la funcion imshow se abre con una ventana con la imagen a mostrar y el titulo que se le haya asignado a la ventana
cv2.waitKey(0) # establece que espere hasta que se presione una tecla para continuar 