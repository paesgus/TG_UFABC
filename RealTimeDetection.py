from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
import os
from threading import Thread
from LED import LED_Mode
from moduloMotor import Motor, openAndClose

def detection(frame, faceModel, maskModel):

    # Gera um blob a partir do frame capturado na imagem
	(h, w) = frame.shape[:2] # altura e largura
	blob = cv2.dnn.blobFromImage(frame, 1.0, (250, 250),
		(104.0, 177.0, 123.0))

	# Passa o blob na rede neural do modelo de rostos
	faceModel.setInput(blob)
	faceDetections = faceModel.forward()
    
    # Inicialização de variaveis
	faces = []
	locs = []
	preds = []

	# loop com base das detecções de rostos encontradas na imagem
	for i in range(0, faceDetections.shape[2]):
        
        # Extrair porcentagem da deteccao
		confidence = faceDetections[0, 0, i, 2]

        # Se a porcentagem de detecção for maior que 50%
		if confidence > 0.5:
            
            # Extrai pontos para formar uma "caixa"
			box = faceDetections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

            # Garantir que a caixa esteja dentro do frame
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            # Cortar o frame nas dimensoes da caixa, onde esta o rosto da pessoa
            # Aplicar pre-processamentos para que o modelo leia a imagem
			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (150, 150))
			face = img_to_array(face)
			face = preprocess_input(face)

            # Adicionar a imagem do rosto e as coordenadas nas variaves
			faces.append(face)
			locs.append((startX, startY, endX, endY))

	# Detectar mascara caso haja rosto
	if len(faces) > 0:
        # Aplicar modelo de deteccao de mascara
		faces = np.array(faces, dtype="float32")
		preds = maskModel.predict(faces, batch_size=32)
    # returna os resultados
	return (locs, preds)

# Carregar modelo de reconhecimento de rostos
prototxtPath = "deploy.prototxt"
weightsPath = "res10_300x300_ssd_iter_140000.caffemodel"
faceModel = cv2.dnn.readNet(prototxtPath, weightsPath)

# Carregar modelo de reconhecimento de mascaras
maskModel = load_model("FaceMask_ModelV2.h5")

# Iniciar camera
cam = VideoStream(src=0).start()

# Codigo que roda em tempo real
while True:
    # Le frame da camera
	frame = cam.read()
	frame = imutils.resize(frame, width=800)

	(locs, preds) = detection(frame, faceModel, maskModel)

	for (box, pred) in zip(locs, preds):
		# Extrai coordenadas da caixa
		(startX, startY, endX, endY) = box
        # Extrai valoores da predicao
		(Com_mascara, Mascara_errada, Sem_mascara) = pred

        # Determina a cor e o rotulo da caixa
		if Sem_mascara == max(Com_mascara, Mascara_errada, Sem_mascara):
			label = "Sem_mascara"
			color = (0, 0, 255)
		elif Mascara_errada == max(Com_mascara, Mascara_errada, Sem_mascara):
			label = "Mascara_errada"
			color = (0, 255, 255)
		else:
			label = "Com_mascara"
			color = (0, 255, 0)

		# Inclui probabilidade da classificacao
		label = "{}: {:.2f}%".format(label, max(Com_mascara, Mascara_errada, Sem_mascara) * 100)

        if Com_mascara>Mascara_errada and Com_mascara>Sem_mascara:
            Thread(target=openAndClose,args=(2.5,10)).start()
        else: 
            Thread(target=LED_Mode,args=(1,)).start()

        # Inclui as informacoes no frame
		cv2.putText(frame, label, (startX, startY - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
		cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

	# Mostra o frame na tela
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# aperta s para sair do programa
	if key == ord("s"):
		break

# Limpa a tela
cv2.destroyAllWindows()
cam.stop()






