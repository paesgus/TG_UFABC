import time
from gpiozero import OutputDevice as stepper
from gpiozero import GPIOPinInUse


def Motor(direcao,tempo_de_abertura,sleepTime):
  try:
    IN1 = stepper(12)
    IN2 = stepper(16)
    IN3 = stepper(20)
    IN4 = stepper(21)
    
    stepPins = [IN1,IN2,IN3,IN4] # pinos Motor GPIO 
    stepDir = direcao            # 1 = horario ; -1 anti-horario
    mode = 0                     # mode = 1: Baixa Velocidade ==> Alta Potencia
                                 # mode = 0: Alta Velocidade ==> Baixa Potencia
    if mode:                     
      seq = [[1,0,0,1],          # Sequencia de passo definida no datasheet
             [1,0,0,0], 
             [1,1,0,0],
             [0,1,0,0],
             [0,1,1,0],
             [0,0,1,0],
             [0,0,1,1],
             [0,0,0,1]]
    else:                        
      seq = [[1,0,0,0],          
             [0,1,0,0],
             [0,0,1,0],
             [0,0,0,1]]
    stepCount = len(seq)

    waitTime = 0.004             # Tempo entre os passos

    stepCounter = 0
    abertura_ou_fechamento = 0
    # Loop principal
    while abertura_ou_fechamento < tempo_de_abertura:    
      for pin in range(0,4):
        xPin=stepPins[pin]          
        if seq[stepCounter][pin]!=0:
          xPin.on()
        else:
          xPin.off()
      stepCounter += stepDir
      if (stepCounter >= stepCount):
        stepCounter = 0
      if (stepCounter < 0):
        stepCounter = stepCount+stepDir
      time.sleep(waitTime)
      abertura_ou_fechamento += waitTime
    if sleepTime > 0:
      time.sleep(sleepTime)
  except GPIOPinInUse:
    print('Motor encontra-se atualmente em uso')

def openAndClose(tempo_de_abertura,sleepTime):
  Motor(1,tempo_de_abertura,sleepTime/2)
  Motor(-1,tempo_de_abertura,sleepTime/2)
