from gpiozero import LED
from gpiozero import GPIOPinInUse
from time import sleep

def LED_Mode(tempo=3):
    try:
        led = LED(18)
        led.on()
        sleep(tempo)
        led.off()
    except GPIOPinInUse:
        print('LED encontra-se atualmente em uso')
