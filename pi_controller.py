import RPi.GPIO as GPIO
from pynput.keyboard import Key, Controller
import time

keyboard = Controller()

LEFT_PIN = 17
RIGHT_PIN = 18
ENTER_PIN = 27

GPIO.setmode(GPIO.BCM)
GPIO.setup(LEFT_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP) 
GPIO.setup(RIGHT_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(ENTER_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)

last_left_state = True
last_right_state = True
last_enter_state = True
debounce_time = 0.05 # 50ms

try:
    while True:
        left_pressed = not GPIO.input(LEFT_PIN) 
        right_pressed = not GPIO.input(RIGHT_PIN)
        enter_pressed = not GPIO.input(ENTER_PIN)

        if left_pressed and last_left_state:
            print("Left pressed")
            keyboard.press(Key.left)
            keyboard.release(Key.left)
            last_left_state = False
            time.sleep(debounce_time)
        elif not left_pressed:
            last_left_state = True

        if right_pressed and last_right_state:
            print("Right pressed")
            keyboard.press(Key.right)
            keyboard.release(Key.right)
            last_right_state = False
            time.sleep(debounce_time)
        elif not right_pressed:
            last_right_state = True

        if enter_pressed and last_enter_state:
            print("Enter pressed")
            keyboard.press(Key.enter)
            keyboard.release(Key.enter)
            last_enter_state = False
            time.sleep(debounce_time)
        elif not enter_pressed:
            last_enter_state = True

        time.sleep(0.01) # Small delay to reduce CPU usage

except KeyboardInterrupt:
    GPIO.cleanup()