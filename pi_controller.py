import RPi.GPIO as GPIO
from pynput.keyboard import Key, Controller
import time

keyboard = Controller()

UP_PIN = 17
DOWN_PIN = 18
ENTER_PIN = 27

GPIO.setmode(GPIO.BCM)
GPIO.setup(UP_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP) 
GPIO.setup(DOWN_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(ENTER_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)

last_left_state = True
last_right_state = True
last_enter_state = True
debounce_time = 0.05 # 50ms

try:
    while True:
        left_pressed = not GPIO.input(UP_PIN) 
        right_pressed = not GPIO.input(DOWN_PIN)
        enter_pressed = not GPIO.input(ENTER_PIN)

        if left_pressed and last_left_state:
            print("Up pressed")
            keyboard.press(Key.up)
            keyboard.release(Key.up)
            last_left_state = False
            time.sleep(debounce_time)
        elif not left_pressed:
            last_left_state = True

        if right_pressed and last_right_state:
            print("Down pressed")
            keyboard.press(Key.down)
            keyboard.release(Key.down)
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
