import serial
import time

try:
    ser = serial.Serial('/dev/ttyACM0', 9600, timeout=1)
    time.sleep(2) 

    while True:
        line_of_data = ser.readline()
        if line_of_data:
            string_data = line_of_data.decode('utf-8').strip()
            print(f"{string_data}")

except serial.SerialException as e:
    print(f"Error: Could not open port COM5. Is it correct?")
    print(f"Details: {e}")

except KeyboardInterrupt:
    print("\nProgram stopped by user. Closing serial port.")

finally:
    if 'ser' in locals() and ser.is_open:
        ser.close()
        print("Serial port closed.")