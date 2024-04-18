import serial
ser = serial.Serial('COM3')
# ser.rs485_mode
ser.timeout = 0.01
# ser.open()
ser.read(4)
ser.close()