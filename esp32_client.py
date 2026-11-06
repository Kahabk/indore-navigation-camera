import glob
import time
import serial

import config


def find_port():
    if config.SERIAL_PORT:
        return config.SERIAL_PORT
    ports = sorted(glob.glob("/dev/ttyUSB*") + glob.glob("/dev/ttyACM*"))
    return ports[0] if ports else None


class Esp32Client:
    def __init__(self, port=None):
        port = port or find_port()
        if not port:
            raise RuntimeError("No ESP32 serial port found (/dev/ttyUSB* or /dev/ttyACM*)")
        print(f"[esp32] opening {port}")
        self.ser = serial.Serial(port, config.BAUD, timeout=0.01)
        time.sleep(2.0)  # ESP32 resets when the port opens
        self.ser.reset_input_buffer()

    def send(self, left, right):
        """left/right normalized -1.0 .. 1.0"""
        if config.INVERT_LEFT:
            left = -left
        if config.INVERT_RIGHT:
            right = -right
        self.ser.write(f"M {left:.2f} {right:.2f}\n".encode())
        self.ser.reset_input_buffer()   # discard ESP32 log text

    def stop(self):
        self.ser.write(b"S\n")

    def close(self):
        try:
            self.stop()
            self.ser.close()
        except Exception:
            pass


if __name__ == "__main__":
    # Motor test: KEEP WHEELS OFF THE GROUND
    bot = Esp32Client()
    steps = [("both forward", 1.0, 1.0), ("left wheel only", 1.0, 0.0),
             ("right wheel only", 0.0, 1.0), ("spin right", 1.0, -1.0)]
    try:
        for name, l, r in steps:
            print(name)
            t0 = time.time()
            while time.time() - t0 < 1.5:
                bot.send(l, r)
                time.sleep(0.1)
            bot.stop()
            time.sleep(1.0)
    finally:
        bot.close()
