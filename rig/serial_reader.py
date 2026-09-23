"""Read live temperature from the Arduino cure rig over USB serial.

read_mock() provides a synthetic ramp-hold profile so the rest of the
pipeline is runnable before the rig is wired up.
"""
import time
import serial


def read_serial(port="COM3", baud=9600, timeout=1.0):
    ser = serial.Serial(port, baud, timeout=timeout)
    try:
        while True:
            line = ser.readline().decode(errors="ignore").strip()
            if line:
                yield float(line)
    finally:
        ser.close()


def read_mock(start_c=20.0, ramp_c_per_s=0.05, hold_c=180.0, sleep_s=1.0):
    """ponytail: fixed ramp-then-hold mock, swap for read_serial once the rig is wired up."""
    t = 0.0
    while True:
        yield min(start_c + ramp_c_per_s * t, hold_c)
        t += sleep_s
        time.sleep(sleep_s)


def demo():
    gen = read_mock(ramp_c_per_s=10.0, hold_c=50.0, sleep_s=0.01)
    readings = [next(gen) for _ in range(3)]
    assert readings == sorted(readings), "mock ramp should be non-decreasing"
    print(f"demo ok: {readings}")


if __name__ == "__main__":
    demo()
