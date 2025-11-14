# Motors.py
import RPi.GPIO as GPIO


class MotorConfig:
    def __init__(self, pwm_freq=5000, max_speed=40.0):
        # GPIO pins (BCM numbering)
        self.M1A = 17  # Motor 1 - A (PWM1)
        self.M1B = 22  # Motor 1 - B (PWM2)
        self.M2A = 27  # Motor 2 - A (PWM3)
        self.M2B = 23  # Motor 2 - B (PWM4)

        if not (1 <= pwm_freq <= 20000):
            raise ValueError("pwm_freq must be between 1 and 20000 Hz")
        self.PWM_FREQ = pwm_freq

        if not (0 <= max_speed <= 100):
            raise ValueError("max_speed must be between 0 and 100")
        self.MAX_SPEED = max_speed  # max duty cycle (%)

        # PWM handles
        self.pwm_M1A = None
        self.pwm_M1B = None
        self.pwm_M2A = None
        self.pwm_M2B = None

    def setup(self):
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)

        GPIO.setup(self.M1A, GPIO.OUT)
        GPIO.setup(self.M1B, GPIO.OUT)
        GPIO.setup(self.M2A, GPIO.OUT)
        GPIO.setup(self.M2B, GPIO.OUT)

        self.pwm_M1A = GPIO.PWM(self.M1A, self.PWM_FREQ)
        self.pwm_M1B = GPIO.PWM(self.M1B, self.PWM_FREQ)
        self.pwm_M2A = GPIO.PWM(self.M2A, self.PWM_FREQ)
        self.pwm_M2B = GPIO.PWM(self.M2B, self.PWM_FREQ)

        self.pwm_M1A.start(0)
        self.pwm_M1B.start(0)
        self.pwm_M2A.start(0)
        self.pwm_M2B.start(0)

    def stop_all(self):
        """Set all duty cycles to 0 (no torque)."""
        for pwm in (self.pwm_M1A, self.pwm_M1B, self.pwm_M2A, self.pwm_M2B):
            if pwm is not None:
                pwm.ChangeDutyCycle(0)

    def set_motor_speed(self, speed, pwmA, pwmB):
        """speed: -100..100, scaled by MAX_SPEED inside."""
        speed = max(-100, min(100, speed))

        # limit power
        speed = speed * self.MAX_SPEED / 100.0

        if speed > 0:
            pwmA.ChangeDutyCycle(abs(speed))
            pwmB.ChangeDutyCycle(0)
        elif speed < 0:
            pwmA.ChangeDutyCycle(0)
            pwmB.ChangeDutyCycle(abs(speed))
        else:
            pwmA.ChangeDutyCycle(0)
            pwmB.ChangeDutyCycle(0)

    def set_speeds(self, m1_speed, m2_speed):
        # Motor 1 is reversed to match real forward direction
        self.set_motor_speed(-m1_speed, self.pwm_M1A, self.pwm_M1B)
        self.set_motor_speed(m2_speed, self.pwm_M2A, self.pwm_M2B)

    def cleanup(self):
        """Stop PWM and clean up GPIO."""
        self.stop_all()
        for pwm in (self.pwm_M1A, self.pwm_M1B, self.pwm_M2A, self.pwm_M2B):
            if pwm is not None:
                pwm.stop()
        GPIO.cleanup()
