import math
import cv2
from Motors import MotorConfig
from collections import deque

class DriveControl:

    # ------------- basic callback for trackbars -------------
    @staticmethod
    def _nothing(a):
        """Dummy callback for OpenCV trackbars."""
        pass

    def __init__(self, motor_config: MotorConfig, DEBUG=False):
        self.DEBUG = DEBUG
        self.motor = motor_config
        self.motor.setup()
        self.history_length = 20  # Length of history to keep
        self.history = deque(maxlen=self.history_length) # History of states: lane_ok, left_lane_ok, right_lane_ok, crossroad_type, poly
        self.in_fallback = False
        self.last_good_state = None
        self.BASE_SPEED = 30  # Base speed for driving forward
        self.MAX_STEERING = 25  # Maximum steering 
        self.K_lateral = 1.2  # Lateral control gain
        self.K_head = 0.6  # Angular control gain
        self.Min_good_state_count = int(self.history_length * 0.5)  # Minimum count to consider last state as good
        self.Max_good_state_count = int(self.history_length * 0.75)  # Maximum count to keep last state
        self.y_L = 0.7  # Lookahead distance in pixels
    # Initialize OpenCV trackbars (debug only)
        if self.DEBUG:
            self._initialize_values_trackbars()

    # ====================== TRACKBARS (PERSPECTIVE) ======================
    def _initialize_values_trackbars(self):
        cv2.namedWindow("Values Setup")
        cv2.resizeWindow("Values Setup", 400, 200)

        # Max steering: 0..50
        cv2.createTrackbar(
            "MaxSteering", "Values Setup",
            int(self.MAX_STEERING), 50,
            self._nothing
        )

        # K_lateral in range 0.00..3.00, stored as x100
        cv2.createTrackbar(
            "K_lat x100", "Values Setup",
            int(self.K_lateral * 100), 300,
            self._nothing
        )

        # K_head in range 0.00..2.00, stored as x100
        cv2.createTrackbar(
            "K_head x100", "Values Setup",
            int(self.K_head * 100), 200,
            self._nothing
        )

        # y_L in range 0.50..0.90 (x100)
        cv2.createTrackbar(
            "y_L x100", "Values Setup",
            int(self.y_L * 100), 90,
            self._nothing
        )

    def _update_values_from_trackbars(self):
        if not self.DEBUG:
            return

        self.MAX_STEERING = cv2.getTrackbarPos("MaxSteering", "Values Setup")

        k_lat_x100 = cv2.getTrackbarPos("K_lat x100", "Values Setup")
        self.K_lateral = k_lat_x100 / 100.0

        k_head_x100 = cv2.getTrackbarPos("K_head x100", "Values Setup")
        self.K_head = k_head_x100 / 100.0

        y_L_x100 = cv2.getTrackbarPos("y_L x100", "Values Setup")
        # ograniczenie od dołu, żeby nie skończyć z 0.0
        self.y_L = max(0.3, y_L_x100 / 100.0)


    
    def update_history(self, lane_result: dict):

        if lane_result is None:
            self.motor.set_speeds(0, 0)
            return    
        history_result = {"lane_ok": lane_result.get("lane_ok"),
                "left_lane_ok": lane_result.get("left_lane_ok"),
                "right_lane_ok": lane_result.get("right_lane_ok"),
                "crossroad_type": lane_result.get("crossroad_type"),
                "poly": lane_result.get("poly")}
        self.history.append(history_result)

        if history_result["lane_ok"]:
            self.last_good_state = history_result

    def update_fallback(self):
        if len(self.history) < self.history_length:
            return  # Not enough data yet
        good_state_count = sum(1 for state in self.history if state["lane_ok"])
        if good_state_count < self.Min_good_state_count:
            self.in_fallback = True
        elif good_state_count > self.Max_good_state_count:
            self.in_fallback = False

    def crossroad_step(self, lane_result: dict):
        return
    
    def fallback_step(self, lane_result: dict):
        left_lane_ok_history = self.history["left_lane_ok"]
        left_lane_ok_history = self.history["left_lane_ok"]
        return
    
    def lane_following_step(self, lane_result: dict):

        poly = lane_result.get("poly")
        if poly is None or len(poly) < 3:
            self.motor.set_speeds(0, 0)
            return
        a,b,c = poly
        x_L = a * self.y_L**2 + b * self.y_L + c
        e_lat = x_L
        dx_dy = 2 * a * self.y_L + b
        e_head = math.atan(dx_dy)
        steering = -self.K_lateral * e_lat - self.K_head * e_head
        if steering > self.MAX_STEERING:
            steering = self.MAX_STEERING
        elif steering < -self.MAX_STEERING:
            steering = -self.MAX_STEERING

        base_speed = self.BASE_SPEED
        left_speed = base_speed + steering
        right_speed = base_speed - steering
        self.motor.set_speeds(left_speed, right_speed)

        if self.DEBUG:
            print(f"e_lat={e_lat:.3f}, e_head={e_head:.3f}, steering={steering:.2f}")

        return
    
    def control_step(self, lane_result: dict):
        # 1. Brak wyniku z detektora
        #    - na razie możesz po prostu zatrzymać silniki
        #    - i return
        if lane_result is None:
            self.motor.set_speeds(0, 0)
            return
        
        if self.DEBUG:
            self._update_values_from_trackbars()

        self.update_history(lane_result)
        # 3. Sprawdź typ skrzyżowania:
        #    jeśli crossroad_type != "not_detected":
        #        wywołaj self.crossroad_step(lane_result)
        #        return
        crossroad_type = lane_result.get("crossroad_type", "not_detected")
        
        if crossroad_type != "not_detected":
            self.crossroad_step(lane_result)
            return
        
        self.update_fallback()
        # 5. Jeśli self.in_fallback albo lane_ok == False:
        #        self.fallback_step(lane_result)
        #    w przeciwnym razie:
        #        self.lane_following_step(lane_result)
        if self.in_fallback or not lane_result.get("lane_ok", False):
            self.fallback_step(lane_result)
        else:
            self.lane_following_step(lane_result)
