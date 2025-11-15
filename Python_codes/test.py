import time
import numpy as np
import cv2
from picamera2 import Picamera2

from Lane_detect import LaneDetector
from Drive_control import DriveControl
from Motors import MotorConfig

# ================== CONFIG ==================

# Set this to True if you want to use fisheye calibration (undistortion)
# Set to False if you want to use raw camera frames (no undistortion)
USE_CALIBRATION = True

CALIB_FILE = "camera_fisheye_calib.npz"

# Default resolution used when calibration is disabled or loading fails
DEFAULT_DIM = (640, 480)
IS_BGR = True

# ----------------- Calibration loading ----------------- #

def load_fisheye_calibration(path=CALIB_FILE):
    """
    Load fisheye calibration from .npz file.

    Returns:
        K   : 3x3 camera matrix
        D   : distortion coefficients (fisheye model, shape 4x1 or 1x4)
        DIM : (width, height)
    """
    data = np.load(path)
    K = data["K"]
    D = data["D"]
    DIM = tuple(data["DIM"])  # (width, height)
    print("[INFO] Loaded calibration from", path)
    print("[INFO] DIM =", DIM)
    return K, D, DIM


# ----------------- Camera initialization ----------------- #

def init_picamera(dim):
    """
    Initialize Picamera2 with given frame size.

    Args:
        dim: (width, height)

    Returns:
        picam2: configured and started Picamera2 instance
    """
    width, height = dim
    picam2 = Picamera2()

    config = picam2.create_video_configuration(
        main={"size": (width, height), "format": "RGB888"}
    )
    picam2.configure(config)

    picam2.start()
    time.sleep(1.0)

    print("[INFO] Picamera2 started with size", dim)
    return picam2


# ----------------- Undistortion maps ----------------- #

def create_undistort_maps(K, D, dim, balance=0.0):
    """
    Create undistort rectify maps for fisheye lens.

    Args:
        K      : 3x3 camera matrix
        D      : fisheye distortion coefficients
        dim    : (width, height)
        balance: 0.0..1.0, trade-off between FOV and cropping

    Returns:
        map1, map2: maps for cv2.remap
    """
    width, height = dim
    DIM = (width, height)
    R = np.eye(3)

    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        K, D, DIM, R, balance=balance
    )

    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K, D, R, new_K, DIM, cv2.CV_16SC2
    )

    print("[INFO] Undistort maps created (balance =", balance, ")")
    return map1, map2


# ----------------- Modules initialization ----------------- #

def init_modules(frame_width, frame_height):
    """
    Initialize LaneDetector, MotorConfig and DriveControl.

    Returns:
        lane_detector, drive_controller
    """
    lane_detector = LaneDetector(
        frame_width=frame_width,
        frame_height=frame_height,
        debug=True,
        display=True,
        ipm_trapezoid_init=(140, 240, 116, 240),
    )

    motor_config = MotorConfig(
        pwm_freq=5000,
        max_speed=40.0,
    )

    drive_controller = DriveControl(
        motor_config=motor_config,
        DEBUG=True,
    )

    print("[INFO] Modules initialized (LaneDetector, DriveControl)")
    return lane_detector, drive_controller


# ----------------- Main processing loop ----------------- #

def run_main_loop(picam2, lane_detector, drive_controller,
                  map1=None, map2=None, show_debug=True):
    """
    Main loop: capture frame, optionally undistort,
    run lane detection and drive control.

    If map1 and map2 are None -> no undistortion, raw frame is used.
    """
    use_undistort = (map1 is not None) and (map2 is not None)
    print("[INFO] Main loop started. Press 'q' to quit.")
    print(f"[INFO] Undistortion enabled: {use_undistort}")

    while True:
        # 1) Capture RGB frame from Picamera2
        frame_bgr = picam2.capture_array()

        # 2) Convert to BGR for OpenCV
        if IS_BGR == False:
            frame_bgr = cv2.cvtColor(frame_bgr, cv2.COLOR_RGB2BGR)

        # 3) Optionally undistort using precomputed maps
        if use_undistort:
            frame_proc = cv2.remap(
                frame_bgr,
                map1,
                map2,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
            )
        else:
            frame_proc = frame_bgr

        # 4) Lane detection on processed image
        lane_result = lane_detector.process_frame(frame_proc)

        # 5) Drive control based on lane detection result
        drive_controller.control_step(lane_result)

        # 6) Optional debug windows
        if show_debug:
            cv2.imshow("Original (BGR)", frame_bgr)
            if use_undistort:
                cv2.imshow("Undistorted", frame_proc)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("[INFO] Quit requested.")
            break

    print("[INFO] Exiting main loop.")


# ----------------- Entry point ----------------- #

def main():
    # local flag, in case calibration fails
    use_calib = USE_CALIBRATION

    map1 = None
    map2 = None

    if use_calib:
        try:
            # 1) Load calibration
            K, D, DIM = load_fisheye_calibration(CALIB_FILE)
            cam_dim = DIM
        except Exception as e:
            print("[WARN] Failed to load calibration:", e)
            print("[WARN] Falling back to raw camera (no undistortion).")
            use_calib = False
            cam_dim = DEFAULT_DIM
    else:
        cam_dim = DEFAULT_DIM

    # 2) Init camera
    picam2 = init_picamera(cam_dim)

    # 3) Prepare undistortion maps if calibration is enabled
    if use_calib:
        map1, map2 = create_undistort_maps(K, D, cam_dim, balance=0.0)

    # 4) Init modules (lane detection + drive)
    width, height = cam_dim
    lane_detector, drive_controller = init_modules(width, height)

    # 5) Run main loop
    try:
        run_main_loop(
            picam2=picam2,
            lane_detector=lane_detector,
            drive_controller=drive_controller,
            map1=map1,
            map2=map2,
            show_debug=True,
        )
    finally:
        picam2.stop()
        cv2.destroyAllWindows()
        print("[INFO] Clean shutdown.")


if __name__ == "__main__":
    main()
