# ====================== Main lane center detection code ======================
import cv2
import numpy as np

# ====================== GLOBAL CONFIG ======================
DEBUG = True
DISPLAY = True

# HSV threshold parameters for lane marking detection
h_min = 0
s_min = 0
v_min = 150
h_max = 179
s_max = 255
v_max = 255

curveList = []
CURVELIST_LENGTH = 10


def nothing(a):
    """Dummy callback for OpenCV trackbars (does nothing)."""
    pass


def getHistogram(img, minPer=0.1, display=None, region=1):
    """
    Compute lane center position from a column-wise histogram.

    Args:
        img (np.ndarray): Binary or grayscale warped image.
        minPer (float): Fraction of the maximum column value used as threshold
                        to ignore weak columns.
        display (bool | None): If True, returns a debug histogram image.
                               If None, uses global DISPLAY flag.
        region (int): If 1 -> use full height (prediction, more stable).
                      If >1 -> use only the bottom part of the image
                                (current position, more precise).

    Returns:
        basePoint (int): Estimated lane center x-coordinate.
        imgHist (np.ndarray | None): Optional debug image with drawn histogram,
                                     or None if display is False.
    """
    if display is None:
        display = DISPLAY

    h, w = img.shape[:2]

    # Region Of Interest based on region parameter
    if region == 1:
        roi = img
    else:
        start_row = h - h // region
        roi = img[start_row:, :]

    # Column-wise histogram
    hisValues = np.sum(roi, axis=0)
    midpoint = w // 2
    left_half = hisValues[:midpoint]
    right_half = hisValues[midpoint:]

    # Fallback to image center if one side is empty
    if left_half.max() == 0 or right_half.max() == 0:
        basePoint = midpoint
        imgHist = np.zeros((h, w, 3), np.uint8) if display else None
        return basePoint, imgHist

    # Prediction mode (region == 1): weighted centroid on both halves
    if region == 1:
        thr_left = minPer * (left_half.max() if left_half.max() > 0 else 1)
        idxs_l = np.where(left_half >= thr_left)[0]
        if idxs_l.size == 0:
            left_mean = np.argmax(left_half)
        else:
            weights_l = left_half[idxs_l].astype(np.float64)
            left_mean = np.average(idxs_l, weights=weights_l)

        thr_right = minPer * (right_half.max() if right_half.max() > 0 else 1)
        idxs_r = np.where(right_half >= thr_right)[0]
        if idxs_r.size == 0:
            right_mean = np.argmax(right_half) + midpoint
        else:
            weights_r = right_half[idxs_r].astype(np.float64)
            right_mean = np.average(idxs_r, weights=weights_r) + midpoint
    else:
        # Current position mode (region != 1): use strongest peaks
        left_mean = np.argmax(left_half)
        right_mean = np.argmax(right_half) + midpoint

    # Resulting lane center between left and right edges
    basePoint = int((int(round(left_mean)) + int(round(right_mean))) // 2)

    imgHist = None
    if display:
        imgHist = np.zeros((h, w, 3), np.uint8)
        maxv = hisValues.max() if hisValues.max() > 0 else 1
        for x, intensity in enumerate(hisValues):
            h_line = int((intensity / maxv) * h)
            cv2.line(imgHist, (x, h), (x, h - h_line), (255, 0, 255), 1)

        # Mark left/right points and center point
        lx = int(round(left_mean))
        rx = int(round(right_mean))
        cv2.circle(imgHist, (lx, h), 6, (0, 255, 0), cv2.FILLED)       # left (green)
        cv2.circle(imgHist, (rx, h), 6, (255, 0, 0), cv2.FILLED)       # right (blue)
        cv2.circle(imgHist, (basePoint, h), 8, (0, 255, 255), cv2.FILLED)  # center (yellow)

    return basePoint, imgHist


# ====================== TRACKBARS ======================
def initializeTrackbars(initialTracbarVals, wT=480, hT=240):
    """
    Initialize OpenCV window and trackbars for perspective transform trapezoid.

    Args:
        initialTracbarVals (list[int]): Initial values for [widthTop, widthBottom, heightTop, heightBottom].
        wT (int): Target width.
        hT (int): Target height.
    """
    cv2.namedWindow("Points_Setup")
    cv2.resizeWindow("Points_Setup", 600, 300)
    cv2.createTrackbar("Width Top", "Points_Setup", initialTracbarVals[0], wT // 2, nothing)
    cv2.createTrackbar("Width Bottom", "Points_Setup", initialTracbarVals[1], wT // 2, nothing)
    cv2.createTrackbar("Height Top", "Points_Setup", initialTracbarVals[2], hT, nothing)
    cv2.createTrackbar("Height Bottom", "Points_Setup", initialTracbarVals[3], hT, nothing)


def getTrackbarValues(wT=480, hT=240):
    """
    Read trapezoid coordinates from trackbars and return them as OpenCV-compatible points.

    Args:
        wT (int): Target width.
        hT (int): Target height.

    Returns:
        points (np.ndarray): 4 points [TL, TR, BL, BR] as float32.
    """
    widthTop = cv2.getTrackbarPos("Width Top", "Points_Setup")
    widthBottom = cv2.getTrackbarPos("Width Bottom", "Points_Setup")
    heightTop = cv2.getTrackbarPos("Height Top", "Points_Setup")
    heightBottom = cv2.getTrackbarPos("Height Bottom", "Points_Setup")

    # Symmetric trapezoid around image center
    points = np.float32([
        [wT // 2 - widthTop, heightTop],        # Top-Left
        [wT // 2 + widthTop, heightTop],        # Top-Right
        [wT // 2 - widthBottom, heightBottom],  # Bottom-Left
        [wT // 2 + widthBottom, heightBottom]   # Bottom-Right
    ])
    return points


def drawPoints(img, points):
    """
    Draw trapezoid points and outline on image.

    Args:
        img (np.ndarray): BGR image.
        points (np.ndarray): 4 points [TL, TR, BL, BR].

    Returns:
        img (np.ndarray): Image with drawn trapezoid.
    """
    reorder = np.array([0, 1, 3, 2])  # TL, TR, BR, BL
    pts = np.int32(points[reorder])
    for x, y in pts:
        cv2.circle(img, (int(x), int(y)), 15, (0, 0, 255), cv2.FILLED)
    cv2.polylines(img, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
    return img


# ====================== PERSPECTIVE TRANSFORM ======================
def warpIMG(img, points, w, h):
    """
    Apply perspective warp to image based on 4 points.

    Args:
        img (np.ndarray): Source image.
        points (np.ndarray): 4 points for perspective transform.
        w (int): Target width.
        h (int): Target height.

    Returns:
        imgWarp (np.ndarray): Warped image.
    """
    pts1 = np.float32(points)
    pts2 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgWarp = cv2.warpPerspective(img, matrix, (w, h))
    return imgWarp


def thresholding(img):
    """
    Apply HSV-based thresholding to extract bright lane markings.

    Args:
        img (np.ndarray): Input BGR frame or road image.

    Returns:
        maskWhite (np.ndarray): Binary mask of lane candidates.
                                Optionally thinned if ximgproc.thinning is available.
    """
    imgHsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lowerWhite = np.array([h_min, s_min, v_min])
    upperWhite = np.array([h_max, s_max, v_max])
    maskWhite = cv2.inRange(imgHsv, lowerWhite, upperWhite)
    maskWhite = cv2.morphologyEx(maskWhite, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    maskWhite = cv2.GaussianBlur(maskWhite, (5, 5), 0)

    # Optional thinning: only if ximgproc.thinning is available in this OpenCV build
    if hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "thinning"):
        maskWhite = cv2.ximgproc.thinning(maskWhite)

    return maskWhite


# ====================== MAIN LANE PIPELINE ======================
def getLaneCurve(img):
    """
    High-level lane detection pipeline on a static road image.

    Steps:
        - Read trapezoid ROI from trackbars.
        - Threshold image in HSV space.
        - Apply perspective warp.
        - Compute histogram-based lane center and curve.
        - Accumulate curve values for smoothing.

    Args:
        img (np.ndarray): BGR road image.

    Returns:
        None  # Draws debug windows and prints curve value to console.
    """
    h, w, c = img.shape
    points = getTrackbarValues(w, h)
    imgThresh = thresholding(img)
    imgWarp = warpIMG(imgThresh, points, w, h)

    imgWarpPoints = drawPoints(img.copy(), points)

    middlePoint, img_midHist = getHistogram(imgWarp, minPer=0.5, display=DISPLAY, region=4)
    curveAveragePoint, img_CurveHist = getHistogram(imgWarp, minPer=0.5, display=DISPLAY)
    curveRaw = curveAveragePoint - middlePoint

    curveList.append(curveRaw)
    if len(curveList) > CURVELIST_LENGTH:
        curveList.pop(0)
    curve = int(sum(curveList) / len(curveList))

    print(curve)
    cv2.circle(imgWarpPoints, (curveAveragePoint, img.shape[0]), 15, (0, 255, 255), cv2.FILLED)
    cv2.imshow('Warped', imgWarp)
    cv2.imshow('Warped Points', imgWarpPoints)
    cv2.imshow('Mid Histogram', img_midHist)
    cv2.imshow('Curve Histogram', img_CurveHist)
    return None


# ====================== ENTRY POINT ======================
img = cv2.imread('road_images/road2.png')
img = cv2.resize(img, (480, 240))

if DEBUG:
    initialTrackbarValues = [140, 240, 116, 240]  # initial trapezoid parameters
    initializeTrackbars(initialTrackbarValues, wT=480, hT=240)

while True:
    getLaneCurve(img)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
