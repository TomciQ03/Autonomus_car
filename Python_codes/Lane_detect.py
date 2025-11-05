# ====================== Main lane center detection code ======================
import cv2
import numpy as np
import time

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


def roi_segment(img, vertices=3):
    """
    Split the warped image into `vertices` horizontal stripes (top -> bottom).

    Returns:
        rois      : list of sub-images [roi_0, roi_1, roi_2, ...]
        centers_y : list of vertical centers [y_center_0, y_center_1, ...]
    """
    h, w = img.shape[:2]
    if vertices <= 0:
        return [], []

    segment_h = h // vertices
    rois = []
    centers_y = []

    for i in range(vertices):
        y_start = i * segment_h
        # Last segment takes the remaining pixels to cover full height
        y_end = h if i == vertices - 1 else (i + 1) * segment_h
        roi = img[y_start:y_end, :]
        y_center = (y_start + y_end) // 2

        rois.append(roi)
        centers_y.append(y_center)

    return rois, centers_y


def getHistogram(img, minPer=0.1, display=None, region=1):
    """
    Compute lane center position using a column-wise histogram.

    Args:
        img (np.ndarray): Binary or grayscale warped image.
        minPer (float): Threshold ratio of the maximum column value; columns
                        below this threshold are treated as noise.
        display (bool | None): If True, returns a debug histogram image.
                               If None, uses global DISPLAY flag.
        region (int):
            - 1 : use full height (prediction, more stable).
            - >1: use only the bottom 1/region part of the image
                  (current position, more precise).

    Returns:
        basePoint (int): Estimated lane center x-coordinate (in pixels).
        imgHist (np.ndarray | None): Debug image with plotted histogram,
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

    # Column-wise histogram (sum of pixel values in each column)
    hisValues = np.sum(roi, axis=0)
    midpoint = w // 2
    left_half = hisValues[:midpoint]
    right_half = hisValues[midpoint:]

    # Fallback to image center if one side has no lane pixels
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
        # Current position mode (region != 1): use strongest peaks only
        left_mean = np.argmax(left_half)
        right_mean = np.argmax(right_half) + midpoint

    # Lane center: average between left and right lane positions
    basePoint = int((int(round(left_mean)) + int(round(right_mean))) // 2)

    imgHist = None
    if display:
        imgHist = np.zeros((h, w, 3), np.uint8)
        maxv = hisValues.max() if hisValues.max() > 0 else 1
        for x, intensity in enumerate(hisValues):
            h_line = int((intensity / maxv) * h)
            cv2.line(imgHist, (x, h), (x, h - h_line), (255, 0, 255), 1)

        # Visual markers: left peak (green), right peak (blue), center (yellow)
        lx = int(round(left_mean))
        rx = int(round(right_mean))
        cv2.circle(imgHist, (lx, h), 6, (0, 255, 0), cv2.FILLED)
        cv2.circle(imgHist, (rx, h), 6, (255, 0, 0), cv2.FILLED)
        cv2.circle(imgHist, (basePoint, h), 8, (0, 255, 255), cv2.FILLED)

    return basePoint, imgHist


# ====================== TRACKBARS ======================
def initializeTrackbars(initialTracbarVals, wT=480, hT=240):
    """
    Initialize OpenCV window and trackbars for perspective transform trapezoid.

    Args:
        initialTracbarVals (list[int]): Initial values for
            [widthTop, widthBottom, heightTop, heightBottom].
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
    Read trapezoid coordinates from trackbars and return them as points.

    Args:
        wT (int): Target width.
        hT (int): Target height.

    Returns:
        points (np.ndarray): 4 points [TL, TR, BL, BR] as float32, defining
                             the IPM trapezoid in the original image.
    """
    widthTop = cv2.getTrackbarPos("Width Top", "Points_Setup")
    widthBottom = cv2.getTrackbarPos("Width Bottom", "Points_Setup")
    heightTop = cv2.getTrackbarPos("Height Top", "Points_Setup")
    heightBottom = cv2.getTrackbarPos("Height Bottom", "Points_Setup")

    # Symmetric trapezoid around the image center
    points = np.float32([
        [wT // 2 - widthTop, heightTop],        # Top-Left
        [wT // 2 + widthTop, heightTop],        # Top-Right
        [wT // 2 - widthBottom, heightBottom],  # Bottom-Left
        [wT // 2 + widthBottom, heightBottom]   # Bottom-Right
    ])
    return points


def drawPoints(img, points):
    """
    Draw trapezoid points and its outline on the image.

    Args:
        img (np.ndarray): BGR image.
        points (np.ndarray): 4 points [TL, TR, BL, BR].

    Returns:
        img (np.ndarray): Image with drawn trapezoid.
    """
    reorder = np.array([0, 1, 3, 2])  # TL, TR, BR, BL (proper polygon order)
    pts = np.int32(points[reorder])
    for x, y in pts:
        cv2.circle(img, (int(x), int(y)), 15, (0, 0, 255), cv2.FILLED)
    cv2.polylines(img, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
    return img


# ====================== PERSPECTIVE TRANSFORM ======================
def warpIMG(img, points, w, h):
    """
    Apply perspective (bird's-eye) warp to the image based on 4 points.

    Args:
        img (np.ndarray): Source image.
        points (np.ndarray): 4 points for perspective transform.
        w (int): Target width.
        h (int): Target height.

    Returns:
        imgWarp (np.ndarray): Warped (IPM) image.
    """
    pts1 = np.float32(points)
    pts2 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgWarp = cv2.warpPerspective(img, matrix, (w, h))
    return imgWarp


def thresholding(img):
    """
    Apply HSV-based thresholding and simple morphology to extract lane markings.

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

    # Optional skeletonization/thinning if available in this OpenCV build
    if hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "thinning"):
        maskWhite = cv2.ximgproc.thinning(maskWhite)

    return maskWhite


def normalize_range_points(point, y, w, h):
    """
    Normalize image coordinates to a compact range.

    Args:
        point (float): x coordinate in pixels.
        y (float): y coordinate in pixels.
        w (int): image width in pixels.
        h (int): image height in pixels.

    Returns:
        norm_point (float): normalized x in range [-1, 1]
                            (-1 = far left, 0 = center, +1 = far right).
        norm_y (float): normalized y in range [0, 1]
                        (0 = top of IPM, 1 = bottom of IPM).
    """
    norm_point = (point - (w / 2)) / (w / 2)
    norm_y = y / h
    return norm_point, norm_y


def polynomial_curve_fit(road_points):
    """
    Fit a 2nd-degree polynomial x(y) to lane center points.

    Args:
        road_points (list[tuple]): list of (y_norm, x_norm) points, where:
            y_norm in [0, 1], x_norm in [-1, 1].

    Returns:
        x_fit (np.ndarray): fitted x values (normalized) for sampled y.
        y_fit (np.ndarray): sampled y values in [0, 1].
        a (float): curvature term    (how strong the bend is).
        b (float): slope term        (initial turning direction).
        c (float): offset term       (horizontal shift of the lane).
    """
    # Convert list of points to numpy array: shape (N, 2) -> [y, x]
    road_points_np = np.array(road_points, dtype=np.float32)

    # Separate X and Y (x = f(y))
    xs = road_points_np[:, 1]
    ys = road_points_np[:, 0]

    # Fit 2nd-degree polynomial x = a*y^2 + b*y + c
    coeffs = np.polyfit(ys, xs, 2)

    # a - curvature strength (how sharp the turn is)
    # b - turning direction near the start
    # c - horizontal offset from the image center
    a, b, c = coeffs

    # Debug print of polynomial parameters
    print(f"Polynomial coefficients: a={a:.4f}, b={b:.4f}, c={c:.4f}")

    # Sample the fitted polynomial in normalized y space [0, 1]
    y_fit = np.linspace(0, 1, num=100)
    x_fit = np.polyval(coeffs, y_fit)

    return x_fit, y_fit, a, b, c


# ====================== MAIN LANE PIPELINE ======================
def getLaneCurve(img):
    """
    High-level lane detection pipeline on a static road image.

    Steps:
      1. Read IPM trapezoid from trackbars.
      2. Threshold and warp the image to bird's-eye view.
      3. Split the IPM into 3 horizontal regions (top/mid/bottom).
      4. Use histograms to estimate lane centers in each region.
      5. Normalize lane centers and build a set of (y,x) road points.
      6. Fit a 2nd-degree polynomial x(y) through those points.
      7. Compute a smoothed 'curve' value and visualize everything.
    """
    road_points = []
    h, w, c = img.shape

    # 1. Trapezoid from trackbars (IPM source region)
    points = getTrackbarValues(w, h)

    # 2. Thresholding + warp (on mask)
    imgThresh = thresholding(img)
    imgWarp = warpIMG(imgThresh, points, w, h)

    # 3. Warp original BGR for visualization in IPM coordinates
    imgWarpColor = warpIMG(img, points, w, h)
    imgWarpPoints = imgWarpColor.copy()

    # 4. Split IPM into 3 horizontal regions of interest
    rois, centers_y = roi_segment(imgWarp, vertices=3)
    if len(rois) != 3:
        print("ROI segmentation failed")
        return

    roi_top, roi_middle, roi_bottom = rois
    y_top, y_middle, y_bottom = centers_y

    # 5. Histograms:
    #    - full warped: current bottom lane center (region=4) + global prediction (region=1)
    middlePoint_full_roi, img_midHist = getHistogram(imgWarp, minPer=0.5, display=DISPLAY, region=4)
    curveAveragePoint_full_roi, img_CurveHist = getHistogram(imgWarp, minPer=0.5, display=DISPLAY, region=1)

    #    - separate centers for top/middle/bottom ROIs (no debug images)
    curveAveragePoint_top_roi, _ = getHistogram(roi_top, minPer=0.5, display=False, region=1)
    curveAveragePoint_middle_roi, _ = getHistogram(roi_middle, minPer=0.5, display=False, region=1)
    curveAveragePoint_bottom_roi, _ = getHistogram(roi_bottom, minPer=0.5, display=False, region=1)

    # Normalized versions (x in [-1,1], y in [0,1])
    norm_middlePoint_full_roi, norm_y_middlePoint_full_roi = normalize_range_points(middlePoint_full_roi, h, w, h)
    norm_curveAveragePoint_full_roi, norm_y_curveAveragePoint_full_roi = normalize_range_points(curveAveragePoint_full_roi, h / 2, w, h)
    norm_curveAveragePoint_top_roi, norm_y_curveAveragePoint_top_roi = normalize_range_points(curveAveragePoint_top_roi, y_top, w, h)
    norm_curveAveragePoint_middle_roi, norm_y_curveAveragePoint_middle_roi = normalize_range_points(curveAveragePoint_middle_roi, y_middle, w, h)
    norm_curveAveragePoint_bottom_roi, norm_y_curveAveragePoint_bottom_roi = normalize_range_points(curveAveragePoint_bottom_roi, y_bottom, w, h)

    # Collect normalized road points as (y_norm, x_norm)
    road_points.append((norm_y_middlePoint_full_roi, norm_middlePoint_full_roi))
    road_points.append((norm_y_curveAveragePoint_bottom_roi, norm_curveAveragePoint_bottom_roi))
    road_points.append((norm_y_curveAveragePoint_middle_roi, norm_curveAveragePoint_middle_roi))
    road_points.append((norm_y_curveAveragePoint_top_roi, norm_curveAveragePoint_top_roi))

    # 6. Compute curve: difference between current bottom center and global prediction
    curveRaw = norm_middlePoint_full_roi - norm_curveAveragePoint_full_roi

    curveList.append(curveRaw)
    if len(curveList) > CURVELIST_LENGTH:
        curveList.pop(0)
    curve = float(sum(curveList) / len(curveList))

    print("curve:", curve)
    print("center (bottom):", norm_middlePoint_full_roi)
    print("predicted center:", norm_curveAveragePoint_full_roi)

    # 7. Fit polynomial through road points
    x_fit, y_fit, a, b, c = polynomial_curve_fit(road_points)

    if DISPLAY:
        # Draw center markers on IPM view
        cv2.circle(imgWarpPoints, (middlePoint_full_roi, h), 7, (255, 0, 255), cv2.FILLED)
        cv2.circle(imgWarpPoints, (curveAveragePoint_full_roi, (h // 2) - 20), 7, (255, 255, 255), cv2.FILLED)

        # Top / middle / bottom ROI centers
        cv2.circle(imgWarpPoints, (int(curveAveragePoint_top_roi), int(y_top)), 6, (0, 0, 255), cv2.FILLED)
        cv2.circle(imgWarpPoints, (int(curveAveragePoint_middle_roi), int(y_middle)), 6, (0, 255, 0), cv2.FILLED)
        cv2.circle(imgWarpPoints, (int(curveAveragePoint_bottom_roi), int(y_bottom)), 6, (255, 0, 0), cv2.FILLED)

        # Draw fitted polynomial centerline (back to pixel space)
        x_fit_px = (x_fit * (w / 2)) + (w / 2)
        y_fit_px = y_fit * h
        for x_px, y_px in zip(x_fit_px.astype(int), y_fit_px.astype(int)):
            cv2.circle(imgWarpPoints, (x_px, int(y_px)), 2, (0, 255, 255), -1)

        # Debug views
        imgMainView = drawPoints(img.copy(), points)
        cv2.circle(imgMainView, (middlePoint_full_roi, img.shape[0] - 10), 10, (0, 255, 255), cv2.FILLED)

        cv2.imshow("Main View", imgMainView)
        cv2.imshow("Warped", imgWarp)
        cv2.imshow("Warped Points", imgWarpPoints)
        cv2.imshow("Mid Histogram", img_midHist)
        cv2.imshow("Curve Histogram", img_CurveHist)

    return a, b, c


# ====================== ENTRY POINT ======================
img = cv2.imread('road_images/crossroad.png')
img = cv2.resize(img, (480, 240))

if DEBUG:
    initialTrackbarValues = [140, 240, 116, 240]  # initial trapezoid parameters
    initializeTrackbars(initialTrackbarValues, wT=480, hT=240)

while True:
    getLaneCurve(img)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
