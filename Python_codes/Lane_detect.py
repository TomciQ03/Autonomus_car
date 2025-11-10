# ====================== Main lane center detection code ======================
import cv2
import numpy as np
import time

# ====================== GLOBAL CONFIG ======================
DEBUG = True
DISPLAY = True
# ====================== MOST VALUABLE DATA ======================
# "curve": curve,
# "offset": norm_middlePoint_full_roi,
# "poly": (a, b, c),
# "lane_ok": lane_ok,
# "left_lane_ok": left_lane_ok,
# "right_lane_ok": right_lane_ok,

#======================= PARAMS ======================
# HSV threshold parameters for lane marking detection
h_min = 0
s_min = 0
v_min = 150
h_max = 179
s_max = 255
v_max = 255
# progi do wykrywania skrzyżowań (będą sterowane z trackbarów)
k_center_energy = 0.06   # ile energii w środku = "wypełniony środek"
k_side_energy   = 0.04   # minimalna energia boku aby uznać że pas istnieje
k_peak_sim      = 0.5    # podobieństwo pików (0..1)
spread_branch   = 0.30   # spread > 0.25 => szeroka odnoga
spread_lane     = 0.30   # spread < 0.10 => typowy wąski pas

left_lane_ok = False
right_lane_ok = False
lanes_detected = False
crossroad_type = "not_detected"  # "not_detected", "T_crossroad", "X_crossroad" ,"L_turn-straight", "R_turn-straight"
threshold_lane_detect = 0.05
confidence_left = 0.0
confidence_right = 0.0
curveList = []
CURVELIST_LENGTH = 10

def initializeCrossroadTrackbars():
    """
    Window + trackbars for tuning crossroad detection thresholds:
      - k_center_energy
      - k_side_energy
      - k_peak_sim
    """
    cv2.namedWindow("Crossroad_Debug")
    cv2.resizeWindow("Crossroad_Debug", 500, 300)

    # trackbary 0..100 -> 0.00 .. 1.00
    cv2.createTrackbar("k_center x100", "Crossroad_Debug", int(k_center_energy * 100), 100, nothing)
    cv2.createTrackbar("k_side   x100", "Crossroad_Debug", int(k_side_energy * 100),   100, nothing)
    cv2.createTrackbar("k_peak   x100", "Crossroad_Debug", int(k_peak_sim * 100),      100, nothing)
    cv2.createTrackbar("spread branch x100", "Crossroad_Debug", int(spread_branch * 100), 100, nothing)
    cv2.createTrackbar("spread lane   x100", "Crossroad_Debug", int(spread_lane * 100),   100, nothing)


def time_to_exec(func):
    """Decorator to measure execution time of functions."""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        exec_time = end_time - start_time
        print(f"Execution time of {func.__name__}: {exec_time:.6f} seconds")
        return result
    return wrapper

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


def crossroad_in_roi(hisValues, left_mean, right_mean, h_roi, w_roi):
    """
    Analiza histogramu w dolnym ROI i próba wykrycia typu skrzyżowania.

    Ustawia globalny crossroad_type na jeden z:
        - "X_crossroad"
        - "T_crossroad"
        - "L_turn-straight"
        - "R_turn-straight"
        - "not_detected"

    Dodatkowo rysuje okno "Crossroad_Debug" z bieżącymi parametrami.
    """
    global crossroad_type, k_center_energy, k_side_energy, k_peak_sim, spread_branch, spread_lane

    # --- odczyt progów z trackbarów (0..100 -> 0.00..1.00) ---
    k_center_energy = cv2.getTrackbarPos("k_center x100", "Crossroad_Debug") / 100.0
    k_side_energy   = cv2.getTrackbarPos("k_side   x100", "Crossroad_Debug") / 100.0
    k_peak_sim      = cv2.getTrackbarPos("k_peak   x100", "Crossroad_Debug") / 100.0
    spread_branch   = cv2.getTrackbarPos("spread branch x100", "Crossroad_Debug") / 100.0
    spread_lane     = cv2.getTrackbarPos("spread lane   x100", "Crossroad_Debug") / 100.0

    # --- bezpieczeństwo indeksów ---
    w_hist = len(hisValues)
    if w_hist == 0:
        crossroad_type = "not_detected"
        return None

    left_i  = int(round(left_mean))
    right_i = int(round(right_mean))

    left_i  = max(1, min(left_i,  w_hist - 2))
    right_i = max(left_i + 1, min(right_i, w_hist - 1))

    # --- podział na 3 sekcje ---
    left_section   = hisValues[:left_i]
    center_section = hisValues[left_i:right_i]
    right_section  = hisValues[right_i:]

    left_width   = max(len(left_section),   1)
    center_width = max(len(center_section), 1)
    right_width  = max(len(right_section),  1)

    def section_energy(section, width):
        energy = float(np.sum(section))
        max_energy = float(h_roi * width * 255)
        norm = energy / max_energy if max_energy > 0 else 0.0
        return energy, norm

    def section_spread(section):
        """0..1 – jak szeroka jest część 'wysoka' w danej sekcji."""
        if section.size == 0:
            return 0.0
        peak = float(section.max())
        if peak <= 0:
            return 0.0
        thr = 0.3 * peak
        width_high = float(np.sum(section >= thr))
        return width_high / float(len(section))

    # --- energie (wypełnienie) ---
    _, norm_left   = section_energy(left_section,   left_width)
    _, norm_center = section_energy(center_section, center_width)
    _, norm_right  = section_energy(right_section,  right_width)

    # --- spread (kształt) ---
    spread_left   = section_spread(left_section)
    spread_center = section_spread(center_section)
    spread_right  = section_spread(right_section)

    # --- piki i podobieństwo ---
    left_peak  = float(hisValues[left_i])
    right_peak = float(hisValues[right_i])
    peak_diff  = abs(left_peak - right_peak)
    max_peak   = max(left_peak, right_peak, 1.0)
    peaks_similar = (peak_diff <= k_peak_sim * max_peak)

    # --- proste flagi ---
    left_section_filled   = norm_left   > k_side_energy
    right_section_filled  = norm_right  > k_side_energy
    center_section_filled = norm_center > k_center_energy

    # Czy zachowują się jak wąski pas czy jak szeroka odnoga?
    left_is_branch  = spread_left  > spread_branch
    right_is_branch = spread_right > spread_branch

    left_is_lane  = spread_left  < spread_lane
    right_is_lane = spread_right < spread_lane

    # ----------------- KLASYFIKACJA WG TWOICH ZASAD -----------------

    # 1) X_crossroad: środek pusty, piki podobne, oba boki jak pasy
    if (not center_section_filled and
        peaks_similar and
        left_is_lane and right_is_lane):
        crossroad_type = "X_crossroad"
    # 2) T_crossroad: środek wypełniony, piki podobne, szerokie odnogi po obu stronach
    elif (center_section_filled and
          peaks_similar and
          left_is_branch and right_is_branch):
        crossroad_type = "T_crossroad"
    # 3) L_turn-straight: środek pusty, piki różne, szeroka odnoga tylko po lewej
    elif (not center_section_filled and
          not peaks_similar and
          left_section_filled and left_is_branch and
          right_section_filled and right_is_lane):
        crossroad_type = "L_turn-straight"
    # 4) R_turn-straight: środek pusty, piki różne, szeroka odnoga tylko po prawej
    elif (not center_section_filled and
          not peaks_similar and
          right_section_filled and right_is_branch and
          left_section_filled and left_is_lane):
        crossroad_type = "R_turn-straight"
    else:
        crossroad_type = "not_detected"

    # ----------------- OKNO DEBUG -----------------
    debug_img = np.zeros((260, 500, 3), np.uint8)

    y = 20
    dy = 20

    cv2.putText(debug_img, f"k_center={k_center_energy:.2f}  k_side={k_side_energy:.2f}  k_peak={k_peak_sim:.2f}",
                (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 255), 1)
    y += dy

    cv2.putText(debug_img, f"Energies: L={norm_left:.3f}  C={norm_center:.3f}  R={norm_right:.3f}",
                (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    y += dy

    cv2.putText(debug_img, f"Spreads : L={spread_left:.3f}  C={spread_center:.3f}  R={spread_right:.3f}",
                (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    y += dy

    cv2.putText(debug_img, f"Peaks   : L={left_peak:.0f}  R={right_peak:.0f}  diff={peak_diff:.0f}  similar={peaks_similar}",
                (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)
    y += dy

    cv2.putText(debug_img, f"Filled  : L={left_section_filled}  C={center_section_filled}  R={right_section_filled}",
                (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 255, 200), 1)
    y += dy

    cv2.putText(debug_img, f"Lane    : L={left_is_lane}  R={right_is_lane}",
                (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 255), 1)
    y += dy

    cv2.putText(debug_img, f"Branch  : L={left_is_branch}  R={right_is_branch}",
                (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 150, 150), 1)
    y += dy

    cv2.putText(debug_img, f"TYPE: {crossroad_type}",
                (10, y+10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    cv2.imshow("Crossroad_Debug", debug_img)

    return None





def energy_in_roi(left_half, right_half, h_roi, midpoint, w_roi):
    global left_lane_ok, right_lane_ok, lanes_detected, threshold_lane_detect, confidence_left, confidence_right
    """ Calculate normalized energy in left and right halves of the ROI.
    Args:
        left_half (np.ndarray): Pixel values of the left half.
        right_half (np.ndarray): Pixel values of the right half.
        h_roi (int): Height of the ROI.
        midpoint (int): Midpoint column index.
        w_roi (int): Width of the ROI.

    Returns:
        tuple: (normalized_left_energy, normalized_right_energy)
    """
    left_energy_sum  = np.sum(left_half)
    right_energy_sum = np.sum(right_half)

    # Max theoretical energy per half (all pixels = 255)
    max_left_energy  = h_roi * midpoint * 255
    max_right_energy = h_roi * (w_roi - midpoint) * 255

    normalized_left_energy  = left_energy_sum  / max_left_energy  if max_left_energy  > 0 else 0.0
    normalized_right_energy = right_energy_sum / max_right_energy if max_right_energy > 0 else 0.0

    confidence_left = min(normalized_left_energy / threshold_lane_detect, 1.0)
    confidence_right = min(normalized_right_energy / threshold_lane_detect, 1.0)


    left_lane_ok  = normalized_left_energy  > threshold_lane_detect
    right_lane_ok = normalized_right_energy > threshold_lane_detect
    lanes_detected = left_lane_ok and right_lane_ok

    return normalized_left_energy, normalized_right_energy


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
    h_roi, w_roi = roi.shape[:2]
    # Column-wise histogram (sum of pixel values in each column)
    hisValues = np.sum(roi, axis=0)
    midpoint = w_roi // 2
    left_half = hisValues[:midpoint]
    right_half = hisValues[midpoint:]

    if region == 4:
        # For region=4, set global line detect flags based on energy
        energy_left, energy_right = energy_in_roi(left_half, right_half, h_roi, midpoint, w_roi)
    else:
        energy_left, energy_right = 0.0, 0.0
    # Fallback to image center if one side has no lane pixels
    if left_half.max() == 0 or right_half.max() == 0:
        basePoint = midpoint
        imgHist = np.zeros((h, w, 3), np.uint8) if display else None
        return basePoint, imgHist, energy_left, energy_right

    

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
        
    if region != 4:
        crossroad_in_roi(hisValues, left_mean, right_mean, h_roi, w_roi)
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

    return basePoint, imgHist, energy_left, energy_right


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
    middlePoint_full_roi, img_midHist, energy_left, energy_right = getHistogram(imgWarp, minPer=0.5, display=DISPLAY, region=4)
    curveAveragePoint_full_roi, img_CurveHist, _ , _= getHistogram(imgWarp, minPer=0.5, display=DISPLAY, region=1)

    #    - separate centers for top/middle/bottom ROIs (no debug images)
    curveAveragePoint_top_roi, _, _, _ = getHistogram(roi_top, minPer=0.5, display=False, region=1)
    curveAveragePoint_middle_roi, _, _, _ = getHistogram(roi_middle, minPer=0.5, display=False, region=1)
    curveAveragePoint_bottom_roi, _, _, _ = getHistogram(roi_bottom, minPer=0.5, display=False, region=1)

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
        cv2.putText(imgWarpPoints, "crossroad %r" % crossroad_type, (150, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 1) # Blue color text
        # Top / middle / bottom ROI centers
        cv2.circle(imgWarpPoints, (int(curveAveragePoint_top_roi), int(y_top)), 6, (0, 0, 255), cv2.FILLED)
        cv2.circle(imgWarpPoints, (int(curveAveragePoint_middle_roi), int(y_middle)), 6, (0, 255, 0), cv2.FILLED)
        cv2.circle(imgWarpPoints, (int(curveAveragePoint_bottom_roi), int(y_bottom)), 6, (255, 0, 0), cv2.FILLED)

        # Draw fitted polynomial centerline (back to pixel space)
        x_fit_px = (x_fit * (w / 2)) + (w / 2)
        y_fit_px = y_fit * h
        for x_px, y_px in zip(x_fit_px.astype(int), y_fit_px.astype(int)):
            cv2.circle(imgWarpPoints, (x_px, int(y_px)), 2, (0, 255, 255), -1)
        # Write lanes status on imgWarpPoints
        cv2.putText(imgWarp, "left lane %r" % left_lane_ok, (150, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 1) # Blue color text
        cv2.putText(imgWarp, "right lane %r" % right_lane_ok, (150, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 1)
        cv2.putText(imgWarp, "Are lanes detected %r" % lanes_detected, (150, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 1)
        cv2.putText(imgWarp, "left energy %.3f" % energy_left, (150, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 1)
        cv2.putText(imgWarp, "left confidence %.1f%%" % (confidence_left*100), (150, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 1)
        cv2.putText(imgWarp, "right energy %.3f" % energy_right, (150, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 1)
        cv2.putText(imgWarp, "right confidence %.1f%%" % (confidence_right*100), (150, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 1)
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
img = cv2.imread('road_images/crossroad-right.png')
img = cv2.resize(img, (480, 240))

if DEBUG:
    initialTrackbarValues = [140, 240, 116, 240]  # initial trapezoid parameters
    initializeTrackbars(initialTrackbarValues, wT=480, hT=240)
    initializeCrossroadTrackbars()
while True:
    # Clean terminal output
    print("\033c", end="")
    time_to_exec(getLaneCurve)(img)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
