# ====================== Główny kod do wykrwania środka pasa ======================
import cv2
import numpy as np
# ====================== PROGOWANIE ======================
DEBUG = True
DISPLAY = True
h_min = 0
s_min = 0
v_min = 150
h_max = 179
s_max = 255
v_max = 255
curveList = []
CURVELIST_LENGTH = 10
def nothing(a):
    pass

def getHistogram(img, minPer=0.1, display=None, region=1):
    """
    Compute lane center position from a column-wise histogram

    Args:
        img (np.ndarray): binary or grayscale warped image
        minPer (float): threshold (fraction of max) used to ignore weak columns
        display (bool|None): if True, returns a debug histogram image
                             if None, uses global DISPLAY flag
        region (int): if 1 -> use full height (prediction)
                      if >1 -> use bottom part of the image (current position)

    Returns:
        basePoint (int): estimated lane center x-coordinate
        imgHist (np.ndarray|None): optional debug image with drawn histogram
    """
    # Jeśli nie przekazano jawnie display, użyj globalnej wartości DISPLAY
    if display is None:
        display = DISPLAY

    h, w = img.shape[:2]

    # Wybierz ROI w zależności od regionu:
    if region == 1:
        roi = img
    else:
        start_row = h - h // region
        roi = img[start_row:, :]

    # Histogram: suma wartości pikseli po kolumnach
    hisValues = np.sum(roi, axis=0)
    midpoint = w // 2
    left_half = hisValues[:midpoint]
    right_half = hisValues[midpoint:]

    # Jeśli brak danych po którejkolwiek stronie -> fallback na środek obrazu
    if left_half.max() == 0 or right_half.max() == 0:
        basePoint = midpoint
        imgHist = np.zeros((h, w, 3), np.uint8) if display else None
        return basePoint, imgHist

    # Dwa tryby: prognoza (region==1) -> uśrednianie (centroid ważony intensywnością)
    # i aktualny (region!=1) -> argmax (maksimum pików)
    if region == 1:
        # prognoza: wybieramy indeksy przekraczające threshold i liczymy centroid (ważony)
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
        # aktualny środek: bierzemy najsilniejsze piki (dokładność lokalna)
        left_mean = np.argmax(left_half)
        right_mean = np.argmax(right_half) + midpoint

    # wynikowy punkt (środek między wykrytymi krawędziami)
    basePoint = int((int(round(left_mean)) + int(round(right_mean))) // 2)

    # Rysowanie histogramu (opcjonalnie)
    imgHist = None
    if display:
        imgHist = np.zeros((h, w, 3), np.uint8)
        maxv = hisValues.max() if hisValues.max() > 0 else 1
        for x, intensity in enumerate(hisValues):
            h_line = int((intensity / maxv) * h)
            cv2.line(imgHist, (x, h), (x, h - h_line), (255, 0, 255), 1)

        # zaznacz lewy/prawy wykryty punkt i basePoint
        lx = int(round(left_mean))
        rx = int(round(right_mean))
        cv2.circle(imgHist, (lx, h), 6, (0, 255, 0), cv2.FILLED)      # lewy (zielony)
        cv2.circle(imgHist, (rx, h), 6, (255, 0, 0), cv2.FILLED)      # prawy (niebieski)
        cv2.circle(imgHist, (basePoint, h), 8, (0, 255, 255), cv2.FILLED)  # center (żółty)

    return basePoint, imgHist





# ====================== TRACKBARY ======================
def initializeTrackbars(initialTracbarVals, wT=480, hT=240):
    """Inicjalizuje okno i suwaki do ustawiania trapezu do transformacji perspektywy"""
    cv2.namedWindow("Points_Setup")
    cv2.resizeWindow("Points_Setup", 600, 300)
    cv2.createTrackbar("Width Top", "Points_Setup", initialTracbarVals[0], wT//2, nothing)
    cv2.createTrackbar("Width Bottom", "Points_Setup", initialTracbarVals[1], wT//2, nothing)
    cv2.createTrackbar("Height Top", "Points_Setup", initialTracbarVals[2], hT, nothing)
    cv2.createTrackbar("Height Bottom", "Points_Setup", initialTracbarVals[3], hT, nothing)


def getTrackbarValues(wT=480, hT=240):
    """Zwraca cztery punkty (TL, TR, BL, BR) w formacie wymaganym przez OpenCV."""
    widthTop = cv2.getTrackbarPos("Width Top", "Points_Setup")
    widthBottom = cv2.getTrackbarPos("Width Bottom", "Points_Setup")
    heightTop = cv2.getTrackbarPos("Height Top", "Points_Setup")
    heightBottom = cv2.getTrackbarPos("Height Bottom", "Points_Setup")

    # Tworzymy trapez symetryczny względem środka obrazu
    points = np.float32([
        [wT//2 - widthTop, heightTop],       # Top-Left
        [wT//2 + widthTop, heightTop],       # Top-Right
        [wT//2 - widthBottom, heightBottom], # Bottom-Left
        [wT//2 + widthBottom, heightBottom]  # Bottom-Right
    ])
    return points

def drawPoints(img, points):
    """Rysuje punkty i trapez poprawnie, bez krzyżowania linii"""
    reorder = np.array([0, 1, 3, 2])  # TL, TR, BR, BL zamiast TL, TR, BL, BR
    pts = np.int32(points[reorder])
    for x, y in pts:
        cv2.circle(img, (int(x), int(y)), 15, (0, 0, 255), cv2.FILLED)
    cv2.polylines(img, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
    return img

# ====================== TRANSFORMACJA ======================
def warpIMG(img, points, w, h):
    """Zwraca obraz po perspektywicznym przekształceniu"""
    pts1 = np.float32(points)
    pts2 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgWarp = cv2.warpPerspective(img, matrix, (w, h))
    return imgWarp




def thresholding(img):
    """
    Apply HSV-based thresholding to extract bright lane markings

    Args:
        img (np.ndarray): input BGR frame or road image

    Returns:
        maskWhite (np.ndarray): binary mask of lane candidates
        (optionally thinned if ximgproc.thinning is available)
    """
    imgHsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lowerWhite = np.array([h_min, s_min, v_min])
    upperWhite = np.array([h_max, s_max, v_max])
    maskWhite = cv2.inRange(imgHsv, lowerWhite, upperWhite)
    maskWhite = cv2.morphologyEx(maskWhite, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))
    maskWhite = cv2.GaussianBlur(maskWhite, (5,5), 0)
    if hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "thinning"):
        maskWhite = cv2.ximgproc.thinning(maskWhite)

    return maskWhite


# ====================== GŁÓWNA FUNKCJA ======================
def getLaneCurve(img):
    """
    Estimate lane center offset on a static road image

    Processing steps:
        - read trapezoid ROI from trackbars
        - threshold image in HSV space
        - apply perspective warp
        - compute histogram-based lane center and average curve

    Args:
        img (np.ndarray): BGR image of the road

    Returns:
        None  # draws multiple debug windows and prints the curve value to console
    """
    h, w, c = img.shape
    points = getTrackbarValues(w, h)
    imgThresh = thresholding(img)
    imgWarp = warpIMG(imgThresh, points, w, h)

    imgWarpPoints = drawPoints(img.copy(), points)

    middlePoint, img_midHist = getHistogram(imgWarp,minPer=0.5, display=DISPLAY, region=4)
    curveAveragePoint, img_CurveHist = getHistogram(imgWarp,minPer=0.5, display=DISPLAY)
    curveRaw = curveAveragePoint-middlePoint
    
    curveList.append(curveRaw)
    if len(curveList) > CURVELIST_LENGTH:
        curveList.pop(0)
    curve = int(sum(curveList)/len(curveList))


    print(curve)
    cv2.circle(imgWarpPoints, (curveAveragePoint, img.shape[0]), 15, (0,255,255), cv2.FILLED)
    cv2.imshow('Warped', imgWarp)
    cv2.imshow('Warped Points', imgWarpPoints)
    cv2.imshow('Mid Histogram', img_midHist)
    cv2.imshow('Curve Histogram', img_CurveHist)
    return None


# ====================== MAIN ======================
img = cv2.imread('Linia_drogi/droga2.png')
img = cv2.resize(img, (480, 240))

if DEBUG:
    initializeTrackbarsValues = [140, 240, 116, 240]  # początkowe wartości suwaków
    initializeTrackbars(initializeTrackbarsValues, wT=480, hT=240)

while True:
    getLaneCurve(img)


    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
