import cv2
import numpy as np
import math
import os

# =========================================
# --- KONFIGURACJA --- To jest mój kod do pracy inż
# =========================================

BASE_PATH = 'baza_do_porownania'
N_FEATURES = 3000
MATCH_RATIO = 0.8
MIN_SCORE = 30.0
DEBUG = True

# =========================================
# --- FUNKCJE ORB ---
# =========================================

def load_orb_database(folder_path):
    """Wczytuje wszystkie obrazy z podanego folderu i generuje deskryptory ORB."""
    orb = cv2.ORB_create(nfeatures=N_FEATURES)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    des_list, class_names, img_list = [], [], []

    if not os.path.exists(folder_path):
        if DEBUG:
            print(f"[WARN] Brak folderu: {folder_path}")
        return orb, bf, des_list, class_names, img_list

    for file in os.listdir(folder_path):
        if not file.lower().endswith(('.jpg', '.png', '.jpeg')):
            continue
        img = cv2.imread(os.path.join(folder_path, file), 0)
        kp, des = orb.detectAndCompute(img, None)
        if des is not None:
            des_list.append(des)
            class_names.append(os.path.splitext(file)[0])
            img_list.append(img)

    if DEBUG:
        print(f"[INFO] Wczytano {len(class_names)} wzorców z {folder_path}")
    return orb, bf, des_list, class_names, img_list


def orb_match(roi, orb, bf, des_list, class_names, img_list):
    """Porównuje ROI z bazą znaków z wybranego folderu."""
    if len(des_list) == 0:
        return None, 0, None

    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    kp2, des2 = orb.detectAndCompute(gray_roi, None)
    if des2 is None:
        return None, 0, None

    best_score = 0
    best_name = None
    best_vis = None

    for i, des in enumerate(des_list):
        matches = bf.match(des, des2)
        if not matches:
            continue
        matches = sorted(matches, key=lambda x: x.distance)
        good_matches = [m for m in matches if m.distance < MATCH_RATIO * 100]

        avg_distance = np.mean([m.distance for m in good_matches]) if good_matches else 100
        score = max(0, 100 - avg_distance)

        if score > best_score:
            best_score = score
            best_name = class_names[i]
            best_vis = cv2.drawMatches(
                img_list[i], orb.detectAndCompute(img_list[i], None)[0],
                gray_roi, kp2, matches[:20], None,
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
            )

    if DEBUG and best_vis is not None:
        cv2.imshow("Porownanie ORB", best_vis)

    return best_name, best_score, gray_roi


# =========================================
# --- FUNKCJE KOLOR + KSZTAŁT ---
# =========================================

def hsv_frame_process(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

def color_name_from_hue(hue):
    if (hue <= 5) or (hue >= 170):
        return "Czerwony"
    if 5 < hue < 30:
        return "Pomaranczowy"
    if 85 <= hue < 125:
        return "Niebieski"
    return "Inny"

def color_checking_area(hsv_frame, contour, x, y, w, h):
    H, W = hsv_frame.shape[:2]
    x0, y0 = max(0, int(x)), max(0, int(y))
    x1, y1 = min(W, int(x + w)), min(H, int(y + h))
    if x0 >= x1 or y0 >= y1:
        return None, 0.0

    roi_hsv = hsv_frame[y0:y1, x0:x1]
    contour_pts = contour.reshape(-1, 2).astype(np.int32)
    shifted = contour_pts - np.array([x0, y0])
    mask = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
    cv2.drawContours(mask, [shifted], -1, 255, thickness=-1)
    hue_channel = roi_hsv[:, :, 0]
    selected_hues = hue_channel[mask == 255]
    if selected_hues.size == 0:
        return None, 0.0
    unique_hues, counts = np.unique(selected_hues, return_counts=True)
    name_counts = {}
    for u, c in zip(unique_hues, counts):
        name = color_name_from_hue(int(u))
        name_counts[name] = name_counts.get(name, 0) + int(c)
    dominant_name = max(name_counts, key=name_counts.get)
    fraction = name_counts[dominant_name] / selected_hues.size
    return dominant_name, fraction


# =========================================
# --- KLASYFIKATOR ZNAKÓW ---
# =========================================

def trafficsign_classifier(shape_name, dominant_color, roi):
    """Sprawdza kształt i kolor, wybiera odpowiednią bazę i zwraca dopasowanie ORB."""
    
    # Dobór folderu na podstawie kształtu i koloru
    if shape_name == "Osmiokat" and dominant_color == "Czerwony":
        group_folder = os.path.join(BASE_PATH, "STOP")
    elif shape_name == "Trojkat" and dominant_color == "Pomaranczowy":
        group_folder = os.path.join(BASE_PATH, "OSTRZEGAWCZE")
    elif shape_name == "Kolo" and dominant_color != "Niebieski":
        group_folder = os.path.join(BASE_PATH, "OGRANICZENIA")
    elif shape_name == "Kolo" and dominant_color == "Niebieski":
        print(f"[DEBUG] Shape: {shape_name}, Dominant color: {dominant_color}")
        group_folder = os.path.join(BASE_PATH, "NAKAZ")
    else:
        if DEBUG:
            print("[INFO] Niezidentyfikowany znak.")
        return "Niezidentyfikowany", 0

    if DEBUG:
        print(f"[INFO] Klasyfikacja: {shape_name}, kolor: {dominant_color}, folder: {group_folder}")

    orb, bf, des_list, class_names, img_list = load_orb_database(group_folder)
    name, score, _ = orb_match(roi, orb, bf, des_list, class_names, img_list)
    return name, score


# =========================================
# --- DETEKCJA KSZTAŁTÓW ---
# =========================================

def detect_shapes(frame):
    clahe = cv2.createCLAHE(clipLimit=3, tileGridSize=(8, 8))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    equalized = clahe.apply(gray)
    blur = cv2.GaussianBlur(equalized, (5, 5), 0)
    edges = cv2.Canny(blur, 80, 170)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    output = frame.copy()
    hsv_frame = hsv_frame_process(frame)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 600:
            continue
            
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue

        print(f"Area: {area}")
        approx = cv2.approxPolyDP(cnt, 0.012 * perimeter, True)
        vertices = len(approx)
        circularity = 4 * math.pi * area / (perimeter * perimeter)
        x, y, w, h = cv2.boundingRect(approx)

        shape_name = None
        color_draw = (0, 255, 0)

        if vertices <= 6 and circularity < 0.7:
            shape_name = "Trojkat"
            color_draw = (0, 255, 255)
        elif vertices > 8 and circularity > 0.85:
            aspect_ratio = float(w) / h if h != 0 else float('inf')
            if 0.7 < aspect_ratio < 1.5:
                shape_name = "Kolo"
                color_draw = (255, 0, 0)
        elif vertices == 8:
            aspect_ratio = float(w) / h if h != 0 else float('inf')
            if 0.9 < aspect_ratio < 1.1:
                shape_name = "Osmiokat"
                color_draw = (255, 0, 255)

        if shape_name:
            dominant_color, fraction = color_checking_area(hsv_frame, approx, x, y, w, h)
            if not dominant_color:
                continue

            roi = frame[y:y+h, x:x+w]
            if roi.size == 0:
                continue

            if DEBUG:
                cv2.imshow("ROI", roi)

            name, score = trafficsign_classifier(shape_name, dominant_color, roi)
            if name and score >= MIN_SCORE:
                label = f"{name} ({score:.0f}%)"
                cv2.drawContours(output, [approx], -1, color_draw, 2)
                cv2.putText(output, label, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_draw, 2)

    return output, edges, blur


# =========================================
# --- GŁÓWNA PĘTLA ---
# =========================================

def main():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        output, edges, blur = detect_shapes(frame)
        cv2.imshow("Krawedzie", edges)
        cv2.imshow("Rozmycie", blur)
        cv2.imshow("Wykrywanie znakow", output)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
