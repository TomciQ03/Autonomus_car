import cv2
import numpy as np
import math
import os


class SignDetector:

    def __init__(self, debug=True, display=False, base_path='sign_database', preload=True):
        self.DEBUG = debug
        self.DISPLAY = display

        # Configuration
        self.BASE_PATH = base_path
        self.N_FEATURES = 3000
        self.MATCH_RATIO = 0.8
        self.MIN_SCORE = 30.0
        self.orb = cv2.ORB_create(nfeatures=self.N_FEATURES)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.clahe = cv2.createCLAHE(clipLimit=3, tileGridSize=(8, 8))

        # cache: folder_path -> {"des_list":..., "class_names":..., "img_list":...}
        self.databases = {}

        # opcjonalne preloadowanie wszystkich grup od razu
        if preload:
            self.preload_databases()

    def preload_databases(self, folders=None, verbose=True):
        """
        Preload all ORB databases from BASE_PATH into memory (self.databases).
        If `folders` is None -> iterate over all subdirectories inside BASE_PATH.
        """
        if not os.path.exists(self.BASE_PATH):
            if self.DEBUG:
                print(f"[WARN] Base path not found: {self.BASE_PATH}")
            return

        # list of folders to preload
        if folders is None:
            folders = []
            for name in os.listdir(self.BASE_PATH):
                full = os.path.join(self.BASE_PATH, name)
                if os.path.isdir(full):
                    folders.append(full)

        if self.DEBUG and verbose:
            print(f"[INFO] Preloading {len(folders)} sign groups from {self.BASE_PATH}...")

        for idx, folder in enumerate(folders, start=1):
            if self.DEBUG and verbose:
                print(f"  [{idx}/{len(folders)}] Loading: {folder}")
            # this call will load & cache
            self.load_orb_database(folder)

        if self.DEBUG and verbose:
            print("[INFO] Preload finished.")


        # --- adjust load_orb_database to return and use cache if present ---
    def load_orb_database(self, folder_path):
        """
        Load all images from a given folder and generate ORB descriptors.

        Uses simple caching in self.databases.
        """
        # 1) return cached if exists
        cache = self.databases.get(folder_path)
        if cache is not None:
            if self.DEBUG:
                print(f"[DEBUG] Using cached DB for: {folder_path}")
            return cache["des_list"], cache["class_names"], cache["img_list"]

        # 2) otherwise load from disk
        orb = self.orb
        des_list, class_names, img_list = [], [], []

        if not os.path.exists(folder_path):
            if self.DEBUG:
                print(f"[WARN] Folder not found: {folder_path}")
            # store empty cache to avoid repeated attempts
            self.databases[folder_path] = {
                "des_list": des_list,
                "class_names": class_names,
                "img_list": img_list,
            }
            return des_list, class_names, img_list

        for file in os.listdir(folder_path):
            if not file.lower().endswith(('.jpg', '.png', '.jpeg')):
                continue
            img = cv2.imread(os.path.join(folder_path, file), 0)
            if img is None:
                continue
            kp, des = orb.detectAndCompute(img, None)
            if des is not None:
                des_list.append(des)
                class_names.append(os.path.splitext(file)[0])
                img_list.append(img)

        if self.DEBUG:
            print(f"[INFO] Loaded {len(class_names)} templates from {folder_path}")

        # 3) save to cache and return
        self.databases[folder_path] = {
            "des_list": des_list,
            "class_names": class_names,
            "img_list": img_list,
        }
        return des_list, class_names, img_list
    
    def orb_match(self, roi, des_list, class_names, img_list):
        """
        Compare a region of interest (ROI) with a database of sign templates using ORB.

        Args:
            roi (np.ndarray): BGR region of interest (candidate sign).
            orb (cv2.ORB): ORB detector instance.
            bf (cv2.BFMatcher): Brute-force matcher.
            des_list (list[np.ndarray]): Descriptors from database templates.
            class_names (list[str]): Names for each template.
            img_list (list[np.ndarray]): Grayscale template images.

        Returns:
            best_name (str | None): Name of best-matched template, or None.
            best_score (float): Matching score (0–100), higher is better.
            gray_roi (np.ndarray): Grayscale ROI used for matching.
        """
        if len(des_list) == 0:
            return None, 0, None

        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        kp2, des2 = self.orb.detectAndCompute(gray_roi, None)
        if des2 is None:
            return None, 0, None

        best_score = 0
        best_name = None
        best_vis = None

        for i, des in enumerate(des_list):
            matches = self.bf.match(des, des2)
            if not matches:
                continue
            matches = sorted(matches, key=lambda x: x.distance)
            good_matches = [m for m in matches if m.distance < self.MATCH_RATIO * 100]

            avg_distance = np.mean([m.distance for m in good_matches]) if good_matches else 100
            score = max(0, 100 - avg_distance)

            if score > best_score:
                best_score = score
                best_name = class_names[i]
                best_vis = cv2.drawMatches(
                    img_list[i], self.orb.detectAndCompute(img_list[i], None)[0],
                    gray_roi, kp2, matches[:20], None,
                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
                )

        if self.DEBUG and best_vis is not None:
            cv2.imshow("ORB Matching", best_vis)

        return best_name, best_score, gray_roi


    # =========================================
    # --- COLOR + SHAPE HELPERS ---
    # =========================================
    @staticmethod
    def hsv_frame_process(frame):
        """Convert BGR frame to HSV color space."""
        return cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    @staticmethod
    def color_name_from_hue(hue):
        """
        Map hue value to a rough color name.

        Args:
            hue (int): Hue channel value [0–179].

        Returns:
            str: One of "Red", "Orange", "Blue", "Other".
        """
        if (hue <= 5) or (hue >= 170):
            return "Red"
        if 5 < hue < 30:
            return "Orange"
        if 85 <= hue < 125:
            return "Blue"
        return "Other"


    def color_checking_area(self, hsv_frame, contour, x, y, w, h):
        """
        Estimate dominant color name inside a given contour in HSV frame.

        Args:
            hsv_frame (np.ndarray): Full HSV frame.
            contour (np.ndarray): Contour points.
            x, y, w, h (int): Bounding rectangle of contour.

        Returns:
            dominant_name (str | None): Dominant color name ("Red", "Orange", "Blue", "Other").
            fraction (float): Fraction of pixels belonging to that dominant color.
        """
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
            name = self.color_name_from_hue(int(u))
            name_counts[name] = name_counts.get(name, 0) + int(c)

        dominant_name = max(name_counts, key=name_counts.get)
        fraction = name_counts[dominant_name] / selected_hues.size
        return dominant_name, fraction


    # =========================================
    # --- TRAFFIC SIGN CLASSIFIER ---
    # =========================================

    def trafficsign_classifier(self, shape_name, dominant_color, roi):
        """
        Choose the correct sign group based on shape and dominant color,
        then run ORB matching inside that group.

        Args:
            shape_name (str): Geometric class ("Triangle", "Circle", "Octagon").
            dominant_color (str): Color label ("Red", "Orange", "Blue", "Other").
            roi (np.ndarray): BGR region of interest (cropped sign candidate).

        Returns:
            name (str): Best matched sign name from the database, or "Unidentified".
            score (float): Matching score (0–100), higher is better.
        """
        # Select folder based on shape and color
        if shape_name == "Octagon" and dominant_color == "Red":
            group_folder = os.path.join(self.BASE_PATH, "stop_signs")
        elif shape_name == "Triangle" and dominant_color == "Orange":
            group_folder = os.path.join(self.BASE_PATH, "warning_signs")
        elif shape_name == "Circle" and dominant_color != "Blue":
            group_folder = os.path.join(self.BASE_PATH, "speed_limits_signs")
        elif shape_name == "Circle" and dominant_color == "Blue":
            if self.DEBUG:
                print(f"[DEBUG] Shape: {shape_name}, Dominant color: {dominant_color}")
            group_folder = os.path.join(self.BASE_PATH, "mandatory_signs")
        else:
            if self.DEBUG:
                print("[INFO] Unidentified sign candidate (shape/color combination).")
            return "Unidentified", 0

        if self.DEBUG:
            print(f"[INFO] Classification: shape={shape_name}, color={dominant_color}, folder={group_folder}")

        des_list, class_names, img_list = self.load_orb_database(group_folder)
        name, score, _ = self.orb_match(roi, des_list, class_names, img_list)
        return name, score

    def debug_display(self, blur, edges, roi, output):
        """Display debug / output windows depending on DEBUG/DISPLAY flags."""
        if not self.DISPLAY:
            return

        if self.DEBUG:
            cv2.imshow("Blurred", blur)
            cv2.imshow("Edges", edges)
            if roi is not None:
                cv2.imshow("Detected ROI", roi)
            cv2.imshow("Output", output)
        else:
            cv2.imshow("Output", output)


    # =========================================
    # --- SHAPE DETECTION ---
    # =========================================

    def detect(self, frame):
        """
        Detect traffic-sign candidate shapes on a BGR frame.

        Returns:
            output (np.ndarray): BGR frame with drawn contours and labels.
            edges  (np.ndarray): edge image (debug view).
            blur   (np.ndarray): preprocessed grayscale/blurred image (debug view).
            sign_info (dict): {
                "detected": bool,
                "detections": [
                    {
                        "name": str,
                        "score": float,
                        "bbox": (x, y, w, h),
                        "shape": str,
                        "color": str,
                    },
                    ...
                ]
            }
        """
        roi = None
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        equalized = self.clahe.apply(gray)
        blur = cv2.GaussianBlur(equalized, (5, 5), 0)
        edges = cv2.Canny(blur, 80, 170)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        output = frame.copy()
        hsv_frame = self.hsv_frame_process(frame)

        detections = []  # <<< TU zbieramy wszystkie znaki

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 600:
                continue

            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue

            if self.DEBUG:
                print(f"[DEBUG] Contour area: {area}")

            approx = cv2.approxPolyDP(cnt, 0.012 * perimeter, True)
            vertices = len(approx)
            circularity = 4 * math.pi * area / (perimeter * perimeter)
            x, y, w, h = cv2.boundingRect(approx)

            shape_name = None
            color_draw = (0, 255, 0)

            # Basic shape classification based on vertices and circularity
            if vertices <= 6 and circularity < 0.7:
                shape_name = "Triangle"
                color_draw = (0, 255, 255)
            elif vertices > 8 and circularity > 0.85:
                aspect_ratio = float(w) / h if h != 0 else float('inf')
                if 0.7 < aspect_ratio < 1.5:
                    shape_name = "Circle"
                    color_draw = (255, 0, 0)
            elif vertices == 8:
                aspect_ratio = float(w) / h if h != 0 else float('inf')
                if 0.9 < aspect_ratio < 1.1:
                    shape_name = "Octagon"
                    color_draw = (255, 0, 255)

            if not shape_name:
                continue

            dominant_color, fraction = self.color_checking_area(hsv_frame, approx, x, y, w, h)
            if not dominant_color:
                continue

            roi = frame[y:y + h, x:x + w]
            if roi.size == 0:
                continue

            if self.DISPLAY:
                cv2.imshow("ROI", roi)

            name, score = self.trafficsign_classifier(shape_name, dominant_color, roi)
            if name and score >= self.MIN_SCORE:
                # rysowanie jak wcześniej
                label = f"{name} ({score:.0f}%)"
                cv2.drawContours(output, [approx], -1, color_draw, 2)
                cv2.putText(output, label, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_draw, 2)

                # zapis do listy wykryć
                detections.append({
                    "name": name,
                    "score": float(score),
                    "bbox": (int(x), int(y), int(w), int(h)),
                    "shape": shape_name,
                    "color": dominant_color,
                })


        self.debug_display(blur, edges, roi, output)

        sign_info = {
            "detected": len(detections) > 0,
            "detections": detections,
            "output_frame": output
        }

        return sign_info



