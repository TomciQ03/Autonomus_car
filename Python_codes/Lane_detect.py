# ====================== Lane detection & crossroad detection module ======================
import cv2
import numpy as np
import time


class LaneDetector:
    """
    LaneDetector

    Responsibilities:
      - Apply HSV thresholding to extract lane markings.
      - Warp perspective to bird's-eye (IPM) view using a trapezoid.
      - Split the IPM into horizontal ROIs and compute histograms.
      - Estimate lane center, curvature and polynomial fit.
      - Estimate lane presence (left/right) and confidence from energy.
      - Detect simple crossroad types based on bottom ROI histogram:
          * "X_crossroad"
          * "T_crossroad"
          * "L_turn-straight"
          * "R_turn-straight"
          * "not_detected"
      - Optionally show debug windows and interactive trackbars.
    """

    # ------------- basic callback for trackbars -------------
    @staticmethod
    def _nothing(a):
        """Dummy callback for OpenCV trackbars."""
        pass

    # ------------- constructor -------------
    def __init__(
        self,
        frame_width: int = 480,
        frame_height: int = 240,
        debug: bool = True,
        display: bool = True,
        # initial IPM trapezoid (WidthTop, WidthBottom, HeightTop, HeightBottom)
        ipm_trapezoid_init=(140, 240, 116, 240),
    ):
        # Config
        self.debug = debug
        self.display = display

        # Image size that the detector expects (user should resize frames to this)
        self.frame_width = frame_width
        self.frame_height = frame_height

        # HSV threshold parameters for lane marking detection
        self.h_min = 0
        self.s_min = 0
        self.v_min = 200
        self.h_max = 179
        self.s_max = 255
        self.v_max = 255

        # Crossroad detection thresholds (will be updated by crossroad trackbars)
        self.k_center_energy = 0.06   # how much center energy means "center is filled"
        self.k_side_energy = 0.04     # minimal side energy to treat a lane as present
        self.k_peak_sim = 0.5         # similarity of peaks (0..1)
        self.spread_branch = 0.30     # spread > this => wide branch
        self.spread_lane = 0.30       # spread < this => narrow lane

        # Lane presence / confidence
        self.left_lane_ok = False
        self.right_lane_ok = False
        self.lanes_detected = False
        self.threshold_lane_detect = 0.05
        self.confidence_left = 0.0
        self.confidence_right = 0.0

        # Crossroad state
        self.crossroad_type = "not_detected"  # "not_detected", "T_crossroad", "X_crossroad", "L_turn-straight", "R_turn-straight"

        # Curve smoothing
        self.curve_list = []
        self.curve_list_length = 10

        # IPM trapezoid initial values
        self.ipm_trapezoid_init = ipm_trapezoid_init
        self.ipm_width_top, self.ipm_width_bottom, self.ipm_height_top, self.ipm_height_bottom = ipm_trapezoid_init


        # Initialize OpenCV trackbars (debug only)
        if self.debug:
            self._initialize_perspective_trackbars(
                self.ipm_trapezoid_init,
                self.frame_width,
                self.frame_height,
            )
            self._initialize_crossroad_trackbars()

    # ====================== TRACKBARS (PERSPECTIVE) ======================
    def _initialize_perspective_trackbars(self, initial_vals, wT=480, hT=240):
        """
        Initialize OpenCV window and trackbars for the IPM trapezoid.

        initial_vals: [widthTop, widthBottom, heightTop, heightBottom]
        wT, hT: expected frame width/height used for trackbar ranges.
        """
        cv2.namedWindow("Points_Setup")
        cv2.resizeWindow("Points_Setup", 600, 300)
        cv2.createTrackbar("Width Top", "Points_Setup", initial_vals[0], wT // 2, self._nothing)
        cv2.createTrackbar("Width Bottom", "Points_Setup", initial_vals[1], wT // 2, self._nothing)
        cv2.createTrackbar("Height Top", "Points_Setup", initial_vals[2], hT, self._nothing)
        cv2.createTrackbar("Height Bottom", "Points_Setup", initial_vals[3], hT, self._nothing)

    def _get_perspective_points_from_trackbars(self, wT=480, hT=240):
        """
        Get IPM trapezoid points.

        If debug is True -> read from OpenCV trackbars.
        If debug is False -> use fixed values from ipm_trapezoid_init.
        """
        if self.debug:
            # Read trapezoid coordinates from trackbars
            width_top = cv2.getTrackbarPos("Width Top", "Points_Setup")
            width_bottom = cv2.getTrackbarPos("Width Bottom", "Points_Setup")
            height_top = cv2.getTrackbarPos("Height Top", "Points_Setup")
            height_bottom = cv2.getTrackbarPos("Height Bottom", "Points_Setup")
        else:
            # Use fixed, initialized values (no GUI needed)
            width_top = self.ipm_width_top
            width_bottom = self.ipm_width_bottom
            height_top = self.ipm_height_top
            height_bottom = self.ipm_height_bottom

        # Symmetric trapezoid around the image center
        points = np.float32([
            [wT // 2 - width_top,     height_top],     # Top-Left
            [wT // 2 + width_top,     height_top],     # Top-Right
            [wT // 2 - width_bottom,  height_bottom],  # Bottom-Left
            [wT // 2 + width_bottom,  height_bottom],  # Bottom-Right
        ])
        return points


    # ====================== TRACKBARS (CROSSROAD) ======================
    def _initialize_crossroad_trackbars(self):
        """
        Create window + trackbars for tuning crossroad detection thresholds:

          - k_center_energy
          - k_side_energy
          - k_peak_sim
          - spread_branch
          - spread_lane
        """
        cv2.namedWindow("Crossroad_Debug")
        cv2.resizeWindow("Crossroad_Debug", 500, 300)

        # Trackbars 0..100 -> 0.00..1.00
        cv2.createTrackbar("k_center x100", "Crossroad_Debug", int(self.k_center_energy * 100), 100, self._nothing)
        cv2.createTrackbar("k_side   x100", "Crossroad_Debug", int(self.k_side_energy * 100),   100, self._nothing)
        cv2.createTrackbar("k_peak   x100", "Crossroad_Debug", int(self.k_peak_sim * 100),      100, self._nothing)
        cv2.createTrackbar("spread branch x100", "Crossroad_Debug", int(self.spread_branch * 100), 100, self._nothing)
        cv2.createTrackbar("spread lane   x100", "Crossroad_Debug", int(self.spread_lane * 100),   100, self._nothing)

    def _update_crossroad_params_from_trackbars(self):
        """Update crossroad detection thresholds from trackbars."""
        self.k_center_energy = cv2.getTrackbarPos("k_center x100", "Crossroad_Debug") / 100.0
        self.k_side_energy = cv2.getTrackbarPos("k_side   x100", "Crossroad_Debug") / 100.0
        self.k_peak_sim = cv2.getTrackbarPos("k_peak   x100", "Crossroad_Debug") / 100.0
        self.spread_branch = cv2.getTrackbarPos("spread branch x100", "Crossroad_Debug") / 100.0
        self.spread_lane = cv2.getTrackbarPos("spread lane   x100", "Crossroad_Debug") / 100.0

    # ====================== BASIC HELPERS ======================
    def _roi_segment(self, img, vertices=3):
        """
        Split the image into `vertices` horizontal stripes from top to bottom.

        Returns:
            rois: list of sub-images [roi_0, roi_1, ...]
            centers_y: list of vertical centers [y_center_0, y_center_1, ...]
        """
        h, w = img.shape[:2]
        if vertices <= 0:
            return [], []

        segment_h = h // vertices
        rois = []
        centers_y = []

        for i in range(vertices):
            y_start = i * segment_h
            y_end = h if i == vertices - 1 else (i + 1) * segment_h
            roi = img[y_start:y_end, :]
            y_center = (y_start + y_end) // 2

            rois.append(roi)
            centers_y.append(y_center)

        return rois, centers_y

    def _draw_perspective_points(self, img, points):
        """
        Draw IPM trapezoid points and outline on the image.

        Args:
            img: BGR image.
            points: 4 points [TL, TR, BL, BR].

        Returns:
            Image with drawn trapezoid.
        """
        reorder = np.array([0, 1, 3, 2])  # TL, TR, BR, BL
        pts = np.int32(points[reorder])
        for x, y in pts:
            cv2.circle(img, (int(x), int(y)), 15, (0, 0, 255), cv2.FILLED)
        cv2.polylines(img, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
        return img

    def _warp_image(self, img, points, w, h):
        """
        Apply perspective (bird's-eye) warp to the image based on 4 points.

        Args:
            img: Source image.
            points: 4 points (np.float32) for perspective transform.
            w, h: target width/height.

        Returns:
            Warped (IPM) image.
        """
        pts1 = np.float32(points)
        pts2 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        img_warp = cv2.warpPerspective(img, matrix, (w, h))
        return img_warp

    def _threshold(self, img):
        """
        Apply HSV-based thresholding and morphology to extract lane markings.

        Args:
            img: Input BGR frame or road image.

        Returns:
            Binary mask of lane candidates (optionally thinned).
        """
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_white = np.array([self.h_min, self.s_min, self.v_min])
        upper_white = np.array([self.h_max, self.s_max, self.v_max])
        mask_white = cv2.inRange(img_hsv, lower_white, upper_white)
        mask_white = cv2.morphologyEx(mask_white, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        mask_white = cv2.GaussianBlur(mask_white, (5, 5), 0)

        return mask_white

    @staticmethod
    def _normalize_point(point, y, w, h):
        """
        Normalize image coordinates to a compact range.

        Args:
            point: x coordinate in pixels.
            y: y coordinate in pixels.
            w: image width.
            h: image height.

        Returns:
            norm_point: x in [-1, 1] (-1 = far left, 0 = center, +1 = far right).
            norm_y: y in [0, 1] (0 = top of IPM, 1 = bottom of IPM).
        """
        norm_point = (point - (w / 2)) / (w / 2)
        norm_y = y / h
        return norm_point, norm_y

    @staticmethod
    def _fit_polynomial(road_points):
        """
        Fit a 2nd-degree polynomial x(y) to lane center points in normalized space.

        Args:
            road_points: list of (y_norm, x_norm) points, where
                         y_norm in [0,1], x_norm in [-1,1].

        Returns:
            x_fit: fitted x values (normalized) for sampled y.
            y_fit: sampled y values in [0,1].
            a, b, c: polynomial coefficients x = a*y^2 + b*y + c.
        """
        road_points_np = np.array(road_points, dtype=np.float32)
        xs = road_points_np[:, 1]
        ys = road_points_np[:, 0]

        coeffs = np.polyfit(ys, xs, 2)
        a, b, c = coeffs

        print(f"Polynomial coefficients: a={a:.4f}, b={b:.4f}, c={c:.4f}")

        y_fit = np.linspace(0, 1, num=100)
        x_fit = np.polyval(coeffs, y_fit)
        return x_fit, y_fit, a, b, c

    # ====================== ENERGY BASED LANE PRESENCE ======================
    def _energy_in_roi(self, left_half, right_half, h_roi, midpoint, w_roi):
        """
        Compute normalized energy in left and right halves of the 1D histogram.

        Args:
            left_half, right_half: 1D hist values for left / right half.
            h_roi: ROI height in pixels.
            midpoint: midpoint index (number of columns in left_half).
            w_roi: ROI width in pixels.

        Updates:
            self.left_lane_ok, self.right_lane_ok, self.lanes_detected
            self.confidence_left, self.confidence_right

        Returns:
            normalized_left_energy, normalized_right_energy
        """
        left_energy_sum = np.sum(left_half)
        right_energy_sum = np.sum(right_half)

        max_left_energy = h_roi * midpoint * 255
        max_right_energy = h_roi * (w_roi - midpoint) * 255

        norm_left = left_energy_sum / max_left_energy if max_left_energy > 0 else 0.0
        norm_right = right_energy_sum / max_right_energy if max_right_energy > 0 else 0.0

        self.confidence_left = min(norm_left / self.threshold_lane_detect, 1.0)
        self.confidence_right = min(norm_right / self.threshold_lane_detect, 1.0)

        self.left_lane_ok = norm_left > self.threshold_lane_detect
        self.right_lane_ok = norm_right > self.threshold_lane_detect
        self.lanes_detected = self.left_lane_ok and self.right_lane_ok

        return norm_left, norm_right

    # ====================== CROSSROAD DETECTION ======================
    def _crossroad_in_roi(self, his_values, left_mean, right_mean, h_roi, w_roi):
        """
        Analyze the bottom ROI 1D histogram and try to detect crossroad type.

        Sets self.crossroad_type to one of:
          - "X_crossroad"
          - "T_crossroad"
          - "L_turn-straight"
          - "R_turn-straight"
          - "not_detected"

        Also fills a "Crossroad_Debug" window with numeric info (if debug enabled).
        """
        # Update thresholds from GUI trackbars
        if self.debug:
            self._update_crossroad_params_from_trackbars()

        w_hist = len(his_values)
        if w_hist == 0:
            self.crossroad_type = "not_detected"
            return

        left_i = int(round(left_mean))
        right_i = int(round(right_mean))

        left_i = max(1, min(left_i, w_hist - 2))
        right_i = max(left_i + 1, min(right_i, w_hist - 1))

        # Split into three sections
        left_section = his_values[:left_i]
        center_section = his_values[left_i:right_i]
        right_section = his_values[right_i:]

        left_width = max(len(left_section), 1)
        center_width = max(len(center_section), 1)
        right_width = max(len(right_section), 1)

        def section_energy(section, width):
            energy = float(np.sum(section))
            max_energy = float(h_roi * width * 255)
            norm = energy / max_energy if max_energy > 0 else 0.0
            return energy, norm

        def section_spread(section):
            """
            Returns value in 0..1 – how wide is the "high" part of this section.
            """
            if section.size == 0:
                return 0.0
            peak_val = float(section.max())
            if peak_val <= 0:
                return 0.0
            thr = 0.3 * peak_val
            width_high = float(np.sum(section >= thr))
            return width_high / float(len(section))

        # Energies
        _, norm_left = section_energy(left_section, left_width)
        _, norm_center = section_energy(center_section, center_width)
        _, norm_right = section_energy(right_section, right_width)

        # Spread
        spread_left = section_spread(left_section)
        spread_center = section_spread(center_section)
        spread_right = section_spread(right_section)

        # Peaks & similarity
        left_peak = float(his_values[left_i])
        right_peak = float(his_values[right_i])
        peak_diff = abs(left_peak - right_peak)
        max_peak = max(left_peak, right_peak, 1.0)
        peaks_similar = (peak_diff <= self.k_peak_sim * max_peak)

        # Simple flags
        left_section_filled = norm_left > self.k_side_energy
        right_section_filled = norm_right > self.k_side_energy
        center_section_filled = norm_center > self.k_center_energy

        # Lane-like vs branch-like
        left_is_branch = spread_left > self.spread_branch
        right_is_branch = spread_right > self.spread_branch

        left_is_lane = spread_left < self.spread_lane
        right_is_lane = spread_right < self.spread_lane

        # ---- Classification according to your rules ----

        # 1) X-crossroad: center empty, peaks similar, both sides look like regular lanes
        if (not center_section_filled and
                peaks_similar and
                left_is_lane and right_is_lane):
            self.crossroad_type = "X_crossroad"

        # 2) T-crossroad: center filled, peaks similar, wide branches on both sides
        elif (center_section_filled and
              peaks_similar and
              left_is_branch and right_is_branch):
            self.crossroad_type = "T_crossroad"

        # 3) Left turn + straight: center empty, peaks different,
        #    wide branch only on the left, right looks like a regular lane
        elif (not center_section_filled and
              not peaks_similar and
              left_section_filled and left_is_branch and
              right_section_filled and right_is_lane):
            self.crossroad_type = "L_turn-straight"

        # 4) Right turn + straight: center empty, peaks different,
        #    wide branch only on the right, left looks like a regular lane
        elif (not center_section_filled and
              not peaks_similar and
              right_section_filled and right_is_branch and
              left_section_filled and left_is_lane):
            self.crossroad_type = "R_turn-straight"

        else:
            self.crossroad_type = "not_detected"

        # Debug window
        if self.debug:
            debug_img = np.zeros((260, 500, 3), np.uint8)
            y = 20
            dy = 20

            cv2.putText(debug_img,
                        f"k_center={self.k_center_energy:.2f}  "
                        f"k_side={self.k_side_energy:.2f}  "
                        f"k_peak={self.k_peak_sim:.2f}",
                        (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 255), 1)
            y += dy

            cv2.putText(debug_img,
                        f"Energies: L={norm_left:.3f}  C={norm_center:.3f}  R={norm_right:.3f}",
                        (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            y += dy

            cv2.putText(debug_img,
                        f"Spreads : L={spread_left:.3f}  C={spread_center:.3f}  R={spread_right:.3f}",
                        (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            y += dy

            cv2.putText(debug_img,
                        f"Peaks   : L={left_peak:.0f}  R={right_peak:.0f}  "
                        f"diff={peak_diff:.0f}  similar={peaks_similar}",
                        (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)
            y += dy

            cv2.putText(debug_img,
                        f"Filled  : L={left_section_filled}  "
                        f"C={center_section_filled}  R={right_section_filled}",
                        (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 255, 200), 1)
            y += dy

            cv2.putText(debug_img,
                        f"Lane    : L={left_is_lane}  R={right_is_lane}",
                        (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 255), 1)
            y += dy

            cv2.putText(debug_img,
                        f"Branch  : L={left_is_branch}  R={right_is_branch}",
                        (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 150, 150), 1)
            y += dy

            cv2.putText(debug_img,
                        f"TYPE: {self.crossroad_type}",
                        (10, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            cv2.imshow("Crossroad_Debug", debug_img)

    # ====================== HISTOGRAM & CENTER ======================
    def _get_histogram(self, img, min_per=0.1, region=1, display=None):
        """
        Compute lane center position using a column-wise histogram.

        Args:
            img: Binary or grayscale warped image.
            min_per: Threshold ratio of the maximum column value;
                     columns below this threshold are treated as noise.
            region:
                1  -> use full height (prediction, more stable).
                >1 -> use only bottom 1/region of image (current position).
            display:
                If True  -> build a debug histogram image.
                If None  -> use self.display.

        Returns:
            base_point: estimated lane center x-coordinate (in pixels).
            img_hist: debug image with plotted histogram or None.
            energy_left, energy_right: normalized energies (only meaningful for region==4).
        """
        if display is None:
            display = self.display

        h, w = img.shape[:2]

        # Region of interest
        if region == 1:
            roi = img
        else:
            start_row = h - h // region
            roi = img[start_row:, :]

        h_roi, w_roi = roi.shape[:2]
        his_values = np.sum(roi, axis=0)  # sum over rows
        midpoint = w_roi // 2
        left_half = his_values[:midpoint]
        right_half = his_values[midpoint:]

        # Lane energy / presence (only for bottom region, e.g. region=4)
        if region == 4:
            energy_left, energy_right = self._energy_in_roi(
                left_half, right_half, h_roi, midpoint, w_roi
            )
        else:
            energy_left, energy_right = 0.0, 0.0

        # Fallback to center if one side is empty
        if left_half.max() == 0 or right_half.max() == 0:
            base_point = midpoint
            img_hist = np.zeros((h, w, 3), np.uint8) if display else None
            return base_point, img_hist, energy_left, energy_right

        # Compute left/right means
        if region == 1:
            # Weighted centroid for prediction
            thr_left = min_per * (left_half.max() if left_half.max() > 0 else 1)
            idxs_l = np.where(left_half >= thr_left)[0]
            if idxs_l.size == 0:
                left_mean = np.argmax(left_half)
            else:
                weights_l = left_half[idxs_l].astype(np.float64)
                left_mean = np.average(idxs_l, weights=weights_l)

            thr_right = min_per * (right_half.max() if right_half.max() > 0 else 1)
            idxs_r = np.where(right_half >= thr_right)[0]
            if idxs_r.size == 0:
                right_mean = np.argmax(right_half) + midpoint
            else:
                weights_r = right_half[idxs_r].astype(np.float64)
                right_mean = np.average(idxs_r, weights=weights_r) + midpoint
        else:
            # Strongest peaks only
            left_mean = np.argmax(left_half)
            right_mean = np.argmax(right_half) + midpoint

        # Crossroad detection on bottom region (we can choose any region != 4,
        # but here we used region==1 in the calling pipeline)
        if region != 4:
            self._crossroad_in_roi(his_values, left_mean, right_mean, h_roi, w_roi)

        # Lane center
        base_point = int((int(round(left_mean)) + int(round(right_mean))) // 2)

        img_hist = None
        if display:
            img_hist = np.zeros((h, w, 3), np.uint8)
            maxv = his_values.max() if his_values.max() > 0 else 1
            for x, intensity in enumerate(his_values):
                h_line = int((intensity / maxv) * h)
                cv2.line(img_hist, (x, h), (x, h - h_line), (255, 0, 255), 1)

            lx = int(round(left_mean))
            rx = int(round(right_mean))
            cv2.circle(img_hist, (lx, h), 6, (0, 255, 0), cv2.FILLED)
            cv2.circle(img_hist, (rx, h), 6, (255, 0, 0), cv2.FILLED)
            cv2.circle(img_hist, (base_point, h), 8, (0, 255, 255), cv2.FILLED)

        return base_point, img_hist, energy_left, energy_right

    # ====================== MAIN PIPELINE ======================
    def process_frame(self, img_bgr):
        """
        Full lane detection pipeline on a single BGR frame.

        Returns a dictionary with the most important values:

            {
              "curve": float,
              "offset": float,          # normalized bottom lane center [-1,1]
              "poly": (a, b, c),        # polynomial coefficients
              "lane_ok": bool,
              "left_lane_ok": bool,
              "right_lane_ok": bool,
              "left_energy": float,
              "right_energy": float,
              "left_confidence": float,
              "right_confidence": float,
              "crossroad_type": str,
            }

        Also draws debug windows if self.display == True.
        """
        road_points = []
        h, w, c = img_bgr.shape

        # 1. Perspective trapezoid from trackbars
        points = self._get_perspective_points_from_trackbars(w, h)

        # 2. Threshold + warp (mask)
        img_thresh = self._threshold(img_bgr)
        img_warp = self._warp_image(img_thresh, points, w, h)

        # 3. Warp original BGR for visualizations
        img_warp_color = self._warp_image(img_bgr, points, w, h)
        img_warp_points = img_warp_color.copy()

        # 4. Split IPM into 3 horizontal ROIs
        rois, centers_y = self._roi_segment(img_warp, vertices=3)
        if len(rois) != 3:
            print("ROI segmentation failed.")
            return None

        roi_top, roi_middle, roi_bottom = rois
        y_top, y_middle, y_bottom = centers_y

        # 5. Histograms:
        #    region=4 -> bottom slice (current position)
        #    region=1 -> full height (prediction)
        middle_point_full_roi, img_mid_hist, energy_left, energy_right = self._get_histogram(
            img_warp, min_per=0.5, region=4, display=self.display
        )
        curve_avg_point_full_roi, img_curve_hist, _, _ = self._get_histogram(
            img_warp, min_per=0.5, region=1, display=self.display
        )

        # Per-ROI centers (no debug images)
        curve_top_roi, _, _, _ = self._get_histogram(roi_top, min_per=0.5, region=1, display=False)
        curve_middle_roi, _, _, _ = self._get_histogram(roi_middle, min_per=0.5, region=1, display=False)
        curve_bottom_roi, _, _, _ = self._get_histogram(roi_bottom, min_per=0.5, region=1, display=False)

        # 6. Normalize points
        norm_middle_full, norm_y_middle_full = self._normalize_point(
            middle_point_full_roi, h, w, h
        )
        norm_curve_full, norm_y_curve_full = self._normalize_point(
            curve_avg_point_full_roi, h / 2, w, h
        )
        norm_curve_top, norm_y_top = self._normalize_point(curve_top_roi, y_top, w, h)
        norm_curve_middle, norm_y_middle = self._normalize_point(curve_middle_roi, y_middle, w, h)
        norm_curve_bottom, norm_y_bottom = self._normalize_point(curve_bottom_roi, y_bottom, w, h)

        # Collect road points (y_norm, x_norm)
        road_points.append((norm_y_middle_full, norm_middle_full))
        road_points.append((norm_y_bottom, norm_curve_bottom))
        road_points.append((norm_y_middle, norm_curve_middle))
        road_points.append((norm_y_top, norm_curve_top))

        # 7. Compute curve (difference between bottom center and global prediction)
        curve_raw = norm_middle_full - norm_curve_full
        self.curve_list.append(curve_raw)
        if len(self.curve_list) > self.curve_list_length:
            self.curve_list.pop(0)
        curve = float(sum(self.curve_list) / len(self.curve_list))

        print("curve:", curve)
        print("center (bottom):", norm_middle_full)
        print("predicted center:", norm_curve_full)

        # 8. Fit polynomial
        x_fit, y_fit, a, b, c = self._fit_polynomial(road_points)

        # 9. Visualizations
        if self.display:
            # Lane centers on warped view
            cv2.circle(img_warp_points, (middle_point_full_roi, h), 7, (255, 0, 255), cv2.FILLED)
            cv2.circle(img_warp_points, (curve_avg_point_full_roi, (h // 2) - 20), 7, (255, 255, 255), cv2.FILLED)
            cv2.putText(
                img_warp_points,
                f"crossroad: {self.crossroad_type}",
                (150, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 165, 0),
                1,
            )

            # ROI centers
            cv2.circle(img_warp_points, (int(curve_top_roi), int(y_top)), 6, (0, 0, 255), cv2.FILLED)
            cv2.circle(img_warp_points, (int(curve_middle_roi), int(y_middle)), 6, (0, 255, 0), cv2.FILLED)
            cv2.circle(img_warp_points, (int(curve_bottom_roi), int(y_bottom)), 6, (255, 0, 0), cv2.FILLED)

            # Polynomial centerline in pixel space
            x_fit_px = (x_fit * (w / 2)) + (w / 2)
            y_fit_px = y_fit * h
            for x_px, y_px in zip(x_fit_px.astype(int), y_fit_px.astype(int)):
                cv2.circle(img_warp_points, (x_px, int(y_px)), 2, (0, 255, 255), -1)

            # Lane status & energies on mask image
            cv2.putText(img_warp, f"left lane {self.left_lane_ok}", (150, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 1)
            cv2.putText(img_warp, f"right lane {self.right_lane_ok}", (150, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 1)
            cv2.putText(img_warp, f"lanes detected {self.lanes_detected}", (150, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 1)
            cv2.putText(img_warp, f"left energy {energy_left:.3f}", (150, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 1)
            cv2.putText(img_warp, f"left conf {self.confidence_left * 100:.1f}%", (150, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 1)
            cv2.putText(img_warp, f"right energy {energy_right:.3f}", (150, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 1)
            cv2.putText(img_warp, f"right conf {self.confidence_right * 100:.1f}%", (150, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 1)

            # Main camera view with trapezoid
            img_main_view = self._draw_perspective_points(img_bgr.copy(), points)
            cv2.circle(
                img_main_view,
                (middle_point_full_roi, img_bgr.shape[0] - 10),
                10,
                (0, 255, 255),
                cv2.FILLED,
            )

            cv2.imshow("Main View", img_main_view)
            cv2.imshow("Warped", img_warp)
            cv2.imshow("Warped Points", img_warp_points)
            cv2.imshow("Mid Histogram", img_mid_hist)
            cv2.imshow("Curve Histogram", img_curve_hist)

        # 10. Return most important values
        result = {
            "curve": curve,
            "offset": norm_middle_full,
            "poly": (a, b, c),
            "lane_ok": self.lanes_detected,
            "left_lane_ok": self.left_lane_ok,
            "right_lane_ok": self.right_lane_ok,
            "left_energy": energy_left,
            "right_energy": energy_right,
            "left_confidence": self.confidence_left,
            "right_confidence": self.confidence_right,
            "crossroad_type": self.crossroad_type,
        }
        return result

"""
# ====================== SIMPLE TEST ENTRY POINT ======================
if __name__ == "__main__":
    img = cv2.imread("road_images/test5.jpg")
    img = cv2.resize(img, (480, 240))

    detector = LaneDetector(
        frame_width=480,
        frame_height=240,
        debug=True,
        display=True,
        ipm_trapezoid_init=(140, 240, 116, 240),
    )

    while True:
        print("\033c", end="")  # clear terminal
        start = time.time()
        result = detector.process_frame(img)
        end = time.time()
        print(f"Frame processing time: {end - start:.6f} s")
        print("Result dict:", result)

        if cv2.waitKey(30) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()
"""
"""
# ====================== SIMPLE TEST ENTRY POINT ======================
if __name__ == "__main__":
    base_name = "road_images/test_vid"
    tried = []
    cap = None
    for ext in ["", ".mp4", ".avi", ".mkv", ".mov"]:
        path = base_name + ext
        c = cv2.VideoCapture(path)
        tried.append(path)
        if c.isOpened():
            cap = c
            print(f"[INFO] Otworzono wideo: {path}")
            break

    if cap is None or not cap.isOpened():
        raise FileNotFoundError(
            f"Nie udało się otworzyć pliku wideo. Próbowano: {', '.join(tried)}"
        )

    detector = LaneDetector(
        frame_width=480,
        frame_height=240,
        debug=True,
        display=True,  
        ipm_trapezoid_init=(140, 240, 116, 240),
    )

    fps = cap.get(cv2.CAP_PROP_FPS)
    delay_ms = int(1000 / fps) if fps and fps > 0 else 33

    quit_all = False
    while True:
        ret, frame = cap.read()

        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        frame = cv2.resize(frame, (detector.frame_width, detector.frame_height))

        print("\033c", end="")  
        start = time.time()
        result = detector.process_frame(frame)
        end = time.time()

        print(f"Czas przetwarzania klatki: {end - start:.6f} s")
        print("Wynik:", result)

        key = cv2.waitKey(delay_ms) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("p"): 
            while True:
                k2 = cv2.waitKey(30) & 0xFF
                if k2 == ord("p"):
                    break
                if k2 == ord("q"):
                    quit_all = True
                    break
        if quit_all:
            break

    cap.release()
    cv2.destroyAllWindows()
"""