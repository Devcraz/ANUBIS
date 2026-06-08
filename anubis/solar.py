# ANUBIS solar module: detects shadow direction and finds compatible locations on Earth
import cv2
import numpy as np
import datetime
import pysolar.solar as solar


class SolarDetector:

    def __init__(self, grid_step=0.5):
        self.grid_step = grid_step

    def detect_shadow_mask(self, image):
        """Dark, low-saturation pixels are considered shadows."""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower = np.array([0, 0, 0])
        upper = np.array([180, 80, 80])
        mask = cv2.inRange(hsv, lower, upper)
        return mask

    def estimate_shadow_angle(self, mask, image):
        """Find straight lines in the shadow mask and return their median angle."""
        edges = cv2.Canny(mask, 50, 150)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=50)

        if lines is None:
            return None

        angles = []
        for rho_theta in lines:
            rho, theta = rho_theta[0]
            angle_rad = theta - np.pi / 2
            angle_deg = np.degrees(angle_rad) % 360
            angles.append(angle_deg)

        if not angles:
            return None

        return np.median(angles)

    def compute_solar_azimuth(self, lat, lon, dt):
        """Solar azimuth in degrees (0 = North, clockwise)."""
        pysolar_az = solar.get_azimuth(lat, lon, dt)
        standard_az = (pysolar_az + 180) % 360
        return standard_az

    def find_compatible_locations(self, shadow_angle, timestamp, orientation=0):
        """Brute‑force grid search over the globe for matching solar azimuth."""
        if shadow_angle is None:
            return []

        compatible = []
        for lat in np.arange(-90, 90 + self.grid_step, self.grid_step):
            for lon in np.arange(-180, 180 + self.grid_step, self.grid_step):
                try:
                    solar_az = self.compute_solar_azimuth(lat, lon, timestamp)
                except:
                    continue

                expected_shadow = (solar_az + 180) % 360
                diff = abs((expected_shadow - shadow_angle + 180) % 360 - 180)

                if diff < 10:
                    compatible.append((lat, lon))

        return compatible

    # Full pipeline: load image, get datetime, detect shadow, compute locations
    def process_image(self, image_path, manual_datetime=None, orientation=0):
        img = cv2.imread(image_path)
        if img is None:
            return {"error": "Could not read image"}

        if manual_datetime:
            dt = manual_datetime
        else:
            from datetime import datetime, timezone
            dt = datetime.now(timezone.utc)

        if dt.tzinfo is None:
            from datetime import timezone
            dt = dt.replace(tzinfo=timezone.utc)

        mask = self.detect_shadow_mask(img)
        angle = self.estimate_shadow_angle(mask, img)
        if angle is None:
            return {"error": "Could not estimate shadow angle"}

        locations = self.find_compatible_locations(angle, dt, orientation)
        return {
            "shadow_angle": angle,
            "datetime": dt.isoformat(),
            "compatible_locations": locations,
            "count": len(locations)
        }