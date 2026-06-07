# Solar analysis module for ANUBIS
# Estimates location from shadow direction and solar position.

import cv2
import numpy as np
import datetime
import pysolar.solar as solar
from .utils import get_exif_data


class SolarDetector:
    """
    Detects shadow direction in an image and uses solar geometry
    to find geographic locations compatible with the observed shadow.
    """

    def __init__(self, grid_step=0.5):
        self.grid_step = grid_step

    def detect_shadow_mask(self, image):
        """
        Create a binary mask where shadows are white.
        Uses HSV thresholding for dark and low-saturation areas.
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower = np.array([0, 0, 0])
        upper = np.array([180, 80, 80])
        mask = cv2.inRange(hsv, lower, upper)
        return mask

    def estimate_shadow_angle(self, mask, image):
        """
        Estimate the main direction of shadows.
        Uses edge detection + HoughLines to find straight lines in the shadow mask,
        then returns the median angle of those lines.
        Returns angle in degrees (0 = right, clockwise from right-hand side).
        """
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
        """
        Compute solar azimuth for a given location and datetime (UTC).
        Returns azimuth in degrees (0 = North, clockwise).
        """
        pysolar_az = solar.get_azimuth(lat, lon, dt)
        standard_az = (pysolar_az + 180) % 360
        return standard_az

    def find_compatible_locations(self, shadow_angle, timestamp, orientation=0):
        """
        Given shadow direction, UTC timestamp, and optional camera orientation,
        return a list of (lat, lon) where the solar azimuth matches.
        """
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

    def process_image(self, image_path, manual_datetime=None, orientation=0):
        """
        Full pipeline for an image:
         - loads image
         - extracts EXIF datetime if available
         - detects shadow mask and angle
         - computes compatible locations
        Returns dict with results.
        """
        img = cv2.imread(image_path)
        if img is None:
            return {"error": "Could not read image"}

        exif = get_exif_data(image_path)
        dt = None
        if exif and 'datetime' in exif:
            dt = exif['datetime']
        if manual_datetime:
            dt = manual_datetime
        if dt is None:
            return {"error": "No datetime found. Provide manual_datetime."}

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