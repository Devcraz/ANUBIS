# Misc helpers for ANUBIS - EXIF, drawing, downloading

from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
import datetime
import os
import requests

def get_exif_data(image_path):
    """
    Extract EXIF data: datetime, GPS coords, orientation.
    Returns a dict with keys: datetime, lat, lon, orientation (if available)
    """
    try:
        img = Image.open(image_path)
        exif_data = img._getexif()
        if not exif_data:
            return None

        exif = {}
        for tag_id, value in exif_data.items():
            tag = TAGS.get(tag_id, tag_id)
            if tag == "DateTimeOriginal":
                exif['datetime'] = datetime.datetime.strptime(value, "%Y:%m:%d %H:%M:%S")
            if tag == "GPSInfo":
                gps = {}
                for gps_tag in value:
                    sub_tag = GPSTAGS.get(gps_tag, gps_tag)
                    gps[sub_tag] = value[gps_tag]
                if "GPSLatitude" in gps and "GPSLongitude" in gps:
                    lat = gps["GPSLatitude"]
                    lon = gps["GPSLongitude"]
                    lat = lat[0] + lat[1]/60.0 + lat[2]/3600.0
                    lon = lon[0] + lon[1]/60.0 + lon[2]/3600.0
                    if gps.get("GPSLatitudeRef") == "S":
                        lat = -lat
                    if gps.get("GPSLongitudeRef") == "W":
                        lon = -lon
                    exif['lat'] = lat
                    exif['lon'] = lon
        return exif
    except Exception as e:
        return None

def download_file(url, dest_path, chunk_size=8192):
    """Download a file with progress printing. No fancy bar, just text."""
    print(f"Downloading {url} ...")
    response = requests.get(url, stream=True)
    total = int(response.headers.get('content-length', 0))
    downloaded = 0
    with open(dest_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=chunk_size):
            f.write(chunk)
            downloaded += len(chunk)
            if total:
                percent = (downloaded / total) * 100
                print(f"\r{percent:.1f}%", end="")
    print("\nDone.")
    return True

def dms_to_decimal(deg, min, sec, ref):
    """Helper for EXIF GPS conversion."""
    decimal = deg + min/60.0 + sec/3600.0
    if ref in ['S', 'W']:
        decimal = -decimal
    return decimal