import os
import shutil

import rasterio
import requests
from pyproj import Transformer
from rasterio.windows import from_bounds

# ========== Configuration ==========

# Bounding box in WGS84 (EPSG:4326)
min_lon, min_lat, max_lon, max_lat = -121.67, 39.71, -121.47, 39.91

#min_lon, min_lat, max_lon, max_lat = (
#   -117.52,
#   33.45,
#   -117.32,
#   33.65,
#)
# Output directory
output_dir = "cropped_rasters"
os.makedirs(output_dir, exist_ok=True)

# Remote base URL
base_url = "https://data.pyrecast.org/fuels_and_topography/ca-2021-fuelscape/"
file_map = {
    "asp": "asp.tif",
    "cbd": "cbd.tif",
    "cbh": "cbh.tif",
    "cc": "cc.tif",
    "ch": "ch.tif",
    "dem": "dem.tif",
    "slp": "slp.tif",
    "fbfm": "fbfm40.tif",
}

# ========== Helper Function ==========


def reproject_bounds_if_needed(src, bounds):
    """Reprojects WGS84 bounds to raster CRS if needed."""
    if src.crs.to_epsg() != 4326:
        transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
        min_x, min_y = transformer.transform(bounds[0], bounds[1])
        max_x, max_y = transformer.transform(bounds[2], bounds[3])
        return min_x, min_y, max_x, max_y
    else:
        return bounds


# ========== Main Logic ==========

for shortname, filename in file_map.items():
    print(f"\n⬇️  Downloading {filename}...")
    url = base_url + filename
    local_path = os.path.join(output_dir, filename)

    # Download
    #    try:
    #        if shortname != "asp":
    #            response = requests.get(url, stream=True)
    #            response.raise_for_status()
    #            with open(local_path, "wb") as f:
    #                shutil.copyfileobj(response.raw, f)
    #    except Exception as e:
    #        print(f"❌ Failed to download {filename}: {e}")
    #        continue

    # Crop
    try:
        with rasterio.open(local_path) as src:
            crop_bounds = reproject_bounds_if_needed(
                src, (min_lon, min_lat, max_lon, max_lat)
            )
            window = from_bounds(*crop_bounds, transform=src.transform)
            data = src.read(1, window=window)

            if data.shape[0] == 0 or data.shape[1] == 0:
                print(f"⚠️  Skipped {filename}: crop window is empty")
                os.remove(local_path)
                continue

            # Prepare cropped output
            profile = src.profile.copy()
            profile.update(
                {
                    "height": data.shape[0],
                    "width": data.shape[1],
                    "transform": src.window_transform(window),
                }
            )

            cropped_path = os.path.join(output_dir, f"{shortname}_cropped.tif")
            with rasterio.open(cropped_path, "w", **profile) as dst:
                dst.write(data, 1)

            print(f"✅ Cropped and saved to: {cropped_path}")

    except Exception as e:
        print(f"❌ Failed to crop {filename}: {e}")

    finally:
        if os.path.exists(local_path):
            os.remove(local_path)

print("\n🏁 Done processing all rasters.")
