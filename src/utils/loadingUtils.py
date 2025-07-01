import os
import rasterio
import numpy as np
def load_raster(name, x_interval=None, y_interval=None, raster_dir = "cropped_rasters"):
    """
    Loads a cropped raster, optionally sub-windows it to x_interval/y_interval,
    then flips it vertically to match simulation orientation.
    
    Parameters
    ----------
    name : str
      Base name of the raster (without “_cropped.tif”).
    x_interval : tuple of int (start, end), optional
      Column indices to keep (0-based, [start:end]).
    y_interval : tuple of int (start, end), optional
      Row    indices to keep (0-based, [start:end]).
      
    Returns
    -------
    data : np.ndarray, float32, shape = (rows, cols)
    """
    path = os.path.join(raster_dir, f"{name}_cropped.tif")
    with rasterio.open(path) as src:
        data = src.read(1)  # full 2D array, shape (rows, cols)
    
    # apply subwindowing if requested
    if y_interval is not None or x_interval is not None:
        rows, cols = data.shape
        y0, y1 = y_interval if y_interval is not None else (0, rows)
        x0, x1 = x_interval if x_interval is not None else (0, cols)
        data = data[y0:y1, x0:x1]#
    # flip so row 0 becomes bottom
    data = np.flip(data, axis=0)
    return np.ascontiguousarray(data).astype(np.float32)

    


def convert_to_cube(data, time_steps, datatype = None):
    # TODO: figure out how to properly normalize renormalizing
    if datatype == "cbd":
        data = data/100.0
    elif datatype == "cc":
        data = data/100.0
    elif datatype == "fbfm":
        data = data/1000.0
    if datatype == "slp":
        data = np.tan(np.pi/180 * data)
    if datatype == "ch":
        data = data/100.0
    if datatype == "cbh":
        data = data/100.0
    data = np.flip(data, axis=0)
    return np.repeat(data[np.newaxis, ...], time_steps, axis=0)
