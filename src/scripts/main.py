#==============================================================import os
import os
import numpy as np
import rasterio
from pyretechnics.space_time_cube import SpaceTimeCube
import pyretechnics.eulerian_level_set as els
import time
import pprint

# Directory containing the already-cropped rasters
raster_dir = "cropped_rasters"

def load_raster(name, x_interval=None, y_interval=None):
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
        data = data[y0:y1, x0:x1]
    
    # flip so row 0 becomes bottom
    data = np.flip(data, axis=0)
    return np.ascontiguousarray(data).astype(np.float32)

# Load rasters (already cropped, no need to window)
xsubset = (200,400)
ysubset = (100,300)

slope = load_raster("slp", xsubset, ysubset)
aspect = load_raster("asp", xsubset, ysubset)
dem = load_raster("dem", xsubset, ysubset)
canopy_cover = load_raster("cc", xsubset, ysubset) 
canopy_bulk_density = load_raster("cbd", xsubset, ysubset) 
canopy_base_height = load_raster("cbh", xsubset, ysubset)
canopy_height = load_raster("ch", xsubset, ysubset)
fuel_model = load_raster("fbfm", xsubset, ysubset)
#x,xfin,y,yfin = (100, 150, 100, 150)
#canopy_cover[x:xfin, y:yfin] = 0
#canopy_height[x:xfin, y:yfin] = 0
#canopy_bulk_density[x:xfin, y:yfin] = 0
#canopy_base_height[x:xfin, y:yfin] = 0
#fuel_model[x:xfin, y:yfin] = 0

# Define cube shape (e.g. 24 hours)
time_steps = 1500
rows, cols = slope.shape
cube_shape = (time_steps, rows, cols)

# Option A: np.repeat
slope_cube = np.repeat(np.tan(np.pi/180*slope[np.newaxis, ...]), time_steps, axis=0)
aspect_cube = np.repeat(aspect[np.newaxis, ...], time_steps, axis=0)
dem_cube   = np.repeat(dem[np.newaxis, ...],   time_steps, axis=0)
cc_cube    = np.repeat(canopy_cover[np.newaxis, ...],    time_steps, axis=0)/100
cbd_cube   = np.repeat(canopy_bulk_density[np.newaxis, ...], time_steps, axis=0)/100
cbh_cube   = np.repeat(canopy_base_height[np.newaxis, ...],   time_steps, axis=0)
ch_cube    = np.repeat(canopy_height[np.newaxis, ...],        time_steps, axis=0)
fuel_model_cube    = np.repeat(fuel_model[np.newaxis, ...],        time_steps, axis=0)/1000


# Now all these *_cube arrays have shape (time_steps, rows, cols)
print(slope_cube.shape, aspect_cube.shape, dem_cube.shape, cc_cube.shape)
print("fuel min", fuel_model_cube.min())

# Sanity check prints
print("cube_shape:", cube_shape)
print("canopy_shape:", canopy_cover.shape)

# Build space-time cubes
space_time_cubes = {
    "slope"                        : SpaceTimeCube(cube_shape, slope_cube),
    "aspect"                       : SpaceTimeCube(cube_shape, aspect_cube),
    "fuel_model"                   : SpaceTimeCube(cube_shape, fuel_model),
    "canopy_cover"                 : SpaceTimeCube(cube_shape, cc_cube),
    "canopy_height"                : SpaceTimeCube(cube_shape, ch_cube),
    "canopy_base_height"           : SpaceTimeCube(cube_shape, cbh_cube),
    "canopy_bulk_density"          : SpaceTimeCube(cube_shape, cbd_cube),
    "wind_speed_10m"               : SpaceTimeCube(cube_shape, 10),
    "upwind_direction"             : SpaceTimeCube(cube_shape, 45),
    "fuel_moisture_dead_1hr"       : SpaceTimeCube(cube_shape, 0.10),
    "fuel_moisture_dead_10hr"      : SpaceTimeCube(cube_shape, 0.25),
    "fuel_moisture_dead_100hr"     : SpaceTimeCube(cube_shape, 0.50),
    "fuel_moisture_live_herbaceous": SpaceTimeCube(cube_shape, 0.90),
    "fuel_moisture_live_woody"     : SpaceTimeCube(cube_shape, 0.60),
    "foliar_moisture"              : SpaceTimeCube(cube_shape, 0.90),
    "fuel_spread_adjustment"       : SpaceTimeCube(cube_shape, 1.0),
    "weather_spread_adjustment"    : SpaceTimeCube(cube_shape, 1.0),
}
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
plt.imshow(space_time_cubes["aspect"].data[0,::,::])
plt.colorbar()
fig.savefig('aspect.png', dpi = 300)
plt.close()

fig, ax = plt.subplots()
plt.imshow(space_time_cubes["canopy_cover"].data[0,::,::])
plt.colorbar()
fig.savefig('canopy_cover.png', dpi = 300)
plt.close()

fig, ax = plt.subplots()
plt.imshow(space_time_cubes["canopy_bulk_density"].data[0,::,::])
plt.colorbar()
fig.savefig('canopy_bulk_density.png', dpi = 300)
plt.close()

fig, ax = plt.subplots()
plt.imshow(space_time_cubes["fuel_model"].data[0,::,::])
plt.colorbar()
fig.savefig('fuel_model.png', dpi = 300)
plt.close()
# Day 2 @ 10:30am
start_time = (24 * 60) + (10 * 60) + 30 # minutes

# 8 hours
max_duration = 1000 * 60 # minutes

spread_state = els.SpreadState(cube_shape).ignite_cell((190,190))

cube_resolution = (
    60, # band_duration: minutes
    30, # cell_height:   meters
    30, # cell_width:    meters
)

#============================================================================================
# Spread fire from the start time for the max duration
#============================================================================================

runtime_start       = time.perf_counter()
fire_spread_results = els.spread_fire_with_phi_field(space_time_cubes,
                                                     spread_state,
                                                     cube_resolution,
                                                     start_time,
                                                     max_duration,
                                                     surface_lw_ratio_model="rothermel")
runtime_stop        = time.perf_counter()
stop_time           = fire_spread_results["stop_time"]      # minutes
stop_condition      = fire_spread_results["stop_condition"] # "max duration reached" or "no burnable cells"
spread_state        = fire_spread_results["spread_state"]   # updated SpreadState object (mutated from inputs)
output_matrices     = spread_state.get_full_matrices()

#============================================================================================
# Print out the acres burned, total runtime, and runtime per burned cell
#============================================================================================

num_burned_cells        = np.count_nonzero(output_matrices["fire_type"]) # cells
acres_burned            = num_burned_cells / 4.5                         # acres
simulation_runtime      = runtime_stop - runtime_start                   # seconds
runtime_per_burned_cell = 1000.0 * simulation_runtime / num_burned_cells # ms/cell

print("Acres Burned: " + str(acres_burned))
print("Total Runtime: " + str(simulation_runtime) + " seconds")
print("Runtime Per Burned Cell: " + str(runtime_per_burned_cell) + " ms/cell")


import numpy as np
from pprint import pprint

#============================================================================================
# Display summary statistics of our fire spread results
#============================================================================================

# Used as a filter in get_array_stats below
burned_cells = output_matrices["fire_type"] > 0

def get_array_stats(array, use_burn_scar_mask=True):
    array_values_to_analyze = array[burned_cells] if use_burn_scar_mask else array
    if len(array_values_to_analyze) > 0:
        return {
            "Min"  : np.min(array_values_to_analyze),
            "Max"  : np.max(array_values_to_analyze),
            "Mean" : np.mean(array_values_to_analyze),
            "Stdev": np.std(array_values_to_analyze),
        }
    else:
        return {
            "Min"  : "No Data",
            "Max"  : "No Data",
            "Mean" : "No Data",
            "Stdev": "No Data",
        }

print("Fire Behavior from Day 2 @ 10:30am - Day 2 @ 6:30pm Spreading from Coordinate (50,50)\n" + "=" * 100)

print("Stop Time: " + str(stop_time) + " (minutes)")
print("Stop Condition: " + stop_condition)

print("\nPhi (phi <= 0: burned, phi > 0: unburned")
pprint(get_array_stats(output_matrices["phi"], use_burn_scar_mask=False), sort_dicts=False)

print("\nFire Type (0=unburned, 1=surface, 2=passive_crown, 3=active_crown)")
pprint(get_array_stats(output_matrices["fire_type"]), sort_dicts=False)

print("\nSpread Rate (m/min)")
pprint(get_array_stats(output_matrices["spread_rate"]), sort_dicts=False)

print("\nSpread Direction (degrees clockwise from North)")
pprint(get_array_stats(output_matrices["spread_direction"]), sort_dicts=False)

print("\nFireline Intensity (kW/m)")
pprint(get_array_stats(output_matrices["fireline_intensity"]), sort_dicts=False)

print("\nFlame Length (meters)")
pprint(get_array_stats(output_matrices["flame_length"]), sort_dicts=False)

print("\nTime of Arrival (minutes)")
pprint(get_array_stats(output_matrices["time_of_arrival"]), sort_dicts=False)



import matplotlib.pyplot as plt
import numpy as np


def save_matrix_as_heatmap(matrix, colors, units, title, filename, vmin=None, vmax=None, ticks=None):
    image    = plt.imshow(matrix, origin="lower", cmap=colors, vmin=vmin, vmax=vmax)
    colorbar = plt.colorbar(image, orientation="vertical", ticks=ticks)
    colorbar.set_label(units)
    plt.title(title)
    plt.savefig(filename)
    plt.close("all")


def save_matrix_as_contours(matrix, title, filename, levels=None):
    fig, ax = plt.subplots()
    cs      = ax.contour(matrix, levels=levels)
    ax.clabel(cs, inline=True, fontsize=10)
    ax.set_aspect('equal', 'box')
    ax.set_title(title)
    plt.savefig(filename)
    plt.close("all")


# See https://matplotlib.org/stable/gallery/color/colormap_reference.html for the available options for "colors"
heatmap_configs = [
    {
        "matrix"  : output_matrices["phi"],
        "colors"  : "plasma",
        "units"   : "phi <= 0: burned, phi > 0: unburned",
        "title"   : "Phi",
        "filename": "org/pics/els_phi.png",
    },
    {
        "matrix"  : output_matrices["fire_type"],
        "colors"  : "viridis",
        "units"   : "0=unburned, 1=surface, 2=passive_crown, 3=active_crown",
        "title"   : "Fire Type",
        "filename": "org/pics/els_fire_type.png",
        "vmin"    : 0,
        "vmax"    : 3,
        "ticks"   : [0,1,2,3],
    },
    {
        "matrix"  : output_matrices["spread_rate"],
        "colors"  : "hot",
        "units"   : "m/min",
        "title"   : "Spread Rate",
        "filename": "org/pics/els_spread_rate.png",
    },
    {
        "matrix"  : output_matrices["spread_direction"],
        "colors"  : "viridis",
        "units"   : "degrees clockwise from North",
        "title"   : "Spread Direction",
        "filename": "org/pics/els_spread_direction.png",
        "vmin"    : 0,
        "vmax"    : 360,
        "ticks"   : [0,45,90,135,180,225,270,315,360]
    },
    {
        "matrix"  : output_matrices["fireline_intensity"],
        "colors"  : "hot",
        "units"   : "kW/m",
        "title"   : "Fireline Intensity",
        "filename": "org/pics/els_fireline_intensity.png",
    },
    {
        "matrix"  : output_matrices["flame_length"],
        "colors"  : "hot",
        "units"   : "meters",
        "title"   : "Flame Length",
        "filename": "org/pics/els_flame_length.png",
    },
]


contour_configs = [
    {
        "matrix"  : output_matrices["time_of_arrival"],
        "title"   : "Time of Arrival",
        "filename": "org/pics/els_time_of_arrival.png",
        "levels"  : int(start_time) + np.asarray(range(0, int(max_duration) + 1, 60)),
    },
]


for heatmap_config in heatmap_configs:
    save_matrix_as_heatmap(**heatmap_config)


for contour_config in contour_configs:
    save_matrix_as_contours(**contour_config)

print(fire_spread_results)
import imageio
# --- parameters for the GIF ---
sample_interval = 200                # minutes between frames
vmin, vmax = 0, np.log10(output_matrices["fireline_intensity"].max()+1)
out_dir     = "gif_frames"
gif_name    = "org/pics/fireline_intensity.gif"
duration    = 0.5                   # seconds per frame

# make sure output directory exists
os.makedirs(out_dir, exist_ok=True)

# extract final matrices
fi  = output_matrices["fireline_intensity"]   # kW/m
toa = output_matrices["time_of_arrival"]      # minutes since ignition

# generate list of times (0 .. stop_time) at your interval
times = np.arange(0, int(stop_time) + 1, sample_interval)

print(times)
frames = []
for t in times:
    print(f"{t}/{times[-1]}")
    # mask to only show cells that have ignited by time t
    mask  = (toa > 0) & (toa <= t)
    frame = np.where(mask, fi, 0.0)

    # plot
    fig, ax = plt.subplots(figsize=(6,6))
    img = ax.imshow(np.log10(frame+1), origin="lower", cmap="hot", vmin=vmin, vmax=vmax)
    ax.set_title(f"Fireline Intensity (kW/m)\nTime = {t} min")
    ax.axis("off")
    plt.colorbar(img, ax=ax, label="kW/m", fraction=0.046, pad=0.04)

    # save frame
    fname = os.path.join(out_dir, f"frame_{t:04d}.png")
    fig.savefig(fname, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)

    frames.append(imageio.v2.imread(fname))

# write out the GIF
imageio.mimsave(gif_name, frames, duration=duration, loop = 0)
print(f"→ Saved GIF: {gif_name}")
