# Global imports
import os
import sys
import shutil
import numpy as np
from pprint import pprint
# Pyretechnics imports
from pyretechnics.space_time_cube import SpaceTimeCube
import pyretechnics.eulerian_level_set as els
import time

# Description: This script simulates fire spread modeling based on various input parameters.

script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, os.pardir, os.pardir))
sys.path.insert(0, project_root)
from src.utils.loadingUtils import (
    load_raster,
    normalize,
)
from src.utils.plottingUtils import (
    save_matrix_as_heatmap,
)
from src.utils.parsingUtils import (
    parse_args,
)

# ------------- Tunable parameters ------------- #
x_length, y_length, save_name, centrality, fuel_break_fraction = parse_args()
time_steps = int(2500 * np.sqrt(x_length * y_length) / 400)
# ---------- End of tunable parameters ---------- #

# Define the directory for saving results
savedir = f"results/{save_name}"
fuel_breaks_file = f"{savedir}/{centrality}_{fuel_break_fraction}.txt"
fuel_breaks_img = f"{savedir}/{centrality}_{fuel_break_fraction}.png"

# Create the output directory if it doesn't exist
if fuel_breaks_file:
    savedir = fuel_breaks_file.removesuffix(".txt")
os.makedirs(savedir, exist_ok=True)

# Load raster data with offsets
x_offset = 100
y_offset = 100
x_subset = (x_offset, int(x_offset + x_length))
y_subset = (y_offset, int(y_offset + y_length))

# Load raster data
slope = load_raster("slp", x_subset, y_subset)
aspect = load_raster("asp", x_subset, y_subset)
dem = load_raster("dem", x_subset, y_subset)
cc = load_raster("cc", x_subset, y_subset)
cbd = load_raster("cbd", x_subset, y_subset)
cbh = load_raster("cbh", x_subset, y_subset)
ch = load_raster("ch", x_subset, y_subset)
fuel_model = load_raster("fbfm", x_subset, y_subset)

# Implementing fuel breaks if the file is provided
if fuel_breaks_file:
    fuel_breaks = np.loadtxt(fuel_breaks_file).astype(bool)
    fuel_model[fuel_breaks] = 91
    cc[fuel_breaks] = 0
    cbd[fuel_breaks] = 0

# Clean up repository by moving the fuel_breaks image and removing the file
rel_img_name = fuel_breaks_img.split("/")[-1]
shutil.move(fuel_breaks_img, os.path.join(savedir, rel_img_name))
os.remove(fuel_breaks_file)

# Convert rasters to cubes for simulation
slope_cube = normalize(slope, time_steps, datatype="slp")
aspect_cube = normalize(aspect, time_steps, datatype="asp")
dem_cube = normalize(dem, time_steps, datatype="dem")
cc_cube = normalize(cc, time_steps, datatype="cc")
cbd_cube = normalize(cbd, time_steps, datatype="cbd")
cbh_cube = normalize(cbh, time_steps, datatype="cbh")
ch_cube = normalize(ch, time_steps, datatype="ch")
fuel_model_cube = normalize(fuel_model, time_steps, datatype="fbfm")

# Define cube shape based on raster dimensions
rows, cols = slope.shape
cube_shape = (time_steps, rows, cols)

# Build space-time cubes for different variables
space_time_cubes = {
    "slope": SpaceTimeCube(cube_shape, slope_cube),
    "aspect": SpaceTimeCube(cube_shape, aspect_cube),
    "fuel_model": SpaceTimeCube(cube_shape, fuel_model),
    "canopy_cover": SpaceTimeCube(cube_shape, cc_cube),
    "canopy_height": SpaceTimeCube(cube_shape, ch_cube),
    "canopy_base_height": SpaceTimeCube(cube_shape, cbh_cube),
    "canopy_bulk_density": SpaceTimeCube(cube_shape, cbd_cube),
    "wind_speed_10m": SpaceTimeCube(cube_shape, 0),
    "upwind_direction": SpaceTimeCube(cube_shape, 0),
    "fuel_moisture_dead_1hr": SpaceTimeCube(cube_shape, 0.10),
    "fuel_moisture_dead_10hr": SpaceTimeCube(cube_shape, 0.25),
    "fuel_moisture_dead_100hr": SpaceTimeCube(cube_shape, 0.50),
    "fuel_moisture_live_herbaceous": SpaceTimeCube(cube_shape, 0.90),
    "fuel_moisture_live_woody": SpaceTimeCube(cube_shape, 0.60),
    "foliar_moisture": SpaceTimeCube(cube_shape, 0.90),
    "fuel_spread_adjustment": SpaceTimeCube(cube_shape, 1.0),
    "weather_spread_adjustment": SpaceTimeCube(cube_shape, 1.0),
}

# Set up simulation parameters
start_time = 0  # in minutes
max_duration = int(time_steps * 3 / 4) * 60  # in minutes

# Ignite fire in the center of the grid
x_cord, y_cord = int(x_length / 2), int(y_length / 2)
num_burned_cells = 0
burned_cells_threshold = (x_length * y_length) / np.sqrt(x_length * y_length)

# Change this variable to control the number of simulations to aggregate mean
# behaviour over.
N = 100
for i in range(N):  # Run N simulations
    x_cord = np.random.randint(0, x_length)
    y_cord = np.random.randint(0, y_length)
    spread_state = els.SpreadState(cube_shape).ignite_cell((x_cord, y_cord))
    cube_resolution = (
        60,  # band_duration: minutes
        30,  # cell_height: meters
        30,  # cell_width: meters
    )

    #============================================================================================
    # Spread fire from the start time for the max duration
    #============================================================================================

    runtime_start = time.perf_counter()
    fire_spread_results = els.spread_fire_with_phi_field(
        space_time_cubes,
        spread_state,
        cube_resolution,
        start_time,
        max_duration,
        surface_lw_ratio_model="behave",
    )
    runtime_stop = time.perf_counter()
    stop_time = fire_spread_results["stop_time"]  # in minutes
    stop_condition = fire_spread_results["stop_condition"]  # "max duration reached" or "no burnable cells"
    spread_state = fire_spread_results["spread_state"]  # updated SpreadState object (mutated from inputs)
    temp = spread_state.get_full_matrices()

    # Aggregate results
    if i == 0:
        output_matrices = temp.copy()
    else:
        output_matrices["phi"] += temp['phi']
        output_matrices["fire_type"] += temp['fire_type']
        output_matrices["spread_rate"] += temp['spread_rate']
        output_matrices["spread_direction"] += temp['spread_direction']
        output_matrices["fireline_intensity"] += temp['fireline_intensity']
        output_matrices["flame_length"] += temp['flame_length']
        output_matrices["time_of_arrival"] += temp['time_of_arrival']
 
 # Normalize results after N simulations
output_matrices["phi"] /= N
#output_matrices["fire_type"] /= N
output_matrices["spread_rate"] /= N
output_matrices["spread_direction"] /= N
output_matrices["fireline_intensity"] /= N
output_matrices["flame_length"] /= N
output_matrices["time_of_arrival"] /= N

#============================================================================================
# Print out the acres burned, total runtime, and runtime per burned cell
#============================================================================================

num_burned_cells = np.count_nonzero(output_matrices["fire_type"])  # cells
acres_burned = num_burned_cells / 4.5  # in acres

print(f"[SIMULATE-{centrality}]: Acres Burned: " + str(acres_burned))

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


#----------------- Saving statistics -----------------
vmin, vmax = output_matrices["fireline_intensity"].min() + 1, output_matrices["fireline_intensity"].max()
from contextlib import redirect_stdout
out_path = f"{savedir}/stats.txt"
with open(out_path, "w") as fout, redirect_stdout(fout):
     print("Fire Behavior from Day 2 @ 10:30am - Day 2 @ 6:30pm Spreading from Coordinate (50,50)\n" + "=" * 100)

     print(f"Acres Burned: " + str(acres_burned))
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

 #------------ Finished saving statistics ------------
import matplotlib.pyplot as plt
import numpy as np

output_matrices["fireline_intensity"] += 0.01
# See https://matplotlib.org/stable/gallery/color/colormap_reference.html for the available options for "colors"
heatmap_configs = [
     {
         "matrix"  : output_matrices["phi"],
         "colors"  : "plasma",
         "units"   : "phi <= 0: burned, phi > 0: unburned",
         "title"   : "Phi",
         "filename": f"{savedir}/els_phi.png",
     },
     {
         "matrix"  : output_matrices["fire_type"],
         "colors"  : "viridis",
         "units"   : "0=unburned, 1=surface, 2=passive_crown, 3=active_crown",
         "title"   : "Fire Type",
         "filename": f"{savedir}/els_fire_type.png",
         "vmin"    : 0,
         "vmax"    : 3,
         "ticks"   : [0,1,2,3],
     },
     {
         "matrix"  : output_matrices["spread_rate"],
         "colors"  : "hot",
         "units"   : "m/min",
         "title"   : "Spread Rate",
         "filename": f"{savedir}/els_spread_rate.png",
     },
     {
         "matrix"  : output_matrices["spread_direction"],
         "colors"  : "viridis",
         "units"   : "degrees clockwise from North",
         "title"   : "Spread Direction",
         "filename": f"{savedir}/els_spread_direction.png",
         "vmin"    : 0,
         "vmax"    : 360,
         "ticks"   : [0,45,90,135,180,225,270,315,360]
     },
     {
         "matrix"  : output_matrices["fireline_intensity"],
         "colors"  : "hot",
         "units"   : "kW/m",
         "title"   : "Fireline Intensity",
         "filename": f"{savedir}/els_fireline_intensity.png",
         "vmin"    : output_matrices['fireline_intensity'].min(),
         "vmax"    : output_matrices['fireline_intensity'].max(),
         "norm"    : True
     },
     {
         "matrix"  : output_matrices["flame_length"],
         "colors"  : "hot",
         "units"   : "meters",
         "title"   : "Flame Length",
         "filename": f"{savedir}/els_flame_length.png",
     },
     {
             "matrix"  : np.flip(cc_cube[0,:,:],axis=0),
         "colors"  : "Greens",
         "units"   : "coverage",
         "title"   : "canopy coverage",
         "filename": f"{savedir}/els_canopy_coverage.png",
     },
     {
         "matrix"  : output_matrices["time_of_arrival"],
         "colors"  : "viridis",
         "units"   : "minutes",
         "vmin"    : 1,
         "vmax"    : stop_time,
         "title"   : "Time of Arrival",
         "filename": f"{savedir}/els_time_of_arrival.png",
     },
]

output_matrices["phi"][0:x_length, 0:y_length][fuel_breaks] = np.nan
output_matrices["fire_type"][0:x_length, 0:y_length][fuel_breaks] = 0
output_matrices["spread_rate"][0:x_length, 0:y_length][fuel_breaks] = np.nan
output_matrices["spread_direction"][0:x_length, 0:y_length][fuel_breaks] = np.nan
output_matrices["fireline_intensity"][0:x_length, 0:y_length][fuel_breaks] = np.inf
output_matrices["flame_length"][0:x_length, 0:y_length][fuel_breaks] = np.nan
output_matrices["time_of_arrival"][0:x_length, 0:y_length][fuel_breaks] = np.nan

# contour_configs = []

for heatmap_config in heatmap_configs:
    save_matrix_as_heatmap(**heatmap_config)
