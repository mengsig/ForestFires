import os
import sys
import numpy as np
from pyretechnics.space_time_cube import SpaceTimeCube
import pyretechnics.burn_cells as bc
import matplotlib.pyplot as plt

script_dir   = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, os.pardir, os.pardir))
sys.path.insert(0, project_root)
from src.utils.loadingUtils import (
    load_raster,
    convert_to_cube,
        )
from src.utils.plottingUtils import (
    save_matrix_as_heatmap,
    save_matrix_as_contours,
    )

# Tunable parameters
savename = "adjacency"
time_steps = 24


savedir = f"results/{savename}"
os.makedirs(savedir, exist_ok = True)
# Directory containing the already-cropped rasters
raster_dir = "cropped_rasters"

# Load rasters (already cropped, no need to window)
xsubset = (200,400)
ysubset = (100,300)

slope = load_raster("slp", xsubset, ysubset)
aspect = load_raster("asp", xsubset, ysubset)
dem = load_raster("dem", xsubset, ysubset)
cc = load_raster("cc", xsubset, ysubset) 
cbd = load_raster("cbd", xsubset, ysubset) 
cbh = load_raster("cbh", xsubset, ysubset)
ch = load_raster("ch", xsubset, ysubset)
fuel_model = load_raster("fbfm", xsubset, ysubset)

#convert to cubes
slope_cube = convert_to_cube(slope, time_steps)
aspect_cube = convert_to_cube(aspect, time_steps)
dem_cube   = convert_to_cube(dem, time_steps)
cc_cube    = convert_to_cube(cc, time_steps)
cbd_cube   = convert_to_cube(cbd, time_steps)
cbh_cube   = convert_to_cube(cbh, time_steps)
ch_cube    = convert_to_cube(ch, time_steps)
fuel_model_cube    = convert_to_cube(fuel_model, time_steps)

# Define cube shape (e.g. 24 hours)
rows, cols = slope.shape
cube_shape = (time_steps, rows, cols)

# Sanity check prints
print("cube_shape:", cube_shape)
print("canopy_shape:", cc.shape)

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
#============================================================================================
# Specify a space-time region (t, (y_min, y_max), (x_min, x_max))
# within the extent of the SpaceTimeCube dimensions
#============================================================================================

t = cube_shape[0]
y = cube_shape[1]
x = cube_shape[2]
y_range = (0,y)
x_range = (0,x)
directions = 4 
azimuth_step = 360/directions
spread_rate_mean = np.zeros((4,y,x))
spread_azimuth = 0 # degrees clockwise from North on the horizontal plane
#orig_space_time_cubes = space_time_cubes.copy()
for i in range(directions):
    num_simulations = 0
    for step in range(t):
        if step % 24 != 0:
            continue
        print(step)
        num_simulations += 1
#============================================================================================
# Calculate combined fire behavior in the direction of the azimuth (with wind limit)
#============================================================================================
#        space_time_cubes = orig_space_time_cubes.copy()

        combined_behavior_limited = bc.burn_all_cells_toward_azimuth(space_time_cubes,
                                                                     spread_azimuth,
                                                                     step,
                                                                     y_range,
                                                                     x_range,
                                                                     surface_lw_ratio_model="rothermel")

#============================================================================================
# Calculate combined fire behavior in the direction of the azimuth (without wind limit)
#============================================================================================

        combined_behavior_unlimited = bc.burn_all_cells_toward_azimuth(space_time_cubes,
                                                                       spread_azimuth,
                                                                       step,
                                                                       y_range,
                                                                       x_range,
                                                                       use_wind_limit=False,
                                                                       surface_lw_ratio_model="rothermel")

        spread_rate_mean[i] += combined_behavior_limited["fireline_intensity"]

#============================================================================================
# Update spread azimuth angle
#============================================================================================
    spread_rate_mean[i,::,::] /= num_simulations
    spread_azimuth += azimuth_step

spread_rate_mean = np.log10(spread_rate_mean+1)
vmin = spread_rate_mean.min()
vmax = spread_rate_mean.max()

def build_edgelist_from_spread_rates(spread_rate_mean, x, y):
    """
    Constructs an adjacency list from directional spread rates.

    Parameters:
        spread_rate_mean (np.ndarray): shape (4, y, x), spread rates per direction.
        x (int): width of the grid.
        y (int): height of the grid.

    Returns:
        adjacency (list of tuples): (from_node, to_node, weight)
    """
    adjacency = []
    directions = {
        0: (0, -1),  # North
        1: (1, 0),   # East
        2: (0, 1),   # South
        3: (-1, 0),  # West
    }

    for j in range(y):
        for i in range(x):
            from_node = j * x + i
            for d, (dx, dy) in directions.items():
                ni, nj = i + dx, j + dy
                if 0 <= ni < x and 0 <= nj < y:
                    to_node = nj * x + ni
                    weight = spread_rate_mean[d, j, i]
                    adjacency.append((from_node, to_node, weight))
    return adjacency

def adjacency_list_to_matrix(edgelist, num_nodes):
    """
    Converts an edge list into a dense NumPy adjacency matrix.

    Parameters:
        edgelist (list of tuples): (from_node, to_node, weight)
        num_nodes (int): Total number of nodes in the graph.

    Returns:
        adj_matrix (np.ndarray): [num_nodes x num_nodes] weighted adjacency matrix.
    """
    adj_matrix = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for from_node, to_node, weight in edgelist:
        adj_matrix[from_node, to_node] = weight
    return adj_matrix

edgelist = build_edgelist_from_spread_rates(spread_rate_mean, x, y)
edgelist_array = np.array(edgelist, dtype=np.float32)
np.save(f"{savedir}/spread_edge_list.npy", edgelist_array)
#adjacency_matrix = adjacency_list_to_matrix(edgelist, int(x*y))

#============================================================================================
# Display combined fire behavior in the direction of the azimuth (with wind limit)
#============================================================================================


heatmap_configs = [
    {
        "matrix"  : spread_rate_mean[0,::,::],
        "colors"  : "hot",
        "units"   : "m/min",
        "title"   : "Fireline Intensity Adjacency North",
        "filename": f"{savedir}/adjacency_north.png",
        "vmin"    : vmin,
        "vmax"    : vmax,
    },
    {
        "matrix"  : spread_rate_mean[1,::,::],
        "colors"  : "hot",
        "units"   : "m/min",
        "title"   : "Fireline Intensity Adjacency East",
        "filename": f"{savedir}/adjacency_east.png",
        "vmin"    : vmin,
        "vmax"    : vmax,
    },
    {
        "matrix"  : spread_rate_mean[2,::,::],
        "colors"  : "hot",
        "units"   : "m/min",
        "title"   : "Fireline Intensity Adjacency South",
        "filename": f"{savedir}/adjacency_south.png",
        "vmin"    : vmin,
        "vmax"    : vmax,
    },
    {
        "matrix"  : spread_rate_mean[3,::,::],
        "colors"  : "hot",
        "units"   : "m/min",
        "title"   : "Fireline Intensity Adjacency West",
        "filename": f"{savedir}/adjacency_west.png",
        "vmin"    : vmin,
        "vmax"    : vmax,
    },
]

def save_matrix_as_heatmap(matrix, colors, units, title, filename, vmin=None, vmax=None, ticks=None):
    import matplotlib.pyplot as plt
    image    = plt.imshow(matrix, origin="lower", cmap=colors, vmin=vmin, vmax=vmax)
    colorbar = plt.colorbar(image, orientation="vertical", ticks=ticks)
    colorbar.set_label(units)
    plt.title(title)
    plt.savefig(filename)
    plt.close("all")


for heatmap_config in heatmap_configs:
    save_matrix_as_heatmap(**heatmap_config)

print((spread_rate_mean[0] == spread_rate_mean[1]).sum()/(x*y))
