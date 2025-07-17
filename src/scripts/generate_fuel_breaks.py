import os
import sys
import numpy as np

script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, os.pardir, os.pardir))
sys.path.insert(0, project_root)
from src.utils.networkUtils import (
    create_network_as_sparse_array,
    save_fuel_breaks,
)
from src.utils.parsingUtils import (
    parse_args,
)
from src.utils.centralityUtils import (
    domirank_centrality,
    bonacich_centrality,
    random_centrality,
    degree_centrality,
)

"""
This script computes fuel breaks based on different centrality measures for fire spread modeling.
"""

xlen, ylen, savename, centrality, _ = parse_args()
savedir = f"results/{savename}"

G = create_network_as_sparse_array(f"results/{savename}/spread_edge_list.txt")
G /= G.max()  # normalization
G_protected = create_network_as_sparse_array(
    f"results/{savename}/spread_edge_list_protected.txt"
)
G_protected /= G_protected.max()  # normalization

# used only for plotting
degree_distribution = degree_centrality(G)
degree_distribution += 1
plot_degree = np.reshape(degree_distribution, (xlen, ylen))

# Extracting the centrality measures based on the user input
print(f"[GENERATING-FUEL-BREAKS-{centrality}:] ")
centralityDistribution = np.empty_like(G.shape[0])
if centrality == "domirank":
    centralityDistribution = domirank_centrality(G)
    basename = f"{savedir}/domirank"
elif centrality == "protected_domirank":
    centralityDistribution = domirank_centrality(G_protected)
    basename = f"{savedir}/protected_domirank"
elif centrality == "random":
    centralityDistribution = random_centrality(G)
    basename = f"{savedir}/random"
elif centrality == "degree":
    centralityDistribution = degree_centrality(G)
    basename = f"{savedir}/degree"
elif centrality == "bonacich":
    centralityDistribution = bonacich_centrality(G, alpha=0.25)
    basename = f"{savedir}/bonacich"
else:
    raise ValueError("That centrality is not supported.")

reshapedDistribution = np.reshape(centralityDistribution, (xlen, ylen))

intervals = [0, 5, 10, 15, 20, 25, 30]
save_fuel_breaks(reshapedDistribution, plot_degree, basename, intervals, centrality)
