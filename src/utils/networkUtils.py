import networkx as nx
import numpy as np

def create_network(edgelist_path, sparse_array = False):
    edgelist = np.loadtxt(edgelist_path)
    G = nx.DiGraph()
    for u, v, w in edgelist:
        G.add_edge(int(u), int(v), weight=np.log(float(w)+1)) 
    if sparse_array:
        GAdj = nx.to_scipy_sparse_array(G)
        return GAdj
    else:
        return G

def save_fuel_breaks(data, plot_degreec, basename, intervals):
    import os
    import sys
    script_dir   = os.path.dirname(__file__)
    project_root = os.path.abspath(os.path.join(script_dir, os.pardir, os.pardir))
    sys.path.insert(0, project_root)
    from src.utils.plottingUtils import save_matrix_as_heatmap
    plot_degree = plot_degreec.copy()
    vmin = plot_degree.min()
    vmax = plot_degree.max()
    for cutoff in intervals:
        fuel_breaks = data > np.percentile(data, 100 - cutoff)
        plot_degree = plot_degreec.copy()
        plot_degree[fuel_breaks] = np.inf
        try:
            np.savetxt(f"{basename}_{cutoff}.txt", fuel_breaks)
            print(f"Saved file: {basename}_{cutoff}.txt")

            domirankConfig = {
                    "matrix"  : plot_degree,
                    "colors"  : "hot",
                    "units"   : "m/min",
                    "title"   : "domirank on degree",
                    "filename": f"{basename}_{cutoff}.png",
                    "vmin"    : vmin,
                    "vmax"    : vmax,
                    }
            save_matrix_as_heatmap(**domirankConfig)
        except:
            raise ValueError("Problem saving file")
    #plot original
    config = {
            "matrix"  : plot_degreec,
            "colors"  : "hot",
            "units"   : "m/min",
            "title"   : "adjacency",
            "filename": f"{basename}_adjacency.png",
            "vmin"    : plot_degreec.min(),
            "vmax"    : plot_degreec.max(),
            "norm"    : True
            }
    save_matrix_as_heatmap(**config)
