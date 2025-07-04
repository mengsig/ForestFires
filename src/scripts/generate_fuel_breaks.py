import os 
import sys
import numpy as np
from DomiRank import domirank, find_eigenvalue
from scipy.sparse.linalg import eigs
script_dir   = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, os.pardir, os.pardir))
sys.path.insert(0, project_root)
from src.utils.networkUtils import (
    create_network,
    save_fuel_breaks,
        )
from src.utils.parsingUtils import (
    parse_args,
    )

xlen,ylen, savename, centrality, _ = parse_args()
savedir = f"results/{savename}"

G = create_network(f"results/{savename}/spread_edge_list.txt", sparse_array = True)
G /= G.max() #normalization
degree = G.sum(axis = 0)
degree += 1 #for plotting
plot_degree = np.reshape(degree, (xlen, ylen))
#for plotting
vmin = degree.min()
vmax = degree.max()

#sometimes you will need to change these parameters to get convergence
#extracting smallest (most negative) eigenvector
#lambN = find_eigenvalue(G, maxIter = 1000, dt = 0.01, checkStep = 100, maxDepth = 200, epsilon = 1E-5) 
if centrality == "domirank":
    lambN, _ = eigs(G, k = 1, which = "SR")

    sigma = 0.9999999
    sigma = 1-1/(G.shape[0])

    print(f"[GENERATING-FUEL-BREAKS-{centrality}:] using sigma {sigma}...")
    _, centralityDistribution = domirank(G, sigma = -sigma/lambN, analytical = True) 
    centralityDistribution = centralityDistribution.real
    basename = f"{savedir}/domirank"
elif centrality == "random":
    centralityDistribution = np.random.uniform(0,1,int(G.shape[0]))
    basename = f"{savedir}/random"
elif centrality == "degree":
    centralityDistribution = degree.copy()
    basename = f"{savedir}/degree"
elif centrality == "bonacich":
    import scipy as sp
    lambN, _ = eigs(G, k = 1, which = "LM")
    alpha = 0.5/lambN
    M = sp.sparse.eye(G.shape[0]) - alpha*G
    centralityDistribution = sp.sparse.linalg.spsolve(M, np.ones(G.shape[0]))
    centralityDistribution = centralityDistribution.real
    basename = f"{savedir}/bonacich"
else:
    raise ValueError("That centrality is not supported.")

reshapedDistribution = np.reshape(centralityDistribution, (xlen, ylen))
#renormalization for plotting
#reshapedDistribution += -reshapedDistribution.min()
#reshapedDistribution += 1.001

intervals = [0,5,10,15,20,25,30]
save_fuel_breaks(reshapedDistribution, plot_degree, basename, intervals, centrality)
