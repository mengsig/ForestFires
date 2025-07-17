def domirank_centrality(G, sigma=None):
    from scipy.sparse.linalg import eigs
    from DomiRank import domirank

    lambN, _ = eigs(G, k=1, which="SR")
    if sigma is None:
        sigma = 1 - 1 / (G.shape[0])
    else:
        sigma = sigma
    print(f"Using sigma {sigma}...")
    _, drDist = domirank(G, sigma=-sigma / lambN, analytical=True)
    drDist = drDist.real
    return drDist


def bonacich_centrality(G, alpha=0.25):
    import numpy as np
    from scipy.sparse import eye
    from scipy.sparse.linalg import eigs
    from scipy.sparse.linalg import spsolve

    lambN, _ = eigs(G, k=1, which="LR")
    beta = alpha / lambN
    M = eye(G.shape[0]) - beta * G
    bDist = spsolve(M, np.ones(G.shape[0]))
    bDist = bDist.real
    return bDist


def random_centrality(G):
    import numpy as np

    rDist = np.random.uniform(0, 1, int(G.shape[0]))
    return rDist


def degree_centrality(G):
    dDist = G.sum(axis=0)
    return dDist
