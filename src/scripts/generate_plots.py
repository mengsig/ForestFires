import numpy as np
import matplotlib.pyplot as plt
import os
import sys
script_dir   = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, os.pardir, os.pardir))
sys.path.insert(0, project_root)
from src.utils.parsingUtils import (
    parse_args,
    )


import re
import ast

def extract_stats(path):
    """
    Parses a stats.txt file and returns:
      - acres_burned: float or None
      - phi_mean: float or None
      - phi_std: float or None
    """
    with open(path, 'r') as f:
        text = f.read()

    # Extract Acres Burned
    acres_match = re.search(r'Acres Burned:\s*([\d\.]+)', text)
    acres_burned = float(acres_match.group(1)) if acres_match else None

    # Extract Phi statistics dict
    phi_match = re.search(r'Phi.*?\{([^}]+)\}', text, re.DOTALL)
    phi_mean = phi_std = None
    if phi_match:
        phi_dict = ast.literal_eval('{' + phi_match.group(1) + '}')
        phi_mean = phi_dict.get('Mean')
        # Some files might use 'Stdev' or 'Std'
        phi_std = phi_dict.get('Stdev') or phi_dict.get('Std')

    return acres_burned, phi_mean, phi_std


xlen,ylen, savename, _, _ = parse_args()

folderName = f"results/{savename}"
centralities = ["degree", "bonacich", "domirank", "random"]
fuel_break_fraction = [0, 5, 10, 15, 20, 25, 30]
x = np.array(fuel_break_fraction)/100

fig, ax = plt.subplots()
for centrality in centralities:
    acres_burned = []
    phi_mean = []
    for frac in fuel_break_fraction:
        filename = f"{folderName}/{centrality}_{frac}/stats.txt"
        burnt, phi, _ = extract_stats(filename)
        acres_burned.append(burnt)
        phi_mean.append(phi)
    ax.plot(x, acres_burned, label = centrality)
ax.set_xlabel("fuel-break fraction")
ax.set_ylabel("acres burned")
plt.legend()
fig.savefig(f"{folderName}/burnt.png", dpi = 300)
plt.close()


fig, ax = plt.subplots()
for centrality in centralities:
    acres_burned = []
    phi_mean = []
    for frac in fuel_break_fraction:
        filename = f"{folderName}/{centrality}_{frac}/stats.txt"
        burnt, phi, _ = extract_stats(filename)
        acres_burned.append(burnt)
        phi_mean.append(phi)
    ax.plot(x, phi_mean, label = centrality)
ax.set_xlabel("fuel-break fraction")
ax.set_ylabel("phi")
plt.legend()
fig.savefig(f"{folderName}/phi.png", dpi = 300)
plt.close()
