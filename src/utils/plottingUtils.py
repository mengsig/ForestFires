import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

cmap = plt.get_cmap('hot')
cmap.set_bad("purple")
def save_matrix_as_heatmap(matrix, colors, units, title, filename, vmin=None, vmax=None, ticks=None, norm=False):
    if colors == "hot":
        if norm:
            image    = plt.imshow(matrix, origin="lower", cmap=cmap, norm = LogNorm(vmin,vmax))
        else:
            image    = plt.imshow(matrix, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
    else:
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
