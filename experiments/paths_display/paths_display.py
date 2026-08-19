"""3D plot of the k=2 product Lyons cubature paths (noise_dim = 2).

Each path is a 2D Brownian driver omega(t) = (omega_1(t), omega_2(t)); we render
it as a 3D space-time curve with axes (omega_1, omega_2, t). The P = (3^2)^2 = 81
product paths form the tensor grid of the degree-3 Lyons 1D paths.

Outputs:
  - output/cubature_paths_3d.pdf  (vector manuscript figure)
  - output/cubature_paths_3d.png  (high-resolution preview)
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EXPERIMENTS_DIR = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, EXPERIMENTS_DIR)
OUTPUT_DIR = os.path.join(HERE, "output")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import Normalize

from wsckf.library.cubature_paths import build_product_lyons_paths
from mfts_test.plot_style import DOUBLE_COLUMN_WIDTH, save_publication_figure

NOISE_DIM = 2
ORDER = 2  # k -> (3^k)^2 = 81 paths
GAMMA = 0.6
N_GRID = 200


def main() -> None:

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    ts = np.linspace(0.0, 1.0, N_GRID)
    paths, weights = build_product_lyons_paths(ts, NOISE_DIM, k=ORDER, gamma=GAMMA)
    # Drop the degenerate (omega_1, omega_2) = (0, 0) path; its weight dominates
    # the colormap and line width, hiding variation among the remaining paths.
    keep = ~np.all(paths == 0.0, axis=(1, 2))
    paths = paths[keep]
    weights = weights[keep]
    P = paths.shape[0]
    wnorm = weights / weights.max()
    norm = Normalize(vmin=weights.min(), vmax=weights.max())
    path_colors = cm.viridis(norm(weights))

    figm = plt.figure(figsize=(DOUBLE_COLUMN_WIDTH, 5.4))
    ax = figm.add_subplot(111, projection="3d")
    for p in range(P):
        ax.plot(ts, paths[p, :, 0], paths[p, :, 1],
                color=path_colors[p], lw=0.8 + 2.0 * wnorm[p], alpha=0.85)
    ax.scatter(np.ones(P), paths[:, -1, 0], paths[:, -1, 1],
               c=weights, cmap="viridis", s=30, edgecolor="k", linewidth=0.4)
    ax.set(xlabel="t", ylabel=r"$\omega_1$", zlabel=r"$\omega_2$")
    ax.set_title(rf"Product Lyons paths ($d=2$, $k={ORDER}$, $P={P}$)")
    ax.view_init(elev=22, azim=-65)
    sm = cm.ScalarMappable(cmap="viridis", norm=norm)
    sm.set_array([])
    figm.colorbar(sm, ax=ax, fraction=0.03, pad=0.08, label="Cubature weight")
    png_out = os.path.join(OUTPUT_DIR, "cubature_paths_3d.png")
    plot_png, plot_pdf = save_publication_figure(figm, png_out)
    plt.close(figm)
    print(f"saved {plot_png}")
    print(f"saved {plot_pdf}")
    print(f"  P=(3^{ORDER})^2={P} paths, weights sum to {weights.sum():.6f}")


if __name__ == "__main__":
    main()
