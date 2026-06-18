"""Orientable 3D plot of the k=2 product Lyons cubature paths (noise_dim = 2).

Each path is a 2D Brownian driver omega(t) = (omega_1(t), omega_2(t)); we render
it as a 3D space-time curve with axes (omega_1, omega_2, t). The P = (3^2)^2 = 81
product paths form the tensor grid of the degree-3 Lyons 1D paths.

Outputs:
  - plots/cubature_paths_3d.html  (interactive, rotatable)
  - plots/cubature_paths_3d.png   (static snapshot)
"""

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from matplotlib import cm
from matplotlib.colors import Normalize, to_hex

from cubature_paths import build_product_lyons_paths

NOISE_DIM = 2
ORDER = 2  # k -> (3^k)^2 = 81 paths
GAMMA = 0.6
N_GRID = 200


def main() -> None:
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

    # ---- Interactive Plotly (orientable in a browser) ----
    fig = go.Figure()
    for p in range(P):
        fig.add_trace(
            go.Scatter3d(
                x=ts,
                y=paths[p, :, 0],
                z=paths[p, :, 1],
                mode="lines",
                # Scalar line.color ignores colorscale; use explicit Viridis RGB.
                line=dict(color=to_hex(path_colors[p]), width=2 + 6 * wnorm[p]),
                opacity=0.85,
                showlegend=False,
                hovertemplate=f"path {p}<br>weight={weights[p]:.4f}<extra></extra>",
            )
        )
    # Terminal cubature support points at t = 1.
    fig.add_trace(
        go.Scatter3d(
            x=np.ones(P), y=paths[:, -1, 0], z=paths[:, -1, 1],
            mode="markers",
            marker=dict(size=4, color=weights, colorscale="Viridis",
                        cmin=weights.min(), cmax=weights.max(),
                        colorbar=dict(title="weight"), line=dict(width=1, color="black")),
            showlegend=False,
            hovertext=[f"w={w:.4f}" for w in weights],
        )
    )
    fig.update_layout(
        title=f"Product Lyons Wiener cubature paths (noise_dim=2, k={ORDER}, P={P})",
        scene=dict(xaxis_title="t", yaxis_title="omega_1", zaxis_title="omega_2",
                   camera=dict(eye=dict(x=1.6, y=1.6, z=0.9))),
        width=950, height=800,
    )
    
    # uncomment to save an interactive HTML file (open in a browser and drag to rotate):
    #html_out = "plots/cubature_paths_3d.html"
    #fig.write_html(html_out)
    #print(f"saved {html_out}  (open in a browser and drag to rotate)")

    # ---- Static Matplotlib snapshot (for inline viewing) ----
    figm = plt.figure(figsize=(10, 8))
    ax = figm.add_subplot(111, projection="3d")
    for p in range(P):
        ax.plot(ts, paths[p, :, 0], paths[p, :, 1],
                color=path_colors[p], lw=0.8 + 2.0 * wnorm[p], alpha=0.85)
    ax.scatter(np.ones(P), paths[:, -1, 0], paths[:, -1, 1],
               c=weights, cmap="viridis", s=30, edgecolor="k", linewidth=0.4)
    ax.set(xlabel="t", ylabel=r"$\omega_1$", zlabel=r"$\omega_2$")
    ax.set_title(f"Product Lyons cubature paths (noise_dim=2, k={ORDER}, P={P})")
    ax.view_init(elev=22, azim=-65)
    sm = cm.ScalarMappable(cmap="viridis", norm=norm)
    sm.set_array([])
    figm.colorbar(sm, ax=ax, fraction=0.03, pad=0.08, label="cubature weight")
    png_out = "plots/cubature_paths_3d.png"
    figm.savefig(png_out, dpi=150, bbox_inches="tight")
    plt.close(figm)
    print(f"saved {png_out}")
    print(f"  P=(3^{ORDER})^2={P} paths, weights sum to {weights.sum():.6f}")


if __name__ == "__main__":
    main()
