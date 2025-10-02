import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, ListedColormap
from matplotlib.cm import ScalarMappable
import os

# --- Core Functions ---

def q(z):
    return 2 - np.sqrt(4 + z**2) + z * np.arcsinh(z / 2)

def q_prime(z):
    return np.arcsinh(z / 2)

def compute_c_ft(lambda_pt, c_pt, beta_aux, gamma):
    beta_abs = abs(beta_aux)
    term1 = (lambda_pt / c_pt) + 1
    term2 = c_pt / beta_abs
    sqrt_term = np.sqrt(term2**2 + 1)
    return beta_abs * term1 * (term2 + sqrt_term) + gamma**2

def compute_l_order(beta_main, sqrt_k, epsilon=1e-12):
    sqrt_k = max(sqrt_k, epsilon)
    z = 2 * beta_main / sqrt_k
    qz = q(z)
    if np.isclose(qz, 0):
        return 0.0
    return (2 * beta_main / sqrt_k) * (q_prime(z) / qz)

def compute_fd(beta_aux, l_order, k, dk_dbeta_aux, epsilon=1e-12):
    k = max(k, epsilon)
    factor = 0.5 * (1.0 - l_order)  # uses l_order to avoid recomputation/noise
    return factor * (beta_aux / k) * dk_dbeta_aux

def compute_l_order_and_fd(lambda_pt, c_pt, beta_aux, beta_main, gamma, eps_rel=1e-6):
    c_ft = compute_c_ft(lambda_pt, c_pt, beta_aux, gamma)
    sqrt_k = 2 * c_ft
    k = sqrt_k ** 2

    # relative step for stability across scales of beta_aux
    eps = eps_rel * max(1.0, abs(beta_aux))
    k_plus  = (2 * compute_c_ft(lambda_pt, c_pt, beta_aux + eps, gamma))**2
    k_minus = (2 * compute_c_ft(lambda_pt, c_pt, beta_aux - eps, gamma))**2
    dk_dbeta_aux = (k_plus - k_minus) / (2 * eps)

    l_order = compute_l_order(beta_main, sqrt_k)
    fd = compute_fd(beta_aux, l_order, k, dk_dbeta_aux)
    return l_order, fd

# --- Shared utility for pcolormesh cell edges ---

def _get_edges(arr, log=False):
    arr = np.asarray(arr)
    if log:
        log_arr = np.log10(arr)
        step = np.diff(log_arr) / 2
        edges = np.concatenate([
            [log_arr[0] - step[0]],
            log_arr[:-1] + step,
            [log_arr[-1] + step[-1]]
        ])
        return 10**edges
    else:
        step = np.diff(arr) / 2
        edges = np.concatenate([
            [arr[0] - step[0]],
            arr[:-1] + step,
            [arr[-1] + step[-1]]
        ])
        return edges

# --- Original 3-panel Plotter ---

def plot_heatmaps(
    X, Y, l_order_grid, fd_grid, title_prefix,
    x_label, y_label, log_x=False, log_y=False, settings_dict=None
):
    """
    Produces the original 3-panel figure: l-order, FD, and l-order + FD.
    Saves to 'figures/{title_prefix}.png' and returns the path.
    """
    fig, axs = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

    if settings_dict:
        settings_text = ', '.join([f"{k} = {v}" for k, v in settings_dict.items()])
        fig.suptitle(f"{title_prefix} Settings: {settings_text}", fontsize=12, y=0.98)

    # Construct cell edges for correct pcolormesh geometry
    X_edges = _get_edges(X, log=log_x)
    Y_edges = _get_edges(Y, log=log_y)
    X_grid, Y_grid = np.meshgrid(X_edges, Y_edges)

    for ax, data, subtitle in zip(
        axs,
        [l_order_grid, fd_grid, l_order_grid + fd_grid],
        ["l-order", "Feature Dependence", "l-order + FD"]
    ):
        pcm = ax.pcolormesh(X_grid, Y_grid, data, shading='auto', cmap='viridis')
        ax.set_title(f"{subtitle}")
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        if log_x:
            ax.set_xscale('log')
        if log_y:
            ax.set_yscale('log')
        fig.colorbar(pcm, ax=ax)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    os.makedirs("figures", exist_ok=True)
    save_path = os.path.join("figures", f"{title_prefix}.png")
    plt.savefig(save_path, dpi=200)
    plt.show()
    plt.close()
    return save_path

# --- Compressed Single-Panel Plotter (2-color gradient + shade) ---

def plot_projected_heatmap(
    X, Y, l_order_grid, fd_grid, title_prefix, x_label, y_label,
    log_x=False, log_y=False, settings_dict=None, shade_strength=0.7,
    color_start=(0.1, 0.2, 0.7),   # deep blue
    color_end=(0.9, 0.35, 0.1)     # warm orange
):
    """
    Single-panel heatmap for constraint l + FD >= 1:
      - Gradient COLOR along the projection of (l, FD) onto the boundary line l+FD=1.
        Parameter t = l_proj in [0,1] (on the line FD=1-l). Color = lerp(color_start, color_end, t).
      - SHADE encodes the perpendicular shortfall below the line:
            d = max(0, 1 - (l+FD)) / sqrt(2)
        normalized to [0,1] over the grid; larger shortfall -> darker.
      - Points satisfying l+FD >= 1 get full brightness (no darkening).

    Two colorbars:
      - A custom gradient bar for the projection parameter t (= l_proj).
      - A grayscale bar for normalized shortfall distance.
    Saves to 'figures/{title_prefix} compressed.png' and returns the path.
    """
    assert l_order_grid.shape == fd_grid.shape, "Grid shapes must match."

    l = l_order_grid
    f = fd_grid

    # Color ratio from normalized ranges so both contribute equally:
    #   l in [1,2]  -> l_norm = clip(l-1, 0, 1)
    #   FD in [-1,0] -> fd_norm = clip(FD+1, 0, 1)
    #   t = l_norm / (l_norm + fd_norm)
    eps = 1e-12
    l_norm = np.clip(l - 1.0, 0.0, 1.0)
    fd_norm = np.clip(f + 1.0, 0.0, 1.0)
    denom = np.maximum(eps, l_norm + fd_norm)
    t = np.clip(l_norm / denom, 0.0, 1.0)

    # Radial scaling distance to the boundary l+FD=1 (only for l+FD >= 1)
    l_plus_f = l + f
    s = 1.0 / np.maximum(eps, l_plus_f)
    radial_dist = np.zeros_like(l)
    mask = l_plus_f >= 1.0
    radial_dist[mask] = (1.0 - s[mask]) * np.sqrt(l[mask]**2 + f[mask]**2)
    dmax = np.max(radial_dist) or 1.0
    dist_norm = np.clip(radial_dist / dmax, 0.0, 1.0)

    # Linear interpolation between start/end RGB colors
    c0 = np.array(color_start).reshape(1, 1, 3)
    c1 = np.array(color_end).reshape(1, 1, 3)
    t3 = t[..., None]  # (ny, nx, 1)
    rgb_line = (1.0 - t3) * c0 + t3 * c1  # (ny, nx, 3)

    # Apply shading by shortfall (points above/on the line remain bright)
    brightness = 1.0 - shade_strength * dist_norm[..., None]
    rgb = np.clip(rgb_line * brightness, 0.0, 1.0)
    rgba = np.concatenate([rgb, np.ones((*rgb.shape[:2], 1))], axis=-1)

    # Mesh for pcolormesh
    X_edges = _get_edges(X, log=log_x)
    Y_edges = _get_edges(Y, log=log_y)
    X_grid, Y_grid = np.meshgrid(X_edges, Y_edges)

    fig, ax = plt.subplots(1, 1, figsize=(10.0, 6.3))
    if settings_dict:
        settings_text = ', '.join([f"{k} = {v}" for k, v in settings_dict.items()])
        fig.suptitle(
            f"{title_prefix}  |  Gradient by normalized ratio $t=(\\ell-1)/((\\ell-1)+(\\mathrm{{FD}}+1))$, "
            f"Shade by radial distance to $\\ell+\\mathrm{{FD}}=1$  |  Settings: {settings_text}",
            fontsize=11, y=0.98
        )
    else:
        fig.suptitle(
            f"{title_prefix}  |  Gradient by normalized ratio $t=(\\ell-1)/((\\ell-1)+(\\mathrm{{FD}}+1))$, Shade by radial distance to $\\ell+\\mathrm{{FD}}=1$",
            fontsize=11, y=0.98
        )

    dummy = np.zeros_like(l)  # establish QuadMesh geometry
    quad = ax.pcolormesh(X_grid, Y_grid, dummy, shading='auto')
    quad.set_facecolors(rgba.reshape(-1, 4))
    quad.set_array(None)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if log_x:
        ax.set_xscale('log')
    if log_y:
        ax.set_yscale('log')

    # --- Colorbars / Legends ---

    # 1) Projection gradient bar from color_start -> color_end over t in [0,1]
    grad_cmap = ListedColormap([
        color_start,
        color_end
    ])
    # Create a smooth gradient by sampling the colormap finely
    grad_samples = np.linspace(0, 1, 256)
    grad_colors = (1 - grad_samples)[:, None] * np.array(color_start) + grad_samples[:, None] * np.array(color_end)
    grad_cmap = ListedColormap(grad_colors)

    t_norm = Normalize(vmin=0.0, vmax=1.0)
    t_mappable = ScalarMappable(norm=t_norm, cmap=grad_cmap)
    cbar1 = fig.colorbar(t_mappable, ax=ax, fraction=0.04, pad=0.07)
    cbar1.set_label(r"Color ratio (normalized): $t=\frac{\ell-1}{(\ell-1)+(\mathrm{FD}+1)}\in[0,1]$")

    # 2) Shortfall distance bar (0 = on/above line, 1 = max shortfall in plot)
    dist_mappable = ScalarMappable(norm=Normalize(0, 1), cmap='Greys_r')
    cbar2 = fig.colorbar(dist_mappable, ax=ax, fraction=0.04, pad=0.16)
    cbar2.set_label(r"Radial distance to boundary: $\| (\ell,\mathrm{FD})-(s\ell, s\,\mathrm{FD}) \|$, $s=\frac{1}{\ell+\mathrm{FD}}$, for $\ell+\mathrm{FD}\geq 1$")

    ax.set_title(
        r"Color$=\mathrm{lerp}(c_0,c_1;\;t=\frac{\ell-1}{(\ell-1)+(\mathrm{FD}+1)})$,  "
        r"Shade$=1-\alpha\,\hat d$;  "
        r"$\hat d=\frac{\|(\ell,\mathrm{FD})-(s\ell,s\,\mathrm{FD})\|}{d_{\max}}$, $s=\frac{1}{\ell+\mathrm{FD}}$",
        fontsize=10
    )

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    os.makedirs("figures", exist_ok=True)
    save_path = os.path.join("figures", f"{title_prefix} compressed.png")
    plt.savefig(save_path, dpi=200)
    plt.show()
    plt.close()
    return save_path

# --- Group 1 ---
def group1():
    beta_aux_vals = np.logspace(np.log10(0.0005), np.log10(2000), 100)
    lambda_over_c_vals = np.linspace(-1, 1, 100)
    beta_main = 1
    c_pt_fixed = 1e-1
    gamma = 0

    l_order_grid = np.zeros((len(lambda_over_c_vals), len(beta_aux_vals)))
    fd_grid = np.zeros_like(l_order_grid)

    for i, lo_c in enumerate(lambda_over_c_vals):
        lambda_pt = lo_c * c_pt_fixed
        for j, beta_aux in enumerate(beta_aux_vals):
            l, f = compute_l_order_and_fd(lambda_pt, c_pt_fixed, beta_aux, beta_main, gamma)
            l_order_grid[i, j] = l
            fd_grid[i, j] = f

    settings = {
        r"$\beta_{\mathrm{main}}$":  beta_main,
        r"$c_{\mathrm{PT}}$":        c_pt_fixed,
        r"$\gamma$":                 gamma
    }

    # Original 3-panel
    path_orig = plot_heatmaps(
        beta_aux_vals, lambda_over_c_vals,
        l_order_grid, fd_grid,
        title_prefix="Group 1",
        x_label=r"$|\beta_{\mathrm{aux}}|$",
        y_label=r"$\lambda_{\mathrm{PT}} / c_{\mathrm{PT}}$",
        log_x=True,
        log_y=False,
        settings_dict=settings
    )

    # Compressed single-panel
    path_comp = plot_projected_heatmap(
        beta_aux_vals, lambda_over_c_vals,
        l_order_grid, fd_grid,
        title_prefix="Group 1",
        x_label=r"$|\beta_{\mathrm{aux}}|$",
        y_label=r"$\lambda_{\mathrm{PT}} / c_{\mathrm{PT}}$",
        log_x=True,
        log_y=False,
        settings_dict=settings
    )

    return path_orig, path_comp

# --- Group 2 ---
def group2():
    beta_aux_vals = np.logspace(np.log10(0.0005), np.log10(20), 100)
    c_over_beta_vals = np.logspace(np.log10(0.1), np.log10(5), 100)
    beta_main = 1
    lambda_pt = 0.05
    gamma = 0

    l_order_grid = np.zeros((len(c_over_beta_vals), len(beta_aux_vals)))
    fd_grid = np.zeros_like(l_order_grid)

    for i, c_ratio in enumerate(c_over_beta_vals):
        c_pt = c_ratio * abs(beta_main)
        for j, beta_aux in enumerate(beta_aux_vals):
            l, f = compute_l_order_and_fd(lambda_pt, c_pt, beta_aux, beta_main, gamma)
            l_order_grid[i, j] = l
            fd_grid[i, j] = f

    settings = {
        r"$\beta_{\mathrm{main}}$":  beta_main,
        r"$\lambda_{\mathrm{PT}}$":  lambda_pt,
        r"$\gamma$":                 gamma
    }

    path_orig = plot_heatmaps(
        beta_aux_vals, c_over_beta_vals,
        l_order_grid, fd_grid,
        title_prefix="Group 2",
        x_label=r"$|\beta_{\mathrm{aux}}|$",
        y_label=r"$c_{\mathrm{PT}} / |\beta_{\mathrm{main}}|$",
        log_x=True,
        log_y=True,
        settings_dict=settings
    )

    path_comp = plot_projected_heatmap(
        beta_aux_vals, c_over_beta_vals,
        l_order_grid, fd_grid,
        title_prefix="Group 2",
        x_label=r"$|\beta_{\mathrm{aux}}|$",
        y_label=r"$c_{\mathrm{PT}} / |\beta_{\mathrm{main}}|$",
        log_x=True,
        log_y=True,
        settings_dict=settings
    )

    return path_orig, path_comp

# --- Group 3 ---
def group3():
    beta_aux_vals = np.logspace(np.log10(0.0005), np.log10(20), 10)
    gamma_vals = np.logspace(np.log10(1), np.log10(1000), 10)
    beta_main = 1
    c_pt = 0.001
    lambda_pt = 0.001

    l_order_grid = np.zeros((len(gamma_vals), len(beta_aux_vals)))
    fd_grid = np.zeros_like(l_order_grid)

    for i, gamma in enumerate(gamma_vals):
        for j, beta_aux in enumerate(beta_aux_vals):
            l, f = compute_l_order_and_fd(lambda_pt, c_pt, beta_aux, beta_main, gamma)
            l_order_grid[i, j] = l
            fd_grid[i, j] = f

    settings = {
        r"$\beta_{\mathrm{main}}$":  beta_main,
        r"$c_{\mathrm{PT}}$":        c_pt,
        r"$\lambda_{\mathrm{PT}}$":  lambda_pt
    }

    path_orig = plot_heatmaps(
        beta_aux_vals, gamma_vals,
        l_order_grid, fd_grid,
        title_prefix="Group 3",
        x_label=r"$|\beta_{\mathrm{aux}}|$",
        y_label=r"$\gamma$",
        log_x=True,
        log_y=True,
        settings_dict=settings
    )

    path_comp = plot_projected_heatmap(
        beta_aux_vals, gamma_vals,
        l_order_grid, fd_grid,
        title_prefix="Group 3",
        x_label=r"$|\beta_{\mathrm{aux}}|$",
        y_label=r"$\gamma$",
        log_x=True,
        log_y=True,
        settings_dict=settings
    )

    return path_orig, path_comp

# --- Group 4 (Fixed ranges to get c_PT / |beta_main| in [0.1, 5]) ---
def group4():
    c_pt = 0.6
    beta_aux = 1
    gamma = 0

    lambda_over_c_vals = np.linspace(-1, 1, 1000)

    # Choose beta_main so that x-axis c_pt / |beta_main| increases from 0.1 to 5
    beta_main_vals = np.logspace(np.log10(c_pt/0.1), np.log10(c_pt/5), 1000)  # 6.0 -> 0.12 for c_pt=0.6
    c_over_beta_vals = c_pt / beta_main_vals  # increases 0.1 -> 5

    l_order_grid = np.zeros((len(lambda_over_c_vals), len(beta_main_vals)))
    fd_grid = np.zeros_like(l_order_grid)

    for i, lo_c in enumerate(lambda_over_c_vals):
        lambda_pt = lo_c * c_pt
        for j, beta_main in enumerate(beta_main_vals):
            l, f = compute_l_order_and_fd(lambda_pt, c_pt, beta_aux, beta_main, gamma)
            l_order_grid[i, j] = l
            fd_grid[i, j] = f

    settings = {
        r"$c_{\mathrm{PT}}$":        c_pt,
        r"$\beta_{\mathrm{aux}}$":   beta_aux,
        r"$\gamma$":                 gamma
    }

    path_orig = plot_heatmaps(
        c_over_beta_vals, lambda_over_c_vals,
        l_order_grid, fd_grid,
        title_prefix="Group 4",
        x_label=r"$c_{\mathrm{PT}} / |\beta_{\mathrm{main}}|$",
        y_label=r"$\lambda_{\mathrm{PT}} / c_{\mathrm{PT}}$",
        log_x=True,
        log_y=False,
        settings_dict=settings
    )

    path_comp = plot_projected_heatmap(
        c_over_beta_vals, lambda_over_c_vals,
        l_order_grid, fd_grid,
        title_prefix="Group 4",
        x_label=r"$c_{\mathrm{PT}} / |\beta_{\mathrm{main}}|$",
        y_label=r"$\lambda_{\mathrm{PT}} / c_{\mathrm{PT}}$",
        log_x=True,
        log_y=False,
        settings_dict=settings
    )

    return path_orig, path_comp

# --- Run All Groups ---
if __name__ == "__main__":
    os.makedirs("figures", exist_ok=True)

    print("Generating Group 1 figures...")
    g1_orig, g1_comp = group1()
    print(f"Saved: {g1_orig}")
    print(f"Saved: {g1_comp}")

    print("Generating Group 2 figures...")
    g2_orig, g2_comp = group2()
    print(f"Saved: {g2_orig}")
    print(f"Saved: {g2_comp}")

    print("Generating Group 3 figures...")
    g3_orig, g3_comp = group3()
    print(f"Saved: {g3_orig}")
    print(f"Saved: {g3_comp}")

    print("Generating Group 4 figures...")
    g4_orig, g4_comp = group4()
    print(f"Saved: {g4_orig}")
    print(f"Saved: {g4_comp}")
