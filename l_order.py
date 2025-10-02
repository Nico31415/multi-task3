import numpy as np
import matplotlib.pyplot as plt
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
    c=  (q_prime(z) / qz)
    k= (2 * beta_main / sqrt_k)
    return (2 * beta_main / sqrt_k) * (q_prime(z) / qz)

def compute_fd(beta_aux, beta_main, sqrt_k, k, dk_dbeta_aux, epsilon=1e-12):
    sqrt_k = max(sqrt_k, epsilon)
    k = max(k, epsilon)
    z = 2 * beta_main / sqrt_k
    qz = q(z)
    if np.isclose(qz, 0):
        return 0.0
    factor = (0.5 - (beta_main / sqrt_k) * (q_prime(z) / qz))
    return factor * (beta_aux / k) * dk_dbeta_aux

def compute_l_order_and_fd(lambda_pt, c_pt, beta_aux, beta_main, gamma, epsilon=1e-6):
    c_ft = compute_c_ft(lambda_pt, c_pt, beta_aux, gamma)
    sqrt_k = 2 * c_ft
    k = sqrt_k ** 2

    k_plus = (2 * compute_c_ft(lambda_pt, c_pt, beta_aux + epsilon, gamma))**2
    k_minus = (2 * compute_c_ft(lambda_pt, c_pt, beta_aux - epsilon, gamma))**2
    dk_dbeta_aux = (k_plus - k_minus) / (2 * epsilon)

    l_order = compute_l_order(beta_main, sqrt_k)
    fd = compute_fd(beta_aux, beta_main, sqrt_k, k, dk_dbeta_aux)
    return l_order, fd

# --- Plot Utility ---

def plot_heatmaps(X, Y, l_order_grid, fd_grid, title_prefix, x_label, y_label, log_x=False, log_y=False, settings_dict=None):
    import numpy as np
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

    if settings_dict:
        settings_text = ', '.join([f"{k} = {v}" for k, v in settings_dict.items()])
        fig.suptitle(f"{title_prefix} Settings: {settings_text}", fontsize=12, y=0.98)

    # --- Construct cell edges ---
    def get_edges(arr, log=False):
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

    X_edges = get_edges(X, log=log_x)
    Y_edges = get_edges(Y, log=log_y)
    X_grid, Y_grid = np.meshgrid(X_edges, Y_edges)

    for ax, data, title in zip(
        axs,
        [l_order_grid, fd_grid, l_order_grid + fd_grid],
        ["l-order", "Feature Dependence", "l-order + FD"]
    ):
        pcm = ax.pcolormesh(X_grid, Y_grid, data, shading='auto', cmap='viridis')
        ax.set_title(f"{title_prefix}: {title}")
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        fig.colorbar(pcm, ax=ax)

        if log_x:
            ax.set_xscale('log')
        if log_y:
            ax.set_yscale('log')

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    os.makedirs("figures", exist_ok=True)
    save_path = os.path.join("figures", f"{title_prefix}.png")
    plt.savefig(save_path)
    plt.show()
    plt.close()

# def plot_heatmaps(X, Y, l_order_grid, fd_grid, title_prefix, x_label, y_label, log_x=False, log_y=False, settings_dict=None):
#     import numpy as np
#     import matplotlib.pyplot as plt

#     fig, axs = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

#     if settings_dict:
#         settings_text = ', '.join([f"{k} = {v}" for k, v in settings_dict.items()])
#         fig.suptitle(f"{title_prefix} Settings: {settings_text}", fontsize=12, y=0.98)

#     # Create meshgrid for pcolormesh (must match data grid shape)
#     X_grid, Y_grid = np.meshgrid(X, Y)

#     for ax, data, title in zip(
#         axs,
#         [l_order_grid, fd_grid, l_order_grid + fd_grid],
#         ["l-order", "Feature Dependence", "l-order + FD"]
#     ):
#         # Transpose data because meshgrid assumes (Y, X) shape
#         pcm = ax.pcolormesh(
#             X_grid, Y_grid, data.T,  # <--- key fix: transpose
#             shading='auto',
#             cmap='viridis'
#         )

#         ax.set_title(f"{title_prefix}: {title}")
#         ax.set_xlabel(x_label)
#         ax.set_ylabel(y_label)

#         if log_x:
#             ax.set_xscale('log')
#         if log_y:
#             ax.set_yscale('log')

#         fig.colorbar(pcm, ax=ax)

#     plt.tight_layout(rect=[0, 0, 1, 0.93])
#     plt.show()


# def plot_heatmaps(X, Y, l_order_grid, fd_grid, title_prefix, x_label, y_label, log_x=False, log_y=False, settings_dict=None):
#     fig, axs = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

#     if settings_dict:
#         settings_text = ', '.join([f"{k} = {v}" for k, v in settings_dict.items()])
#         fig.suptitle(f"{title_prefix} Settings: {settings_text}", fontsize=12, y=0.98)

#     for ax, data, title in zip(
#         axs,
#         [l_order_grid, fd_grid, l_order_grid + fd_grid],
#         ["l-order", "Feature Dependence", "l-order + FD"]
#     ):
#         im = ax.imshow(
#             data,
#             aspect='auto',
#             origin='lower',
#             extent=[X[0], X[-1], Y[0], Y[-1]],
#             cmap='viridis'
#         )
#         ax.set_title(f"{title_prefix}: {title}")
#         ax.set_xlabel(x_label)
#         ax.set_ylabel(y_label)
#         fig.colorbar(im, ax=ax)

#         if log_x:
#             ax.set_xscale('log')
#         if log_y:
#             ax.set_yscale('log')

#     plt.tight_layout(rect=[0, 0, 1, 0.93])
#     plt.show()

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
        r"$\beta_{\mathrm{main}}$": beta_main,
        r"$c_{\mathrm{PT}}$": c_pt_fixed,
        r"$\gamma$": gamma
    }

    plot_heatmaps(
        beta_aux_vals, lambda_over_c_vals,
        l_order_grid, fd_grid,
        title_prefix="Group 1",
        x_label=r"$|\beta_{\mathrm{aux}}|$",
        y_label=r"$\lambda_{\mathrm{PT}} / c_{\mathrm{PT}}$",
        log_x=True,
        log_y=False,
        settings_dict=settings
    )

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
        r"$\beta_{\mathrm{main}}$": beta_main,
        r"$\lambda_{\mathrm{PT}}$": lambda_pt,
        r"$\gamma$": gamma
    }

    plot_heatmaps(
        beta_aux_vals, c_over_beta_vals,
        l_order_grid, fd_grid,
        title_prefix="Group 2",
        x_label=r"$|\beta_{\mathrm{aux}}|$",
        y_label=r"$c_{\mathrm{PT}} / |\beta_{\mathrm{main}}|$",
        log_x=True,
        log_y=True,
        settings_dict=settings
    )

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
        r"$\beta_{\mathrm{main}}$": beta_main,
        r"$c_{\mathrm{PT}}$": c_pt,
        r"$\lambda_{\mathrm{PT}}$": lambda_pt
    }

    plot_heatmaps(
        beta_aux_vals, gamma_vals,
        l_order_grid, fd_grid,
        title_prefix="Group 3",
        x_label=r"$|\beta_{\mathrm{aux}}|$",
        y_label=r"$\gamma$",
        log_x=True,
        log_y=True,
        settings_dict=settings
    )

# --- Group 4 (Corrected) ---
def group4():
    c_pt = 0.6
    beta_aux = 1
    gamma = 0

    lambda_over_c_vals = np.linspace(-1, 1, 1000)
    beta_main_vals = np.logspace(np.log10(5e-4), np.log10(5e-10), 1000)  # produces c_pt / beta_main from 0.1 to 5
    c_over_beta_vals = c_pt / beta_main_vals

    l_order_grid = np.zeros((len(lambda_over_c_vals), len(beta_main_vals)))
    fd_grid = np.zeros_like(l_order_grid)

    for i, lo_c in enumerate(lambda_over_c_vals):
        lambda_pt = lo_c * c_pt
        for j, beta_main in enumerate(beta_main_vals):
            l, f = compute_l_order_and_fd(lambda_pt, c_pt, beta_aux, beta_main, gamma)
            l_order_grid[i, j] = l
            fd_grid[i, j] = f

    settings = {
        r"$c_{\mathrm{PT}}$": c_pt,
        r"$\beta_{\mathrm{aux}}$": beta_aux,
        r"$\gamma$": gamma
    }

    plot_heatmaps(
        c_over_beta_vals, lambda_over_c_vals,
        l_order_grid, fd_grid,
        title_prefix="Group 4",
        x_label=r"$c_{\mathrm{PT}} / |\beta_{\mathrm{main}}|$",
        y_label=r"$\lambda_{\mathrm{PT}} / c_{\mathrm{PT}}$",
        log_x=True,
        log_y=False,
        settings_dict=settings
    )

# --- Run All Groups ---
if __name__ == "__main__":
    print("Generating Group 1 heatmaps...")
    group1()
    print("Generating Group 2 heatmaps...")
    group2()
    print("Generating Group 3 heatmaps...")
    group3()
    print("Generating Group 4 heatmaps...")
    group4()