import os, numpy as np, pandas as pd, seaborn as sns, matplotlib.pyplot as plt

def pred_vs_true(y_true, y_pred, path="results/pred_vs_true.png"):
    yt, yp = np.ravel(y_true), np.ravel(y_pred)
    if yt.shape != yp.shape:
        raise ValueError(f"Shape mismatch: {yt.shape} vs {yp.shape}")

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df = pd.DataFrame({"y_true": yt, "y_pred": yp})

    g = sns.jointplot(data=df, x="y_true", y="y_pred",
                      kind="reg", scatter_kws={"s": 10, "alpha": 0.5})
    ax = g.ax_joint
    lo, hi = min(df.min()), max(df.max())
    ax.plot([lo, hi], [lo, hi], "--", lw=1)

    g.fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.show()
    return g

def ldos_pred_vs_true(x, y, Z_true, Z_pred, path="results/ldos_true_pred.png"):
    x = np.asarray(x); y = np.asarray(y)
    Z_true = np.asarray(Z_true); Z_pred = np.asarray(Z_pred)

    # shape checks
    if Z_true.shape != (len(y), len(x)):
        raise ValueError(f"Z_true shape {Z_true.shape} != ({len(y)}, {len(x)})")
    if Z_pred.shape != (len(y), len(x)):
        raise ValueError(f"Z_pred shape {Z_pred.shape} != ({len(y)}, {len(x)})")

    vmin = np.nanmin([Z_true.min(), Z_pred.min()])
    vmax = np.nanmax([Z_true.max(), Z_pred.max()])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    im0 = axes[0].pcolormesh(x, y, Z_true, shading="nearest", cmap="inferno",
                             vmin=vmin, vmax=vmax)
    axes[0].set(title="LDOS (true)", xlabel="site index", ylabel="frequency")

    im1 = axes[1].pcolormesh(x, y, Z_pred, shading="nearest", cmap="inferno",
                             vmin=vmin, vmax=vmax)
    axes[1].set(title="LDOS (pred)", xlabel="site index", ylabel="frequency")

    cbar = fig.colorbar(im1, ax=axes, location="right", shrink=0.9, pad=0.02)
    cbar.set_label("LDOS")

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.show()
    return fig, axes
  
def fidelity_vs_noise_plot(noise_vals, F_train_list, F_test_list,
                           path="results/fidelityvsnoise_nooverfitting.png",
                           title="Fidelity vs Noise"):
    noise = np.asarray(noise_vals)
    Ft = np.asarray(F_train_list)
    Fv = np.asarray(F_test_list)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(noise, Ft, marker="o", label="Train Fidelity")
    ax.plot(noise, Fv, marker="^", label="Test Fidelity")
    ax.set_xlabel("Noise std")
    ax.set_ylabel("Fidelity")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.show()
    return fig, ax
