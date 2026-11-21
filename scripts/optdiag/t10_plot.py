r"""Predicted against actual gain of every Gauss--Newton step of the fits of `t2_fit.py`, and the depth moves.

Usage: run.sh t10_plot.py NAME [NAME ...]
"""
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
from common import FIGS, OUT, log
from scipy.spatial import KDTree

from bbf.inference import init as I
from bbf.io import load_patch

names = sys.argv[1:]
fits = {nm: pickle.loads((OUT / f"fit_{nm}.pkl").read_bytes()) for nm in names}
fig, axs = plt.subplots(1, 3, figsize=(18, 5.5))
for i, (nm, f) in enumerate(fits.items()):
    r = f["recs"][1:]
    pred, act = np.array([x["pred"] for x in r]), np.array([x["actual"] for x in r])
    ok = pred > 0
    axs[0].loglog(pred[ok], np.maximum(act[ok], 1.0), "o-", ms=4, lw=0.7, color=f"C{i}", label=nm)
    axs[1].plot(np.arange(1, len(r) + 1), np.where(ok, act / np.where(ok, pred, 1.0), np.nan), "o-", ms=4,
                color=f"C{i}", label=nm)
    axs[2].semilogy(np.arange(1, len(r) + 1), [max(x["n_clip"], 0.5) for x in r], "o-", ms=4, color=f"C{i}", label=nm)
lim = axs[0].get_xlim()
axs[0].plot(lim, lim, "k--", lw=0.8)
axs[0].set_xlabel("gain predicted by the GGN for the applied step (nats)")
axs[0].set_ylabel("actual gain (nats)")
axs[0].set_title("predicted vs actual gain per step")
axs[1].axhline(1.0, color="k", ls="--", lw=0.8)
axs[1].set_ylim(-0.2, 2.2)
axs[1].set_xlabel("step")
axs[1].set_ylabel("actual / predicted")
axs[1].set_title("gain ratio (missing: predicted gain not positive)")
axs[2].set_xlabel("step")
axs[2].set_ylabel("vertices over the 1 m depth cap in the applied direction")
axs[2].set_title("depth steps beyond the cap")
axs[0].legend(fontsize=8)
fig.tight_layout()
FIGS.mkdir(parents=True, exist_ok=True)
fig.savefig(FIGS / "gain_ratio.png", dpi=120)
log(f"wrote {FIGS / 'gain_ratio.png'}")

# depth moves of each fit
mesh = I.mesh_for_passes([load_patch("/home/ola/rss27_sss_bathymetry/cache/rosbag2_2026_07_10-17_39_53_patches.npz", n)
                          for n in ("A", "B")], h=0.5, buffer=10.0)
v = mesh.vertices
passes = [load_patch("/home/ola/rss27_sss_bathymetry/cache/rosbag2_2026_07_10-17_39_53_patches.npz", n) for n in ("A", "B")]
tracks = [p.local_poses(mesh.origin)[:, :2] for p in passes]
near = KDTree(np.concatenate(tracks)).query(v)[0] < 62
cols = min(len(fits), 4)
rows = int(np.ceil(len(fits) / cols))
fig, axs = plt.subplots(rows, cols, figsize=(5.5 * cols, 4.6 * rows), sharex=True, sharey=True, squeeze=False)
for ax, (nm, f) in zip(axs.ravel(), fits.items()):
    dd = f["d"] - f["d0"]
    sc = ax.scatter(v[near, 0], v[near, 1], c=dd[near], s=0.5, cmap="RdBu_r", vmin=-2, vmax=2)
    for t in tracks:
        ax.plot(t[:, 0], t[:, 1], "k", lw=0.6)
    ax.set_aspect("equal")
    ax.set_title(f"{nm}: depth change (m), sd {dd[near].std():.2f}", fontsize=10)
    fig.colorbar(sc, ax=ax, shrink=0.8)
fig.tight_layout()
fig.savefig(FIGS / "depth_moves.png", dpi=110)
log(f"wrote {FIGS / 'depth_moves.png'}")
