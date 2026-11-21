r"""Which vertices want depth steps beyond the cap, from the per-vertex output of `t1_state.py`.

Usage: run.sh t1_vertices.py K STATE

Tabulates the raw Gauss--Newton depth step by the vertex's distance to the nearest track and by the
share of the prior in its GGN diagonal, with each factor's gradient and information, and draws the maps.
"""
import itertools
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
from common import FIGS, OUT, log
from scipy.spatial import KDTree

k, state = int(sys.argv[1]), sys.argv[2]
r = pickle.loads((OUT / f"t1_k{k}_{state}_e1.pkl").read_bytes())
v = r["vertices"]
tracks = np.concatenate([p[:, :2] for p in r["poses"]])
dist = KDTree(tracks).query(v)[0]
raw = r["raw_d"] / r["over"]  # the step after the psi shortening, before the per-vertex clip
diag_data = r["diag_d"] - r["diag_prior"] - r["diag_fr"]
share = r["diag_prior"] / r["diag_d"]
seen = diag_data > 1e-9 * r["diag_prior"]
big = np.abs(raw) > 1.0
log(f"{len(v)} vertices, {seen.sum()} with any intensity information, {np.sum(r['diag_fr'] > 0)} in first-return rows; "
    f"{big.sum()} want > 1 m after the psi shortening ({np.sum(np.abs(r['raw_d']) > 1)} before it)")
log("by distance to the nearest track: vertices, share wanting > 1 m, median |step|, median information "
    "(intensity, first return, prior diagonal), median |gradient| (intensity, first return, prior)")
edges = [0, 1, 2, 4, 8, 15, 25, 40, 60, 200]
rows = []
for lo, hi in itertools.pairwise(edges):
    m = (dist >= lo) & (dist < hi)
    if not m.any():
        continue
    rows.append((lo, hi, int(m.sum()), big[m].mean(), np.median(np.abs(raw[m])), np.median(diag_data[m]),
                 np.mean(r["diag_fr"][m]), np.median(r["diag_prior"][m]), np.median(np.abs(r["g_data"][m])),
                 np.mean(np.abs(r["g_fr"][m])), np.median(np.abs(r["g_prior"][m]))))
    log(f"  {lo:3d}-{hi:3d} m: {rows[-1][2]:6d}  {rows[-1][3]:6.1%}  {rows[-1][4]:7.3f} m   info {rows[-1][5]:9.2e} "
        f"{rows[-1][6]:9.2e} {rows[-1][7]:9.2e}   grad {rows[-1][8]:9.2e} {rows[-1][9]:9.2e} {rows[-1][10]:9.2e}")
log("by the prior's share of the GGN diagonal: vertices, share of all that want > 1 m, median |step|")
for lo, hi in ((0, 0.01), (0.01, 0.1), (0.1, 0.5), (0.5, 0.9), (0.9, 0.999), (0.999, 1.01)):
    m = (share >= lo) & (share < hi)
    log(f"  prior share {lo:5.3f}-{hi:5.3f}: {m.sum():6d}  {big[m].sum() / max(big.sum(), 1):6.1%}  "
        f"{np.median(np.abs(raw[m])) if m.any() else 0:7.3f} m")
# which factor's gradient the big steps follow
gb = {nm: r[key][big] for nm, key in (("intensity", "g_data"), ("first return", "g_fr"), ("prior", "g_prior"))}
tot = sum(gb.values())
log("vertices wanting > 1 m: share of sum |gradient| by factor: "
    + ", ".join(f"{nm} {np.abs(g).sum() / sum(np.abs(x).sum() for x in gb.values()):.1%}" for nm, g in gb.items())
    + f"; step agrees in sign with the total gradient on {np.mean(np.sign(raw[big]) == np.sign(tot)):.0%}")
log(f"  of them: {np.mean(~seen[big]):.1%} have no intensity information; median distance to track {np.median(dist[big]):.1f} m; "
    f"median intensity/prior diagonal {np.median(diag_data[big] / r['diag_prior'][big]):.2e}")
# total information, intensity vs first return
log(f"sum of GGN diagonal over depths: intensity {diag_data.sum():.3e}, first return {r['diag_fr'].sum():.3e}, "
    f"prior {r['diag_prior'].sum():.3e}; over the first-return vertices: intensity "
    f"{diag_data[r['diag_fr'] > 0].sum():.3e}, first return {r['diag_fr'][r['diag_fr'] > 0].sum():.3e}")
m = r["diag_fr"] > 0
log(f"  first-return vertices: median intensity/first-return information {np.median(diag_data[m] / r['diag_fr'][m]):.1f}, "
    f"median |gradient| intensity/first-return {np.median(np.abs(r['g_data'][m]) / np.maximum(np.abs(r['g_fr'][m]), 1e-12)):.1f}")

fig, axs = plt.subplots(2, 3, figsize=(19, 11), sharex=True, sharey=True)
show = dist < 70


def panel(ax, val, title, **kw):
    sc = ax.scatter(v[show, 0], v[show, 1], c=val[show], s=0.6, **kw)
    fig.colorbar(sc, ax=ax, shrink=0.8)
    for p in r["poses"]:
        ax.plot(p[:, 0], p[:, 1], "k", lw=0.6)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=10)


panel(axs[0, 0], np.clip(raw, -3, 3), "depth step after psi shortening, before the clip (m, colour clipped at 3)",
      cmap="RdBu_r", vmin=-3, vmax=3)
panel(axs[0, 1], np.log10(np.maximum(diag_data, 1e-6)), "log10 intensity information (GGN diagonal)", cmap="viridis")
panel(axs[0, 2], np.log10(np.maximum(share, 1e-6)), "log10 prior share of the GGN diagonal", cmap="magma")
panel(axs[1, 0], np.sign(r["g_data"]) * np.log10(1 + np.abs(r["g_data"])), "signed log10(1+|intensity gradient|)",
      cmap="RdBu_r", vmin=-5, vmax=5)
panel(axs[1, 1], np.sign(r["g_fr"]) * np.log10(1 + np.abs(r["g_fr"])), "signed log10(1+|first-return gradient|)",
      cmap="RdBu_r", vmin=-5, vmax=5)
panel(axs[1, 2], np.clip(r["raw_xi"] / r["over"], -2, 2), "xi step after psi shortening (clipped at 2)", cmap="RdBu_r",
      vmin=-2, vmax=2)
fig.suptitle(f"Gauss-Newton step by vertex, k = {k}, state {state}")
fig.tight_layout()
FIGS.mkdir(parents=True, exist_ok=True)
fig.savefig(FIGS / f"vertices_k{k}_{state}.png", dpi=110)
log(f"wrote {FIGS / f'vertices_k{k}_{state}.png'}")
