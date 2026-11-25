r"""Shared by the residual diagnostics: the per-bin table of `scripts/residual_table.py` as flat arrays.

`load` keeps the bins the ping factor scores that have a seabed echo (:math:`S_0 > 0`) and adds the
covariates the diagnostics group by. The gain :math:`g_c(r_b)` of each bin is rebuilt from the
table's theta, so the predicted mean can be re-formed as
:math:`\bar I = g\,(e^{u} S + N)` under other radiometric laws.
"""
import dataclasses
from dataclasses import dataclass

import numpy as np
from scipy.special import gammainc

DB_TO_LOG_I = np.log(10) / 10
SIDES = ("port", "stbd")


@dataclass(frozen=True)
class Bins:
    r"""Flat arrays over the kept bins (n,), and the table's scalars and per-side parameters in `par`."""
    y: np.ndarray
    I_bar: np.ndarray
    I_sp: np.ndarray
    S: np.ndarray        # unit intensity with the field
    S0: np.ndarray       # and without
    r: np.ndarray        # mean range of the bin's ground (m)
    beta: np.ndarray     # beam angle (rad, below the outboard axis)
    cos: np.ndarray      # incidence cosine
    cos_flat: np.ndarray
    q: np.ndarray        # footprint (n, 2), mesh-local
    xi: np.ndarray
    slope_out: np.ndarray
    slope_fwd: np.ndarray
    vis: np.ndarray      # visible share of the bin's curve, :math:`A / A_1`
    side: np.ndarray
    pass_id: np.ndarray
    ping: np.ndarray     # row of the table: ping index over both passes
    b: np.ndarray        # bin index
    log_g: np.ndarray    # log gain
    offset: np.ndarray   # log transmit power
    u_hat: np.ndarray
    shadow_dist: np.ndarray  # bins to the nearest dark or shadowed bin beyond the first echo
    alt: np.ndarray      # first-echo range of the ping (m)
    yaw_rate: np.ndarray  # rad/s
    depth: np.ndarray    # depth of the bin's ground below the reference height (m)
    par: dict

    @property
    def n(self) -> int:
        return len(self.y)

    @property
    def ps(self) -> np.ndarray:
        """Ping-side index, 2 * ping + side."""
        return 2 * self.ping + self.side

    def resid(self) -> np.ndarray:
        r""":math:`\log(y / \bar I)`."""
        return np.log(self.y / self.I_bar)

    def pit(self) -> np.ndarray:
        r"""Probability integral transform of :math:`y` under the bin mixture at the fitted state."""
        L, eps = float(self.par["L"]), np.asarray(self.par["eps"])[self.side]
        return (1 - eps) * gammainc(L, L * self.y / self.I_bar) + eps * (1 - np.exp(-self.y / self.I_sp))


def _nearest_true(mask: np.ndarray) -> np.ndarray:
    """Distance (in elements, along the last axis) from each element to the nearest True; large where a row has none."""
    n = mask.shape[-1]
    idx = np.arange(n)
    big = 10 * n
    left = np.maximum.accumulate(np.where(mask, idx, -big), axis=-1)
    right = np.minimum.accumulate(np.where(mask, idx, big)[..., ::-1], axis=-1)[..., ::-1]
    return np.minimum(idx - left, right - idx)


def load(path, keep_all: bool = False) -> Bins:
    """The table at `path` as `Bins`; keep_all also keeps scored bins without a predicted echo."""
    z = np.load(path)
    f = {k[2:]: z[k] for k in z.files if k.startswith("f_")}
    n_ping, _, n_bins = z["y"].shape
    r_bin = z["r_bin"]
    use = z["use"] & (z["y"] > 0)
    if not keep_all:
        use &= (f["S0"] > 0) & np.isfinite(f["cos"])
    r0 = z["r0"]
    beyond = r_bin[None, None, :] > np.where(np.isfinite(r0), r0, 0.0)[:, :, None]
    with np.errstate(invalid="ignore", divide="ignore"):
        vis = np.where(f["A1"] > 0, f["A"] / f["A1"], 0.0)
    dark = beyond & (~z["lit"][..., :n_bins] | (vis < 0.5))
    sdist = _nearest_true(dark)
    t = z["t"]
    yaw = np.unwrap(z["poses"][:, 5])
    yaw_rate = np.gradient(yaw) / np.maximum(np.gradient(t), 1e-3)
    pid = z["pass_id"]
    for p in np.unique(pid):  # no rate across the gap between passes
        m = np.flatnonzero(pid == p)
        yaw_rate[m[[0, -1]]] = yaw_rate[m[[1, -2]]]
    ti, ci, bi = np.nonzero(use)
    gain = z["gain_db"]
    rb = r_bin[bi]
    log_g = DB_TO_LOG_I * (gain[ci, 0] + gain[ci, 1] * rb + gain[ci, 2] * np.log10(rb))
    par = {k: z[k] for k in ("dr", "k", "L", "noise", "eps", "eta", "zeta", "ell", "gain_db", "beta0", "omega", "alpha",
                             "log_kappa_mean", "log_kappa_std", "r_bin", "poses", "t", "r0", "pass_id", "z_ref")}
    par["shape"] = (n_ping, 2, n_bins)
    par["noise_slope"] = float(z["noise_slope"]) if "noise_slope" in z.files else 0.0
    par["noise_power"] = float(z["noise_power"]) if "noise_power" in z.files else 0.0
    par["psi_file"] = z["psi_file"] if "psi_file" in z.files else np.full((2, 3), np.nan)

    def g(name):
        return f[name][ti, ci, bi].astype(float)

    return Bins(y=z["y"][ti, ci, bi].astype(float), I_bar=z["I_bar"][ti, ci, bi].astype(float),
                I_sp=z["I_sp"][ti, ci, bi].astype(float), S=g("S"), S0=g("S0"), r=g("r"), beta=g("beta"), cos=g("cos"),
                cos_flat=g("cos_flat"), q=np.column_stack([g("qx"), g("qy")]), xi=g("xi"), slope_out=g("slope_out"),
                slope_fwd=g("slope_fwd"), vis=vis[ti, ci, bi], side=ci, pass_id=pid[ti], ping=ti, b=bi, log_g=log_g,
                offset=z["offset"][ti], u_hat=z["u_hat"][ti, ci], shadow_dist=sdist[ti, ci, bi],
                alt=np.nanmin(np.where(np.isfinite(r0), r0, np.inf), axis=1)[ti], yaw_rate=yaw_rate[ti], depth=g("depth"),
                par=par)


def subset(b: Bins, mask: np.ndarray) -> Bins:
    """The bins in `mask`."""
    return dataclasses.replace(b, **{f.name: getattr(b, f.name)[mask] for f in dataclasses.fields(b) if f.name != "par"})


def cells(q: np.ndarray, size: float) -> tuple[np.ndarray, np.ndarray]:
    """Square ground cells of side `size` (m): (index of each point's cell (n,), cell centres (m, 2))."""
    key, inv = np.unique(np.floor(q / size).astype(np.int64), axis=0, return_inverse=True)
    return inv.ravel(), (key + 0.5) * size
