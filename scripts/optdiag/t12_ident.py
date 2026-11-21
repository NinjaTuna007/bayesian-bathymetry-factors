r"""Joint identifiability of the global modes in the GGN: psi, the means of xi, of the log-kappas and of the depths.

Usage: run.sh t12_ident.py K STATE [TEMPER]

With :math:`H` the augmented GGN at the state and :math:`V` the functionals (means over the vertices the
data reach, means of :math:`u` per pass and side, an across-track tilt of the depths, and the six
:math:`\psi`), prints their marginal standard deviations and correlations from :math:`V^\top H^{-1} V`,
next to the conditional ones (everything else held), and the conditional and marginal standard deviation
of single depths, :math:`\xi` and :math:`u`. TEMPER scales the intensity information (default 1).
"""
import dataclasses
import sys

import numpy as np
import scipy.sparse as sp
from common import log, setup
from scipy.spatial import KDTree
from sksparse.cholmod import cho_factor

from bbf.inference import ggn
from bbf.inference.objective import data_term, value_and_grad

k, state = int(sys.argv[1]), sys.argv[2]
temper = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
s = setup(k, state)
pb = dataclasses.replace(s.problem, mode_grid=49, temper=temper)
lay = ggn.layout(pb)
n, ix, ib, ip, iu = lay["n"], lay["xi"], lay["beta"], lay["psi"], lay["u"]
_, _, det = value_and_grad(pb, s.d, s.beta, data_term(pb), psi=s.psi, xi=s.xi)
J, Jb = s.jac()
H = sp.csc_matrix(ggn.assemble(pb, J, det["S"], det["u"], Jb=Jb, psi=s.psi, fr=det["first_return"]))
diag = H.diagonal()
f = cho_factor(H)
Pd = pb.prior.joint_precision().diagonal()[:n]
seen = diag[:n] > 2 * Pd
v = pb.mesh.vertices
tracks = [p[:, :2] for p in pb.poses]
dist_a = KDTree(tracks[0]).query(v)[0]
log(f"k={k} {state} temper {temper}: {seen.sum()} vertices with intensity information")

cols, names = [], []


def add(name, idx, w=None):
    z = np.zeros(lay["size"])
    z[idx] = (1.0 if w is None else w) / len(idx)
    cols.append(z)
    names.append(name)


obs = np.flatnonzero(seen)
add("mean d", obs)
# across-track tilt of the depths about pass A's track: mean of d weighted by the signed offset
t = tracks[0]
i_near = KDTree(t).query(v[obs])[1]
i2 = np.clip(i_near + 1, 0, len(t) - 1)
i1 = np.clip(i_near - 1, 0, len(t) - 1)
tang = t[i2] - t[i1]
off = (v[obs] - t[i_near])
sign = np.sign(tang[:, 0] * off[:, 1] - tang[:, 1] * off[:, 0])
x = sign * dist_a[obs]
add("tilt d (m per 10 m across A)", obs, w=10 * x / np.mean(x**2))
add("mean xi", ix + obs)
at = lay["u_pass"]
for p, nm in enumerate("AB"):
    m = 2 * len(pb.poses[p])
    for c, sd in enumerate(("port", "stbd")):
        add(f"mean u {nm} {sd}", at[p] + np.arange(c, m, 2))
unit = [np.degrees(1), np.degrees(1), 1.0] * 2
for j, nm in enumerate(["port centre (deg)", "port width (deg)", "port g1 (dB/m)", "stbd centre (deg)", "stbd width (deg)",
                        "stbd g1 (dB/m)"]):
    z = np.zeros(lay["size"])
    z[ip + j] = unit[j]
    cols.append(z)
    names.append(nm)
V = np.stack(cols, axis=1)
C = V.T @ f.solve(V)
sd_m = np.sqrt(np.diag(C))
sd_c = 1.0 / np.sqrt(np.einsum("ij,ij->j", V, H @ V) / np.einsum("ij,ij->j", V, V) ** 2)
R = C / np.outer(sd_m, sd_m)
log("functional: sd with everything else held, marginal sd, inflation")
for j, nm in enumerate(names):
    log(f"  {nm:30s} {sd_c[j]:10.4g} {sd_m[j]:10.4g}  x{sd_m[j] / sd_c[j]:8.1f}")
log("correlations of the functionals in the GGN posterior (rows as above):")
for j in range(len(names)):
    log("  " + " ".join(f"{R[j, i]:+5.2f}" for i in range(len(names))))
w, U = np.linalg.eigh(R)
log(f"eigenvalues of the correlation matrix: {np.round(w, 4)}")
log("its weakest direction (in marginal sd units): " + ", ".join(f"{nm.split(' (')[0]} {U[j, 0]:+.2f}" for j, nm in enumerate(names)))

rng = np.random.default_rng(0)
pick = rng.choice(obs, 40, replace=False)
for label, base in (("depth (m)", 0), ("xi", ix)):
    E = np.zeros((lay["size"], len(pick)))
    E[base + pick, np.arange(len(pick))] = 1.0
    var = np.einsum("ij,ij->j", E, f.solve(E))
    log(f"single {label} at 40 vertices the data reach: median sd held {np.median(1 / np.sqrt(diag[base + pick])):.3f}, "
        f"marginal {np.median(np.sqrt(var)):.3f} (10-90%: {np.percentile(np.sqrt(var), 10):.3f}-{np.percentile(np.sqrt(var), 90):.3f})")
pu = rng.choice(np.arange(iu, ip), 40, replace=False)
E = np.zeros((lay["size"], 40))
E[pu, np.arange(40)] = 1.0
var = np.einsum("ij,ij->j", E, f.solve(E))
log(f"single u at 40 ping-sides: median sd held {np.median(1 / np.sqrt(diag[pu])):.4f}, marginal {np.median(np.sqrt(var)):.4f}")
