r"""Diagnostic-only seabed from the PingDSP 3DSS bathymetry.

The method itself never sees the 3DSS data; these helpers exist only for sanity checks
that compare the factors against the most realistic seabed available.
The 3DSS points, moved to the world frame with each ping's pose, are averaged onto the
mesh vertices; vertices without points get the conditional mean of the GMRF prior given
the others.
"""
import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from bbf import mesh as M
from bbf import ping as P
from bbf import prior as PR
from bbf.io import Pass

H_MESH, BUFFER = 0.5, 5.0
RHO, SIGMA, SIGMA_OBS = 3.0, 2.0, 0.05  # fill-in prior: range, std, per-point noise (m)


def bathymetry_world(ps: Pass, every: int = 2) -> np.ndarray:
    """3DSS points of every `every`-th ping in the world frame, (m, 3)."""
    frames = jax.vmap(P.pose_frame)(jnp.asarray(ps.poses))
    s, R = np.asarray(frames[0]), np.asarray(frames[1])
    keep = ps.bathy_ping % every == 0
    k = ps.bathy_ping[keep]
    return s[k] + np.einsum("mij,mj->mi", R[k], ps.bathy_xyz[keep])


def seabed_on_mesh(world: np.ndarray, z_ref: float) -> tuple[M.Mesh, np.ndarray, np.ndarray]:
    """Mesh over the points, depths from the points, and the vertices that had points."""
    lo, hi = world[:, :2].min(0), world[:, :2].max(0)
    mesh = M.build(lo[0], hi[0], lo[1], hi[1], h=H_MESH, buffer=BUFFER, z_ref=z_ref)
    q = mesh.to_local(world[:, :2])
    tri, bary, inside = (np.asarray(a) for a in M.locate(mesh, q))
    vert = mesh.triangles[tri, np.argmax(bary, axis=1)]
    depth = z_ref - world[:, 2]
    # drop points far off a robust plane (water-column detections)
    A = np.column_stack([q, np.ones(len(q))])
    coef = np.linalg.lstsq(A[inside], depth[inside], rcond=None)[0]
    res = depth - A @ coef
    mad = np.median(np.abs(res[inside]))
    ok = inside & (np.abs(res) < 10 * mad + 0.5)
    n = np.bincount(vert[ok], minlength=mesh.n_vertices).astype(float)
    mean = np.bincount(vert[ok], depth[ok], minlength=mesh.n_vertices) / np.maximum(n, 1)
    # conditional mean of the GMRF given the vertex averages
    prior = PR.Prior(mesh, RHO, SIGMA, mu_beta=coef, Sigma_beta=np.eye(3))
    W = sp.diags(n / SIGMA_OBS**2)
    rhs = prior.Lambda @ (prior.X @ coef) + W @ mean
    d = np.asarray(spla.spsolve((prior.Lambda + W).tocsc(), rhs))
    return mesh, d, n > 0
