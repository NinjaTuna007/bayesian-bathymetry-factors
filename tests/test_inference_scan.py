import jax.numpy as jnp
import numpy as np
import pytest
from scipy.stats import gamma as gamma_dist

from bbf import mesh as M
from bbf import ping as P
from bbf.inference import scan as Sc

SONAR_Z, N_BINS, DR = -5.0, 400, 0.1
U = P.u_grid(N_BINS, DR, du=0.02)


def theta():
    return P.Theta(noise=1e-7, L=0.6, eps=0.02, eta=1e-6, zeta=1e-6, ell=1.0, log_kappa_mean=0.0, log_kappa_std=0.5)


def side_plane_mesh(D, slope, side):
    """A mesh whose seabed is the side's plane z = s_z - D + slope * u for a level ping at the origin heading north."""
    m = M.build(-50.0, 50.0, -10.0, 10.0, h=1.0, origin=(0.0, 0.0), z_ref=0.0)
    x = m.vertices[:, 0]
    u = x if side == 1 else -x  # starboard is east
    z = SONAR_Z - D + slope * u
    return m, jnp.asarray(-z)  # depth below z_ref = 0


@pytest.mark.parametrize("side,D,slope", [(1, 15.0, 0.1), (0, 18.0, -0.2)])
def test_plane_render_matches_mesh_render(side, D, slope):
    m, d = side_plane_mesh(D, slope, side)
    pose = jnp.array([0.0, 0.0, SONAR_Z, 0.0, 0.0, 0.0])
    th = theta()
    S_plane = np.asarray(Sc.plane_unit_intensity(pose, side, th, D, slope, U, N_BINS, DR))
    S_mesh = np.asarray(P.unit_intensity(m, d, pose, side, th, U, N_BINS, DR))
    on = S_mesh > 1e-3 * S_mesh.max()
    np.testing.assert_allclose(S_plane[on], S_mesh[on], rtol=0.01)


def test_scan_recovers_plane():
    rng = np.random.default_rng(0)
    th = theta()
    pose = jnp.array([0.0, 0.0, SONAR_Z, 0.01, 0.03, 0.4])
    for side, D, slope in ((1, 16.0, 0.12), (0, 14.0, -0.15)):
        S = np.asarray(Sc.plane_unit_intensity(pose, side, th, D, slope, U, N_BINS, DR))
        I_bar, _ = (np.asarray(a) for a in P.predicted(jnp.asarray(S), 0.3, th, side, DR))
        y = gamma_dist.rvs(th.L, scale=I_bar / th.L, random_state=rng)
        labels = P.block_labels(y, th, side, DR)
        valid = np.ones_like(labels)
        D_hat, s_hat, _ = Sc.scan_ping_side(pose, side, th, y, labels, valid, U, N_BINS, DR, r0=D - 0.5)
        # depth is pinned by the first echo; one ping-side's slope is only known to about +-0.05
        # (on port, -0.20 scores 0.2 log units above the true -0.15), which averages out over pings
        assert D_hat == pytest.approx(D, abs=0.15)
        assert s_hat == pytest.approx(slope, abs=0.08)
