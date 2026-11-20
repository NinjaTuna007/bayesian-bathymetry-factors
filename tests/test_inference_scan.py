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


@pytest.mark.slow
def test_scan_start_beats_first_echo_plane():
    """With relief under the tracks, per-ping-side planes avoid the tilt a single plane takes on."""
    import test_inference_loop as T

    from bbf import prior as PR
    from bbf.inference import init as I
    from bbf.inference.loop import labels_of
    m = M.build(-50.0, 50.0, -12.0, 12.0, h=1.0, origin=(0.0, 0.0), z_ref=0.0)
    v = m.vertices
    d_true = np.asarray(T.DEPTH - 3.0 * np.exp(-((v[:, 0] - 3) ** 2 + v[:, 1] ** 2) / (2 * 4.0**2)) + 0.03 * v[:, 0])
    th = theta()
    u = P.u_grid(T.N_BINS, T.DR, du=0.05)
    rng = np.random.default_rng(0)
    passes = []
    for x0 in (-6.0, 6.0):
        poses = np.column_stack([np.full(9, x0), np.linspace(-4, 4, 9), np.full(9, T.SONAR_Z), np.zeros((9, 3))])
        passes.append(T.simulate_pass(m, jnp.asarray(d_true), poses, th, u, rng, f"x{x0}"))
    seen = (np.abs(v[:, 1]) <= 4) & (np.abs(v[:, 0]) <= 30)

    def rmse(d):
        return float(np.sqrt(np.mean((d[seen] - d_true[seen]) ** 2)))

    pts = np.concatenate([I.first_echo_points(p, I.first_echo(p, th)) for p in passes])
    plane = I.initial_depths(m, I.plane_fit(pts, m))
    labels = [labels_of(p, th) for p in passes]
    r0 = [I.first_echo(p, th) for p in passes]
    valid = [P.onset_valid(r, lab.shape[-1], th.block * T.DR) for r, lab in zip(r0, labels)]
    d0, _, _, _ = Sc.scan_start(passes, m, th, labels, valid, r0, u,
                                lambda b: PR.Prior(m, 5.0, 2.0, mu_beta=b, Sigma_beta=np.diag([0.01, 0.01, 4.0])),
                                stride=1)
    assert rmse(d0) < 0.6 * rmse(plane)
