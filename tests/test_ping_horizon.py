import dataclasses

import jax.numpy as jnp
import numpy as np
import pytest
import scipy.sparse as sp

from bbf import mesh as M
from bbf import ping as P
from bbf.ping import horizon as Hz

DEPTH, SONAR_Z = 20.0, -5.0
N_BINS, DR = 400, 0.1
U = P.u_grid(N_BINS, DR, du=0.05)
TH = P.Theta(noise=1e-7, L=10.0, eps=0.02, eta=1e-6, zeta=1e-6, ell=1.0, log_kappa_mean=0.0, log_kappa_std=0.3)
NBLK = N_BINS // TH.block


def level(x0, y0=0.0):
    return np.array([x0, y0, SONAR_Z, 0.0, 0.0, 0.0])


@pytest.fixture(scope="module")
def scene():
    """A ridge along y at x = 14, 1 m wide (sd): it shadows the floor behind it on starboard of a sonar at x < 14."""
    m = M.build(-45.0, 45.0, -10.0, 10.0, h=1.0, origin=(0.0, 0.0), z_ref=0.0)
    v = m.vertices

    def ridge(a):
        return DEPTH - a * np.exp(-((v[:, 0] - 14.0) ** 2) / (2 * 1.0**2))

    return m, ridge


def test_scored_blocks_keeps_closed_dark_runs_past_the_first_echo():
    lab = np.zeros((1, 2, 12), bool)
    lab[0, 0, [4, 5, 9]] = True  # lit at 4, 5, 9: dark 6..8 is closed by 9; 10, 11 run out
    lab[0, 1, 3:] = True
    r0 = np.array([[1.6, np.nan]])  # side 0: first echo at 1.6 m, past block 1's centre (block_dr 1), side 1: from its first lit block
    use = Hz.scored_blocks([lab], None, [r0], 1.0, max_gap=3.0)[0]
    np.testing.assert_array_equal(np.flatnonzero(use[0, 0]), [2, 3, 4, 5, 6, 7, 8, 9])
    np.testing.assert_array_equal(np.flatnonzero(use[0, 1]), np.arange(3, 12))
    short = Hz.scored_blocks([lab], None, [r0], 1.0, max_gap=2.0)[0]
    assert not short[0, 0, 6] and short[0, 0, 7]  # block 6 is 3 m from lit seabed
    valid = np.ones_like(lab)
    valid[0, 0, 7] = False
    assert not Hz.scored_blocks([lab], [valid], [r0], 1.0, max_gap=3.0)[0][0, 0, 7]


def test_margin_agrees_with_the_rendered_shadow(scene):
    """Blocks with a clearly positive margin are the dark ones of the render, negative ones lit; flat ground is all open."""
    m, ridge = scene
    hz = Hz.Horizon()
    poses = level(0.0)[None]
    mg, den = (a[0] for a in Hz.margins(m, ridge(4.0), poses, P.STARBOARD, TH, U, DR, NBLK, hz))
    S = np.asarray(P.unit_intensity(m, jnp.asarray(ridge(4.0)), poses[0], P.STARBOARD, TH, U, N_BINS, DR))
    Sb = S.reshape(NBLK, TH.block).mean(1)
    seen = den >= hz.den_min
    shadow, open_ = seen & (mg > 0.2), seen & (mg < -0.2)
    assert shadow.sum() >= 3 and open_.sum() > 10
    assert Sb[shadow].max() < 0.05 * np.median(Sb[open_])
    flat_m, flat_den = (a[0] for a in Hz.margins(m, np.full(m.n_vertices, DEPTH), poses, P.STARBOARD, TH, U, DR, NBLK, hz))
    assert flat_m[flat_den >= hz.den_min].max() < 0
    # a higher ridge casts a longer shadow
    hi = Hz.margins(m, ridge(5.5), poses, P.STARBOARD, TH, U, DR, NBLK, hz)[0][0]
    assert (hi > 0).sum() > (mg > 0).sum()


def _labels_of(m, d, poses, rng):
    out = []
    for x in poses:
        S = np.asarray(P.unit_intensities(m, jnp.asarray(d), x, TH, U, N_BINS, DR))
        y = rng.gamma(TH.L, (S + 1e-7) / TH.L)
        out.append(np.stack([[P.block_labels(y[t, c], TH, c, DR) for c in (0, 1)] for t in range(len(x))]))
    return out


def test_gradient_and_rows_match_finite_differences(scene):
    m, ridge = scene
    rng = np.random.default_rng(0)
    poses = [np.stack([level(-6.0, y) for y in (-2.0, 0.0, 2.0)]), np.stack([level(4.0, y) for y in (-1.0, 1.0)])]
    labels = _labels_of(m, ridge(2.5), poses, rng)
    hz = Hz.Horizon(use=Hz.scored_blocks(labels, None, None, TH.block * DR))
    assert sum(int((h & ~lab).sum()) for h, lab in zip(hz.use, labels)) > 10  # some shadow is scored
    d0 = ridge(1.0) + 0.1 * np.sin(m.vertices[:, 1])
    val, grad, A, W = Hz.horizon_term(m, d0, poses, labels, TH, U, DR, hz)
    assert A.shape[0] == len(W) and np.all(W >= 0)
    assert val < Hz.horizon_term(m, ridge(2.5), poses, labels, TH, U, DR, hz, rows=False)[0]  # the truth scores higher
    near = np.abs(m.vertices[:, 0] - 14.0) < 6
    for _ in range(3):
        e = rng.standard_normal(m.n_vertices) * near
        h = 1e-5
        fp = Hz.horizon_term(m, d0 + h * e, poses, labels, TH, U, DR, hz, rows=False)[0]
        fm = Hz.horizon_term(m, d0 - h * e, poses, labels, TH, U, DR, hz, rows=False)[0]
        assert grad @ e == pytest.approx((fp - fm) / (2 * h), rel=1e-5, abs=1e-6)
    # the rows are the margins' derivatives: A e against the change of the scored margins
    e = rng.standard_normal(m.n_vertices) * near
    h = 1e-5
    full = Hz.Horizon(use=hz.use, row_tol=0.0)
    _, _, A0, _ = Hz.horizon_term(m, d0, poses, labels, TH, U, DR, full)

    def scored_margins(d):
        out = []
        for x, use in zip(poses, hz.use):
            for c in (0, 1):
                mg, den = Hz.margins(m, d, x, c, TH, U, DR, NBLK, hz)
                out.append((mg, use[:, c] & (den >= hz.den_min)))
        return out

    base = scored_margins(d0)
    plus, minus = scored_margins(d0 + h * e), scored_margins(d0 - h * e)
    # rows run pass by pass, side by side, ping by ping; collect the margins in that order
    fd = []
    for (_, sc), (mp, _), (mm, _) in zip(base, plus, minus):
        for t in range(sc.shape[0]):
            fd.append(((mp[t] - mm[t]) / (2 * h))[sc[t]])
    np.testing.assert_allclose(A0 @ e, np.concatenate(fd), rtol=1e-4, atol=1e-7)
    assert isinstance(A, sp.csr_matrix) and A.nnz <= A0.nnz


def test_horizon_replaces_the_dark_block_term(scene):
    r"""With `Problem.horizon` the objective is the ping factors, the prior and the horizon term; :math:`H` gains :math:`A^\top W A`."""
    from bbf import prior as PR
    from bbf.inference import ggn
    from bbf.inference import objective as O
    from bbf.inference.loop import jacobians
    m, ridge = scene
    rng = np.random.default_rng(1)
    poses = [np.stack([level(-6.0, y) for y in (-1.0, 1.0)])]
    d_true = ridge(2.5)
    ys, labels = [], []
    for x in poses:
        S = np.asarray(P.unit_intensities(m, jnp.asarray(d_true), x, TH, U, N_BINS, DR))
        y = rng.gamma(TH.L, (S + 1e-7) / TH.L)
        ys.append(y)
        labels.append(np.stack([[P.block_labels(y[t, c], TH, c, DR) for c in (0, 1)] for t in range(len(x))]))
    prior = PR.Prior(m, 4.0, 1.0, mu_beta=np.array([0.0, 0.0, DEPTH]), Sigma_beta=np.diag([0.01, 0.01, 1.0]))
    pb = O.Problem(mesh=m, prior=prior, theta=TH, poses=poses, y=ys, labels=labels, u=U, dr=DR, chunk=2)
    hz = Hz.Horizon(use=Hz.scored_blocks(labels, None, None, TH.block * DR))
    pbh = dataclasses.replace(pb, horizon=hz)
    d0, beta = ridge(1.5), np.array([0.0, 0.0, DEPTH])
    val, grad_h, A, W = Hz.horizon_term(m, d0, poses, labels, TH, U, DR, hz)
    v1, (g1, _), det1 = O.value_and_grad(pbh, d0, beta)
    assert O.value(pbh, d0, beta) == pytest.approx(v1, rel=1e-10)
    # the ping factors alone, as the plain problem without its dark-block term
    S = det1["S"]
    ping = 0.0
    uses = pb.uses()
    for t in range(len(poses[0])):
        for c in (0, 1):
            ping += float(P.log_ping_side(TH, c, DR, jnp.asarray(ys[0][t, c]), jnp.asarray(S[0][t, c]),
                                          jnp.asarray(uses[0][t, c])))
    assert v1 == pytest.approx(prior.log_density(d0, beta) + ping + val, rel=1e-9)
    J = jacobians(pb, d0)
    H1 = ggn.assemble(pbh, J, det1["S"], det1["u"], hz=det1["horizon"])
    H0 = ggn.assemble(pbh, J, det1["S"], det1["u"])
    n = m.n_vertices
    np.testing.assert_allclose(sp.csr_matrix(H1 - H0)[:n, :n].toarray(), (A.T @ sp.diags(W) @ A).toarray(),
                               atol=1e-9 * abs(H1).max())
    assert np.isfinite(g1).all() and np.abs(grad_h).max() > 0
