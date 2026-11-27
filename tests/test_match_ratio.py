import dataclasses

import jax.numpy as jnp
import numpy as np
import pytest
import scipy.sparse as sp

from bbf import match as Mt
from bbf import mesh as M
from bbf import ping as P
from bbf import prior as PR
from bbf.inference import objective as O
from bbf.match.ratio import (
    RatioAssignment,
    RatioMatch,
    assign,
    fisher_rows,
    log_ratio_match,
)


def _two_cells():
    """Two cells: cell 0 seen by visits 0, 1, 2 (pass 0 twice, pass 1); cell 1 by visits 3, 4."""
    unit = [np.array([[[0, 0, 3, -1]], [[1, 1, 4, -1]]]), np.array([[[2, 2, -1, -1]]])]
    Y = np.array([3.0, 5.0, 2.0, 7.0, 1.5])
    return RatioAssignment(unit=unit, unit_cell=np.array([0, 0, 0, 1, 1]), n_cells=2, Y=Y, L=1.3)


def test_reflectivity_cancels_within_a_cell():
    a = _two_cells()
    rng = np.random.default_rng(0)
    Ib = [rng.uniform(0.5, 2.0, size=u.shape) for u in a.unit]
    base = float(log_ratio_match([jnp.asarray(x) for x in Ib], a))
    # scale every bin of cell 0 (visits 0, 1, 2) by one factor, cell 1 by another
    scaled = []
    for x, u in zip(Ib, a.unit):
        f = np.where(np.isin(u, [0, 1, 2]), 7.0, np.where(np.isin(u, [3, 4]), 0.2, 1.0))
        scaled.append(jnp.asarray(x * f))
    assert float(log_ratio_match(scaled, a)) == pytest.approx(base, rel=1e-12)
    # changing one visit against the others does change it
    other = [jnp.asarray(np.where(u == 1, x * 3.0, x)) for x, u in zip(Ib, a.unit)]
    assert abs(float(log_ratio_match(other, a)) - base) > 1e-3


def test_two_visits_is_the_log_ratio_density_up_to_the_data():
    """For one cell with two visits the term moves with the predicted ratio exactly as log p(e | k_a, k_b)."""
    a = RatioAssignment(unit=[np.array([[[0, 0, 1]]])], unit_cell=np.array([0, 0]), n_cells=1,
                        Y=np.array([2.0, 5.0]), L=0.9)
    vals, refs = [], []
    for ratio in (0.3, 1.0, 2.5, 9.0):
        Ib = np.array([[[1.0, 1.0, ratio]]])
        vals.append(float(log_ratio_match([jnp.asarray(Ib)], a)))
        ka, kb = 0.9 * 2.0**2 / 2.0, 0.9  # shapes of the two sums: two equal bins, one bin
        e = np.log(5.0 / ratio) - np.log(2.0 / 2.0)
        refs.append(float(Mt.log_ratio_pdf(e, ka, kb)))
    np.testing.assert_allclose(np.diff(vals), np.diff(refs), rtol=1e-10)


def test_assign_splits_visits_by_pass_and_gap():
    """One cell: pass 0 at t = 0, 1 and again at 20 s (a second visit), pass 1 once; a second cell seen once."""
    n, B = 4, 3
    q0 = np.full((n, 2, B, 2), np.nan)
    q0[[0, 1, 3], 1, 0] = [0.5, 0.5]  # cell (0, 0) at pings 0, 1 and 3
    q0[2, 1, 1] = [5.5, 5.5]  # cell (5, 5), only here
    q1 = np.full((2, 2, B, 2), np.nan)
    q1[0, 0, 2] = [0.2, 0.7]  # cell (0, 0) from pass 1
    t0, t1 = np.array([0.0, 1.0, 2.0, 20.0]), np.array([100.0, 101.0])
    y0, y1 = np.ones((n, 2, B)), np.full((2, 2, B), 2.0)
    use0, use1 = np.ones((n, 2, B), bool), np.ones((2, 2, B), bool)
    a = assign([q0, q1], [t0, t1], [y0, y1], [use0, use1], 1.0, RatioMatch(cell=1.0, gap=5.0))
    assert a.n_cells == 1 and a.n_units == 3  # the lone cell is not matched
    u0 = a.unit[0]
    assert u0[0, 1, 0] == u0[1, 1, 0] != u0[3, 1, 0]  # 20 s later: a second visit
    assert u0[2, 1, 1] == -1 and a.unit[1][0, 0, 2] >= 0
    np.testing.assert_allclose(np.sort(a.Y), [1.0, 2.0, 2.0])
    assert sum(int(m.sum()) for m in a.matched()) == 4


def test_fisher_rows_give_the_cell_fisher_information():
    """G^T (A^T A - B^T B) G equals sum over cells of J^T (diag k - k k^T / K) J in eta = log I."""
    a = _two_cells()
    rng = np.random.default_rng(1)
    Ib = [rng.uniform(0.5, 2.0, size=u.shape) for u in a.unit]
    dI = [rng.uniform(0.1, 1.0, size=u.shape) for u in a.unit]
    A, B = fisher_rows(Ib, a)
    D = sp.diags(np.concatenate([x.ravel() for x in dI]))
    H = ((A @ D).T @ (A @ D) - (B @ D).T @ (B @ D)).toarray()
    unit = np.concatenate([u.ravel() for u in a.unit])
    I, d = np.concatenate([x.ravel() for x in Ib]), np.concatenate([x.ravel() for x in dI])
    on = unit >= 0
    s1, s2 = np.bincount(unit[on], I[on], 5), np.bincount(unit[on], I[on] ** 2, 5)
    k = a.L * s1**2 / s2
    Jeta = np.zeros((5, len(unit)))
    Jeta[unit[on], np.flatnonzero(on)] = d[on] / s1[unit[on]]
    expect = np.zeros_like(H)
    for c, vis in ((0, [0, 1, 2]), (1, [3, 4])):
        kk = k[vis]
        F = np.diag(kk) - np.outer(kk, kk) / kk.sum()
        expect += Jeta[vis].T @ F @ Jeta[vis]
    np.testing.assert_allclose(H, expect, atol=1e-12)
    assert np.linalg.eigvalsh(H).min() > -1e-10


DEPTH, SONAR_Z, N_BINS, DR = 20.0, -5.0, 300, 0.1


@pytest.fixture(scope="module")
def problem():
    """Two passes over a ridge, with a reflectivity that varies over the ground, and the ratio match."""
    m = M.build(-40.0, 40.0, -10.0, 10.0, h=1.0, origin=(0.0, 0.0), z_ref=0.0)
    v = m.vertices
    d_true = DEPTH - 2.0 * np.exp(-((v[:, 0] - 6.0) ** 2) / (2 * 2.0**2))
    xi_true = 0.5 * np.sin(v[:, 0] / 3.0)
    th = P.Theta(noise=1e-7, L=5.0, eps=0.02, eta=1e-6, zeta=1e-6, ell=1.0, log_kappa_mean=0.0, log_kappa_std=0.3)
    u = P.u_grid(N_BINS, DR, du=0.05)
    rng = np.random.default_rng(2)
    poses, ys, labels, times = [], [], [], []
    for x0 in (-5.0, 9.0):
        pp = np.column_stack([np.full(5, x0), np.linspace(-2, 2, 5), np.full(5, SONAR_Z), np.zeros((5, 3))])
        S = np.asarray(P.unit_intensities(m, jnp.asarray(d_true), pp, th, u, N_BINS, DR, jnp.asarray(xi_true)))
        y = rng.gamma(th.L, (S + 1e-7) / th.L)
        poses.append(pp)
        ys.append(y)
        times.append(np.arange(5) * 0.1 + 100 * len(times))
        labels.append(np.stack([[P.block_labels(y[t, c], th, c, DR) for c in (0, 1)] for t in range(5)]))
    prior = PR.Prior(m, 5.0, 2.0, mu_beta=np.array([0.0, 0.0, DEPTH]), Sigma_beta=np.diag([0.01, 0.01, 25.0]))
    pb = O.Problem(mesh=m, prior=prior, theta=th, poses=poses, y=ys, labels=labels, u=u, dr=DR, chunk=5)
    d0 = d_true + 0.3 * np.sin(v[:, 1] / 3.0)
    fp = [Mt.footprints(m, jnp.asarray(d0), pp, th, u, N_BINS, DR) for pp in poses]
    ra = assign(fp, times, ys, pb.uses_all(), th.L, RatioMatch(cell=1.0))
    return dataclasses.replace(pb, ratio=ra), d0, np.array([0.0, 0.0, DEPTH])


def test_matched_bins_leave_the_ping_factor(problem):
    pb, _, _ = problem
    assert pb.ratio is not None and pb.ratio.n_cells > 20
    for u, ua, mt in zip(pb.uses(), pb.uses_all(), pb.ratio.matched()):
        assert not (u & mt).any() and np.array_equal(u | (ua & mt), ua)


def test_value_gradient_and_ggn_with_the_ratio_match(problem):
    from bbf.inference import ggn
    from bbf.inference.loop import jacobians
    pb, d0, beta = problem
    val, (gd, _), det = O.value_and_grad(pb, d0, beta)
    assert O.value(pb, d0, beta) == pytest.approx(val, rel=1e-10)
    rng = np.random.default_rng(3)
    for _ in range(2):
        e = rng.standard_normal(pb.mesh.n_vertices) * (np.abs(pb.mesh.vertices[:, 0]) < 20)
        h = 1e-5
        fd = (O.value(pb, d0 + h * e, beta) - O.value(pb, d0 - h * e, beta)) / (2 * h)
        assert gd @ e == pytest.approx(fd, rel=1e-4)
    J = jacobians(pb, d0)
    H1 = ggn.assemble(pb, J, det["S"], det["u"], u_ratio=det["u_ratio"])
    H0 = ggn.assemble(dataclasses.replace(pb, ratio=None), J, det["S"], det["u"])
    n = pb.mesh.n_vertices
    with pytest.raises(ValueError):
        ggn.assemble(pb, J, det["S"], det["u"])
    assert abs(sp.csr_matrix(H1 - H1.T)).max() < 1e-9 * abs(H1).max()
    assert abs(sp.csr_matrix(H1 - H0)[:n, :n]).sum() > 0


def test_held_gains_give_a_consistent_gradient_and_curvature(problem):
    """With the gains held (`ratio_gains`), the gradient matches finite differences of the same objective."""
    from bbf.inference import ggn
    from bbf.inference.loop import jacobians
    pb, d0, beta = problem
    held = dataclasses.replace(pb, ratio_gain=O.ratio_gains(pb, O.render(pb, d0)))
    _, (gd, _), det = O.value_and_grad(held, d0, beta)
    for p, g in enumerate(held.ratio_gain):
        np.testing.assert_allclose(det["u_ratio"][p], g)  # the gains are the held ones
    rng = np.random.default_rng(4)
    for _ in range(2):
        e = rng.standard_normal(pb.mesh.n_vertices) * (np.abs(pb.mesh.vertices[:, 0]) < 20)
        h = 1e-5
        fd = (O.value(held, d0 + h * e, beta) - O.value(held, d0 - h * e, beta)) / (2 * h)
        assert gd @ e == pytest.approx(fd, rel=1e-4)
    H = ggn.assemble(held, jacobians(held, d0), det["S"], det["u"], u_ratio=det["u_ratio"])
    assert abs(sp.csr_matrix(H - H.T)).max() < 1e-9 * abs(H).max()


def _two_visits(mismatch, spread, L=13.0, nb=20, model_sd=0.0, hold=True):
    """One cell, two visits of nb bins predicted equal in total; the second measured brighter by exp(mismatch)."""
    from bbf.match.ratio import visit_shapes
    unit = np.concatenate([np.zeros(nb, int), np.ones(nb, int)]).reshape(1, 2, nb)
    a = RatioAssignment([unit], np.array([0, 0]), 1, np.array([nb * 1.0, nb * np.exp(mismatch)]), L)
    flat = np.ones((1, 2, nb))
    prof = np.exp(np.linspace(0, spread, nb))
    Ib = np.stack([prof / prof.mean()] * 2).reshape(1, 2, nb)
    if hold:
        a = dataclasses.replace(a, shape=visit_shapes([flat], a, model_sd))
    return float(log_ratio_match([jnp.asarray(Ib)], a))


def test_held_shapes_do_not_reward_uneven_predictions():
    # followed, the shapes fall with the spread and forgive the mismatch; held, the spread changes nothing
    assert _two_visits(1.0, 10.0, hold=False) > _two_visits(1.0, 0.0, hold=False) + 40
    assert _two_visits(1.0, 10.0) == pytest.approx(_two_visits(1.0, 0.0), abs=1e-9)


def test_model_error_caps_the_shape_and_the_cost_of_a_mismatch():
    from bbf.match.ratio import visit_shapes
    unit = np.zeros((1, 1, 50), int)
    a = RatioAssignment([unit], np.array([0]), 1, np.array([50.0]), 13.0)
    k = visit_shapes([np.ones((1, 1, 50))], a, model_sd=0.5)
    assert k[0] == pytest.approx(650 / (1 + 650 * 0.25)) and k[0] < 4.0
    cost = lambda s: _two_visits(0.0, 0.0, model_sd=s) - _two_visits(1.0, 0.0, model_sd=s)
    assert cost(0.0) > 50  # speckle alone: a factor of e between two visits is far out
    assert 0.5 < cost(0.5) < 2.0  # with the model error it is about one standard deviation of the difference (sqrt 2 x 0.5)


# --- the ratio match in this objective: background law, transmit power, temper; the solver and the loop

def _target_problem(problem, temper=0.5):
    """The fixture's problem with the background law in range and transmit power, per-ping powers, temper, and held gains and shapes."""
    from bbf.inference.loop import with_ratio
    pb, d0, beta = problem
    th = dataclasses.replace(pb.theta, noise=1e-3, noise_slope=1.5, noise_power=0.8)  # a background the bins see
    offsets = [np.linspace(-0.3, 0.2, len(p)) for p in pb.poses]
    base = dataclasses.replace(pb, theta=th, kappa_offset=offsets, temper=temper, ratio=None)
    times = [np.arange(len(p)) * 0.1 + 100 * i for i, p in enumerate(pb.poses)]
    return with_ratio(base, d0, times, RatioMatch(cell=1.0, model_sd=0.3), log=lambda _: None), d0, beta


def test_gradient_with_the_ratio_match_matches_finite_differences_per_vertex(problem):
    """d ell / d d_i of the full objective, at single vertices the matched cells see, against central differences."""
    pb, d0, beta = _target_problem(problem)
    assert pb.ratio is not None and pb.ratio.shape is not None and pb.ratio_gain is not None
    assert pb.ratio.n_cells > 20
    # the ratio term is in the value, tempered: without it the value changes
    v_ratio = O.value(pb, d0, beta)
    v_plain = O.value(dataclasses.replace(pb, ratio=None, ratio_gain=None), d0, beta)
    assert abs(v_ratio - v_plain) > 1.0
    _, (gd, _), det = O.value_and_grad(pb, d0, beta)
    for p, g in enumerate(pb.ratio_gain):
        np.testing.assert_array_equal(det["u_ratio"][p], g)
    # vertices with the largest data gradient: where the bins are
    data = np.abs(det["grad_d_data"])
    for i in np.argsort(data)[-4:]:
        e = np.zeros(pb.mesh.n_vertices)
        e[i] = 1.0
        h = 1e-5
        fd = (O.value(pb, d0 + h * e, beta) - O.value(pb, d0 - h * e, beta)) / (2 * h)
        assert gd[i] == pytest.approx(fd, rel=1e-4, abs=1e-6 * data.max())


def _without_ratio_term(pb):
    """pb with the matched bins still out of the ping factor but no ratio term: an assignment with no visits to score."""
    ra = dataclasses.replace(pb.ratio, unit_cell=np.zeros(0, int), n_cells=0, Y=np.zeros(0), shape=np.zeros(0))
    return dataclasses.replace(pb, ratio=ra)


def test_the_ratio_term_is_tempered(problem):
    """The ratio term's share of the value is temper times the untempered one."""
    pb1, d0, beta = _target_problem(problem, temper=1.0)
    pbh = dataclasses.replace(pb1, temper=0.5)
    share = [O.value(pb, d0, beta) - O.value(_without_ratio_term(pb), d0, beta) for pb in (pb1, pbh)]
    assert abs(share[0]) > 1.0
    assert share[1] == pytest.approx(0.5 * share[0], rel=1e-9)


def test_ggn_with_the_ratio_rows_is_positive_definite(problem):
    """The undamped GGN with the ratio rows factorises; the ratio rows alone are symmetric positive semidefinite."""
    from sksparse.cholmod import cho_factor

    from bbf.inference import ggn
    from bbf.inference.loop import jacobians
    pb, d0, beta = _target_problem(problem)
    _, _, det = O.value_and_grad(pb, d0, beta)
    J = jacobians(pb, d0)
    H = ggn.assemble(pb, J, det["S"], det["u"], fr=det["first_return"], hz=det["horizon"], u_ratio=det["u_ratio"])
    assert abs(sp.csr_matrix(H - H.T)).max() < 1e-9 * abs(H).max()
    cho_factor(sp.csc_matrix(H))  # raises if not positive definite
    # the ratio rows: the difference to the same problem without the ratio term (same ping-factor bins)
    H0 = ggn.assemble(_without_ratio_term(pb), J, det["S"], det["u"], fr=det["first_return"], hz=det["horizon"],
                      u_ratio=det["u_ratio"])
    Hr = sp.csr_matrix(H - H0)
    n = pb.mesh.n_vertices
    assert abs(Hr[n:, :]).sum() < 1e-9 * abs(Hr).sum()  # no entries outside the depths (gains held, psi fixed)
    ev = np.linalg.eigvalsh(Hr[:n, :n].toarray())
    assert ev.max() > 0 and ev.min() > -1e-8 * ev.max()


@pytest.mark.slow
def test_infer_with_the_ratio_match_ascends_as_predicted():
    """A short outer loop with the ratio match on two passes over a bump with patchy reflectivity: every accepted
    step ascends, and the gain the quadratic model predicts is of the order of the actual gain."""
    from bbf.inference.loop import infer
    from bbf.io import Pass
    m = M.build(-50.0, 50.0, -12.0, 12.0, h=1.0, origin=(0.0, 0.0), z_ref=0.0)
    v = m.vertices
    d_true = np.asarray(20.0 - 2.0 * np.exp(-((v[:, 0] - 2) ** 2 + v[:, 1] ** 2) / (2 * 4.0**2)))
    xi_true = jnp.asarray(0.5 * np.sin(v[:, 0] / 3.0 + 1.0) * np.cos(v[:, 1] / 5.0))
    th = P.Theta(noise=1e-7, L=5.0, eps=0.02, eta=1e-6, zeta=1e-6, ell=1.0, log_kappa_mean=0.0, log_kappa_std=0.5)
    n_bins, dr = 400, 0.1
    u = P.u_grid(n_bins, dr, du=0.05)
    rng = np.random.default_rng(5)
    passes = []
    for i, x0 in enumerate((-6.0, 6.0)):
        poses = np.column_stack([np.full(9, x0), np.linspace(-4, 4, 9), np.full(9, -5.0), np.zeros((9, 3))])
        S = np.asarray(P.unit_intensities(m, jnp.asarray(d_true), poses, th, u, n_bins, dr, xi_true))
        y = rng.gamma(th.L, (S + 1e-7) / th.L)
        passes.append(Pass(name=f"x{x0}", poses=poses, y=y, dr=dr, t=np.arange(9) * 0.1 + 100.0 * i))
    beta = np.array([0.0, 0.0, 20.0])
    prior = PR.Prior(m, 5.0, 2.0, mu_beta=beta, Sigma_beta=np.diag([0.01, 0.01, 4.0]))
    lines = []
    with pytest.raises(ValueError):
        infer(passes, m, prior, th, np.full(m.n_vertices, 20.0), beta, u, rounds=1, inner=1, calib_every=3, chunk=9,
              refl=Mt.Reflectivity(m, 4.0, 0.7), ratio=RatioMatch(), log=lambda _: None)
    out = infer(passes, m, prior, th, np.full(m.n_vertices, 20.0), beta, u, rounds=2, inner=4, calib_every=3, chunk=9,
                ratio=RatioMatch(cell=1.0, model_sd=0.3), log=lines.append)
    assert sum("ratio match:" in s for s in lines) == 2 and sum("shapes held" in s for s in lines) == 2
    pb = out["problem"]
    assert pb.ratio is not None and pb.ratio.n_cells > 20 and pb.ratio.shape is not None and pb.ratio_gain is not None
    steps = [h for r in out["history"] for h in r["gn"] if "rho" in h]
    assert len(steps) >= 2
    for r in out["history"]:
        assert np.all(np.diff([h["ell"] for h in r["gn"]]) > 0)
    # the model's gain is of the order of the actual: on this problem the ratios were 0.06 to 1.3, median 0.85 (the
    # 0.06 at round 0's second step, where the plain problem has 0.20: the likelihood is rough at bin scale);
    # never far above one, which would be curvature the objective does not have
    rho = np.array([h["rho"] for h in steps])
    assert np.all((rho > 0.02) & (rho < 3.0)), rho
    assert 0.5 < np.median(rho) < 2.0, rho


def _brute_assign(fp, times, ys, uses, key_of, gap, min_visits):
    """Reference: the visits (sets of (pass, flat bin)) of matched cells, by a dict over cell keys."""
    recs = {}
    for p, (q, t, y, use) in enumerate(zip(fp, times, ys, uses)):
        n, _, B = use.shape
        for i in range(n * 2 * B):
            ti, c, b = np.unravel_index(i, use.shape)
            qi = q[ti, c, b]
            if not use[ti, c, b] or y[ti, c, b] <= 0 or not np.isfinite(qi).all():
                continue
            k = key_of(qi)
            if k is not None:
                recs.setdefault(k, []).append((p, t[ti], i))
    out = set()
    for k, rs in recs.items():
        rs.sort(key=lambda r: (r[0], r[1]))
        visits, cur = [], [rs[0]]
        for prev, r in zip(rs[:-1], rs[1:]):
            if r[0] != prev[0] or r[1] - prev[1] > gap:
                visits.append(cur)
                cur = []
            cur.append(r)
        visits.append(cur)
        if len(visits) >= min_visits:
            out |= {frozenset((p, i) for p, _, i in v) for v in visits}
    return out


def _visits_of(a):
    """The visits of an assignment as sets of (pass, flat bin)."""
    vis = {}
    for p, u in enumerate(a.unit):
        for i in np.flatnonzero(u.ravel() >= 0):
            vis.setdefault(int(u.ravel()[i]), set()).add((p, int(i)))
    return {frozenset(v) for v in vis.values()}


def _random_bins(rng, extent=4.0):
    """Two passes of random footprints over [-extent, extent]^2 (some NaN), times with a gap in pass 0."""
    fp, times, ys, uses = [], [], [], []
    for p, n in enumerate((12, 7)):
        q = rng.uniform(-extent, extent, size=(n, 2, 6, 2))
        q[rng.random((n, 2, 6)) < 0.1] = np.nan
        fp.append(q)
        t = np.arange(n) * 0.5 + 100.0 * p
        t[n // 2:] += 30.0 * (p == 0)  # pass 0 comes back
        times.append(t)
        ys.append(rng.uniform(0.1, 2.0, size=(n, 2, 6)))
        uses.append(rng.random((n, 2, 6)) < 0.9)
    return fp, times, ys, uses


def test_square_cells_are_the_default_and_unchanged():
    rng = np.random.default_rng(5)
    fp, times, ys, uses = _random_bins(rng)
    rm = RatioMatch(cell=1.5, gap=5.0)
    assert rm.cells == "square"
    a = assign(fp, times, ys, uses, 1.0, rm)
    ref = _brute_assign(fp, times, ys, uses, lambda q: tuple(np.floor(q / 1.5).astype(int)), 5.0, 2)
    assert a.n_units > 10 and _visits_of(a) == ref
    b = assign(fp, times, ys, uses, 1.0, RatioMatch(cell=1.5, gap=5.0, cells="square"))
    assert all(np.array_equal(u, v) for u, v in zip(a.unit, b.unit))
    np.testing.assert_array_equal(a.Y, b.Y)


def test_ring_cells_group_bins_by_the_patch_of_their_face():
    m = M.build(-3.0, 3.0, -3.0, 3.0, h=1.0, origin=(0.0, 0.0))
    patch = M.ring_patches(m)
    rng = np.random.default_rng(6)
    fp, times, ys, uses = _random_bins(rng, extent=6.0)  # partly off the mesh
    rm = RatioMatch(gap=5.0, cells="ring")
    with pytest.raises(ValueError):
        assign(fp, times, ys, uses, 1.0, rm)  # ring cells need the mesh
    with pytest.raises(ValueError):
        RatioMatch(cells="hex")
    a = assign(fp, times, ys, uses, 1.0, rm, m)

    def key_of(q):
        tri, _, inside = M.locate(m, q)
        return int(patch[int(tri)]) if bool(inside) else None

    ref = _brute_assign(fp, times, ys, uses, key_of, 5.0, 2)
    assert a.n_units > 10 and _visits_of(a) == ref
    # off the mesh: never matched; the visits of one cell share its patch
    for q, u in zip(fp, a.unit):
        off = ~np.asarray(M.locate(m, np.nan_to_num(q, nan=1e3))[2])
        assert (u[off] == -1).all()
    cell_patch = {}
    for p, (q, u) in enumerate(zip(fp, a.unit)):
        on = u >= 0
        tri = np.asarray(M.locate(m, q[on])[0])
        for v, pt in zip(u[on], patch[tri]):
            assert cell_patch.setdefault(int(a.unit_cell[v]), pt) == pt
    assert len(set(cell_patch.values())) == a.n_cells
    # Y sums the measured bins of each visit
    Y = np.zeros(a.n_units)
    for y, u in zip(ys, a.unit):
        np.add.at(Y, u[u >= 0], y[u >= 0])
    np.testing.assert_allclose(a.Y, Y)


def test_render_footprints_match_the_render_and_cells_footprints(problem):
    pb, d0, _ = problem
    S, fp = O.render_footprints(pb, d0)
    for Sp, Rp in zip(S, O.render(pb, d0)):
        np.testing.assert_allclose(Sp, Rp, rtol=1e-12, atol=0)
    for qp, pp in zip(fp, pb.poses):
        ref = Mt.footprints(pb.mesh, jnp.asarray(d0), pp, pb.theta, pb.u, pb.n_bins, pb.dr)
        np.testing.assert_allclose(qp, ref, rtol=1e-12, atol=1e-12)
        assert np.isfinite(qp).any() and np.array_equal(np.isnan(qp), np.isnan(ref))


def test_padded_ratio_match_equals_the_plain_one():
    """With held shapes, the fixed-size form gives the same value and gradient, whatever the padding."""
    import jax
    from bbf.match.ratio import log_ratio_match_padded, padded
    a = dataclasses.replace(_two_cells(), shape=np.array([1.1, 2.0, 0.7, 3.0, 1.4]))
    rng = np.random.default_rng(6)
    Ib = [jnp.asarray(rng.uniform(0.5, 2.0, size=u.shape)) for u in a.unit]
    base, g = jax.value_and_grad(lambda I: log_ratio_match(I, a))(Ib)
    for caps in ((5, 2), (9, 4), (40, 30)):
        aux = padded(a, *caps)
        val, gp = jax.value_and_grad(lambda I: log_ratio_match_padded(I, aux, caps[0], caps[0] + caps[1]))(Ib)
        assert float(val) == pytest.approx(float(base), rel=1e-12)
        for x, y in zip(g, gp):
            np.testing.assert_allclose(np.asarray(y), np.asarray(x), rtol=1e-10, atol=1e-14)
    with pytest.raises(ValueError):
        padded(a, 4, 2)


def test_a_rebound_data_term_equals_a_fresh_one(problem):
    """`DataTerm.rebind` with a new assignment gives the value and gradient of a term built for it, without a new
    compile while the assignment fits the capacity."""
    pb, d0, beta = problem
    S = O.render(pb, d0)
    held = dataclasses.replace(pb, ratio_gain=O.ratio_gains(pb, S))
    held = dataclasses.replace(held, ratio=dataclasses.replace(
        held.ratio, shape=Mt.ratio.visit_shapes(O.ratio_means(held, S, held.ratio_gain), held.ratio, 0.3)))
    F = O.data_term(held, warm=True)
    # a new assignment: the footprints at other depths
    d1 = d0 + 0.4 * np.cos(pb.mesh.vertices[:, 0] / 4.0)
    S1, fp1 = O.render_footprints(held, d1)
    times = [np.arange(5) * 0.1, np.arange(5) * 0.1 + 100.0]
    ra = assign(fp1, times, held.y, held.uses_all(), held.theta.L, RatioMatch(cell=1.0))
    new = dataclasses.replace(held, ratio=ra, ratio_gain=None)
    gains = O.ratio_gains(new, S1)
    shape = Mt.ratio.visit_shapes(O.ratio_means(new, S1, gains), ra, 0.3)
    new = dataclasses.replace(new, ratio=dataclasses.replace(ra, shape=shape), ratio_gain=gains)
    assert new.ratio.n_units != held.ratio.n_units or not np.array_equal(new.ratio.unit_cell, held.ratio.unit_cell)
    G = F.rebind(new)
    assert G.core is F.core  # the same compiled function
    fresh = O.data_term(new, warm=True)
    v1, (g1, _), _ = O.value_and_grad(new, d1, beta, G, S=S1)
    v2, (g2, _), _ = O.value_and_grad(new, d1, beta, fresh, S=S1)
    assert v1 == pytest.approx(v2, rel=1e-12)
    np.testing.assert_allclose(g1, g2, rtol=1e-9, atol=1e-9 * np.abs(g2).max())


def test_infer_with_visits_every_step_reassigns_and_ascends():
    """With `per_step`, every accepted step is followed by a new assignment (logged with its jump), each step's own
    gain is positive, and the run ends with the last assignment."""
    from bbf.inference.loop import infer
    from bbf.io import Pass
    m = M.build(-50.0, 50.0, -12.0, 12.0, h=1.0, origin=(0.0, 0.0), z_ref=0.0)
    v = m.vertices
    d_true = np.asarray(20.0 - 2.0 * np.exp(-((v[:, 0] - 2) ** 2 + v[:, 1] ** 2) / (2 * 4.0**2)))
    th = P.Theta(noise=1e-7, L=5.0, eps=0.02, eta=1e-6, zeta=1e-6, ell=1.0, log_kappa_mean=0.0, log_kappa_std=0.5)
    n_bins, dr = 400, 0.1
    u = P.u_grid(n_bins, dr, du=0.05)
    rng = np.random.default_rng(5)
    passes = []
    for i, x0 in enumerate((-6.0, 6.0)):
        poses = np.column_stack([np.full(9, x0), np.linspace(-4, 4, 9), np.full(9, -5.0), np.zeros((9, 3))])
        S = np.asarray(P.unit_intensities(m, jnp.asarray(d_true), poses, th, u, n_bins, dr))
        passes.append(Pass(name=f"x{x0}", poses=poses, y=rng.gamma(th.L, (S + 1e-7) / th.L), dr=dr,
                           t=np.arange(9) * 0.1 + 100.0 * i))
    beta = np.array([0.0, 0.0, 20.0])
    prior = PR.Prior(m, 5.0, 2.0, mu_beta=beta, Sigma_beta=np.diag([0.01, 0.01, 4.0]))
    lines = []
    out = infer(passes, m, prior, th, np.full(m.n_vertices, 20.0), beta, u, rounds=1, inner=3, calib_every=3, chunk=9,
                ratio=RatioMatch(cell=1.0, model_sd=0.3, per_step=True), coarse=8.0, log=lines.append)
    steps = [h for r in out["history"] for h in r["gn"] if "rho" in h]
    assert len(steps) >= 2 and all(h["gain"] > 0 for h in steps)
    assert sum("reassigned (" in s for s in lines) == len(steps)
    assert any("last ratio assignment" in s for s in lines)
    pb = out["problem"]
    assert pb.ratio is not None and pb.ratio.shape is not None and pb.ratio_gain is not None


def test_cell_reflectivity_recovers_each_cells_scale():
    """When every visit of a cell measures its prediction times the same factor, that factor comes back with no
    spread; one disagreeing visit gives a spread."""
    from bbf.match.ratio import cell_reflectivity
    a = _two_cells()
    rng = np.random.default_rng(8)
    Ib = [rng.uniform(0.5, 2.0, size=u.shape) for u in a.unit]
    s1 = np.zeros(a.n_units)
    for x, u in zip(Ib, a.unit):
        np.add.at(s1, u[u >= 0], x[u >= 0])
    kappa = np.array([3.0, 0.25])
    a = dataclasses.replace(a, Y=s1 * kappa[a.unit_cell], shape=np.array([1.0, 2.0, 0.5, 3.0, 1.5]))
    lk, sp = cell_reflectivity(Ib, a)
    np.testing.assert_allclose(lk, np.log(kappa), rtol=1e-12)
    np.testing.assert_allclose(sp, 0.0, atol=1e-12)
    Y = a.Y.copy()
    Y[1] *= 2.0
    _, sp = cell_reflectivity(Ib, dataclasses.replace(a, Y=Y))
    assert sp[0] > 0.1 and sp[1] == pytest.approx(0.0, abs=1e-12)


def test_assign_keeps_the_keys_of_the_matched_cells():
    rng = np.random.default_rng(9)
    fp, times, ys, uses = _random_bins(rng)
    a = assign(fp, times, ys, uses, 1.0, RatioMatch(cell=1.0))
    assert a.cell_key is not None and a.cell_key.shape == (a.n_cells, 2)
    q = np.concatenate([f.reshape(-1, 2) for f in fp])
    unit = np.concatenate([u.ravel() for u in a.unit])
    on = unit >= 0
    np.testing.assert_array_equal(np.floor(q[on]).astype(int), a.cell_key[a.unit_cell[unit[on]]])


def test_a_temper_schedule_sets_each_rounds_temper():
    """A sequence of tempers is a continuation: each round's problem carries its entry, the last one repeated."""
    from bbf.inference.loop import infer
    from bbf.io import Pass
    m = M.build(-30.0, 30.0, -8.0, 8.0, h=1.0, origin=(0.0, 0.0), z_ref=0.0)
    v = m.vertices
    d_true = np.asarray(20.0 - 2.0 * np.exp(-((v[:, 0] - 2) ** 2 + v[:, 1] ** 2) / (2 * 4.0**2)))
    th = P.Theta(noise=1e-7, L=5.0, eps=0.02, eta=1e-6, zeta=1e-6, ell=1.0, log_kappa_mean=0.0, log_kappa_std=0.5)
    n_bins, dr = 300, 0.1
    u = P.u_grid(n_bins, dr, du=0.05)
    rng = np.random.default_rng(7)
    passes = []
    for i, x0 in enumerate((-5.0, 5.0)):
        poses = np.column_stack([np.full(5, x0), np.linspace(-2, 2, 5), np.full(5, -5.0), np.zeros((5, 3))])
        S = np.asarray(P.unit_intensities(m, jnp.asarray(d_true), poses, th, u, n_bins, dr))
        passes.append(Pass(name=f"x{x0}", poses=poses, y=rng.gamma(th.L, (S + 1e-7) / th.L), dr=dr,
                           t=np.arange(5) * 0.1 + 100.0 * i))
    beta = np.array([0.0, 0.0, 20.0])
    prior = PR.Prior(m, 5.0, 2.0, mu_beta=beta, Sigma_beta=np.diag([0.01, 0.01, 4.0]))
    lines = []
    out = infer(passes, m, prior, th, np.full(m.n_vertices, 20.0), beta, u, rounds=3, inner=1, calib_every=3, chunk=5,
                ratio=RatioMatch(cell=1.0, model_sd=0.3), temper=[0.1, 1.0], log=lines.append)
    assert [s for s in lines if "(continuation)" in s] == [
        "round 0: temper 0.1 (continuation)", "round 1: temper 1.0 (continuation)", "round 2: temper 1.0 (continuation)"]
    assert out["problem"].temper == 1.0
