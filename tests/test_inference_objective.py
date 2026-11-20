import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bbf import match as Mt
from bbf import mesh as M
from bbf import ping as P
from bbf import prior as PR
from bbf.inference import objective as O

DEPTH, SONAR_Z = 20.0, -5.0
N_BINS, DR = 400, 0.1


def straight_pass(x0, n=6):
    y = np.linspace(-3, 3, n)
    return np.column_stack([np.full(n, x0), y, np.full(n, SONAR_Z), np.zeros(n), np.zeros(n), np.zeros(n)])


@pytest.fixture(scope="module")
def problem():
    m = M.build(-45.0, 45.0, -10.0, 10.0, h=1.0, origin=(0.0, 0.0))
    v = m.vertices
    d_true = jnp.asarray(DEPTH - 3.0 * np.exp(-((v[:, 0] - 2) ** 2 + v[:, 1] ** 2) / (2 * 4.0**2)))
    th = P.Theta(noise=1e-7, L=0.6, eps=0.02, eta=1e-6, zeta=1e-6, ell=1.0, log_kappa_mean=0.0, log_kappa_std=1.0)
    u = P.u_grid(N_BINS, DR, du=0.05)
    rng = np.random.default_rng(0)
    poses, ys, labels = [], [], []
    for x0 in (-6.0, 6.0):
        pp = straight_pass(x0)
        S = np.asarray(P.unit_intensities(m, d_true, pp, th, u, N_BINS, DR))
        y = rng.gamma(th.L, (S + 1e-7) / th.L)
        poses.append(pp)
        ys.append(y)
        labels.append(np.stack([[P.block_labels(y[t, c], th, c, DR) for c in (0, 1)] for t in range(len(pp))]))
    prior = PR.Prior(m, 5.0, 2.0, mu_beta=np.array([0.0, 0.0, DEPTH]), Sigma_beta=np.diag([0.01, 0.01, 25.0]))
    pb = O.Problem(mesh=m, prior=prior, theta=th, poses=poses, y=ys, labels=labels, u=u, dr=DR, chunk=4,
                   refl=Mt.Reflectivity(m, 4.0, 0.7))
    d0 = d_true + 0.3 * jnp.asarray(np.sin(v[:, 0] / 5))
    return pb, d0, np.array([0.0, 0.0, DEPTH])


def xi_of(pb):
    """A log-reflectivity field away from zero, to evaluate at."""
    v = pb.mesh.vertices
    return jnp.asarray(0.4 * np.cos(v[:, 0] / 4) * np.sin(v[:, 1] / 3 + 1.0))


def direct(pb, d, xi=None):
    """The same data term written out factor by factor, with no chunking."""
    total = 0.0
    uses = pb.uses()
    S = [P.unit_intensities(pb.mesh, d, pp, pb.theta, pb.u, pb.n_bins, pb.dr, xi) for pp in pb.poses]
    for p in range(2):
        for t in range(len(pb.poses[p])):
            for c in (0, 1):
                y, s, use = jnp.asarray(pb.y[p][t, c]), S[p][t, c], jnp.asarray(uses[p][t, c])
                total = total + P.log_ping_side(pb.theta, c, pb.dr, y, s, use)
                u_hat, _ = P.mode(pb.theta, c, pb.dr, y, s, use)
                total = total + P.log_shadow_side(pb.theta, c, pb.labels[p][t, c], s, u_hat)
    return total


def test_value_and_grad_match_direct_composition(problem):
    pb, d0, beta = problem
    xi = xi_of(pb)
    value, (gd, gb), details = O.value_and_grad(pb, d0, beta, xi=xi)
    assert pb.refl is not None
    prior = pb.prior.log_density(np.asarray(d0), beta) + pb.refl.log_density(np.asarray(xi))
    assert value == pytest.approx(prior + float(direct(pb, d0, xi)), rel=1e-10)
    assert value == pytest.approx(O.value(pb, d0, beta, xi=xi), rel=1e-12)
    # data gradient against plain reverse mode through the whole composition, in d and in xi
    g_direct, gx_direct = (np.asarray(g) for g in jax.grad(lambda d, x: direct(pb, d, x), argnums=(0, 1))(d0, xi))
    gd_prior, gb_prior = pb.prior.grad(np.asarray(d0), beta)
    np.testing.assert_allclose(gd - gd_prior, g_direct, rtol=1e-6, atol=1e-6 * np.abs(g_direct).max())
    np.testing.assert_allclose(details["grad_d_data"], g_direct, rtol=1e-6, atol=1e-6 * np.abs(g_direct).max())
    np.testing.assert_allclose(details["grad_xi"] - pb.refl.grad(np.asarray(xi)), gx_direct, rtol=1e-6,
                               atol=1e-6 * np.abs(gx_direct).max())
    assert np.abs(gx_direct).max() > 0
    np.testing.assert_allclose(gb, gb_prior)
    assert details["u"][0].shape == (6, 2)


def test_no_field_is_the_field_at_zero(problem):
    """Without `refl` the problem is the uniform-reflectivity one: same data term as xi = 0, and no xi prior."""
    import dataclasses
    pb, d0, beta = problem
    plain = dataclasses.replace(pb, refl=None)
    v0, (g0, _), det0 = O.value_and_grad(plain, d0, beta)
    v1, (g1, _), _ = O.value_and_grad(pb, d0, beta, xi=np.zeros(pb.mesh.n_vertices))
    assert v0 == pytest.approx(v1, rel=1e-12)
    np.testing.assert_allclose(g0, g1, rtol=1e-10)
    np.testing.assert_allclose(det0["S"][0], np.asarray(P.unit_intensities(pb.mesh, d0, pb.poses[0], pb.theta, pb.u,
                                                                              pb.n_bins, pb.dr)), rtol=1e-10)


def test_psi_gradient_matches_direct_composition(problem):
    """The gradient in the sonar parameters (beam through S, gain directly) plus their prior."""
    import dataclasses
    pb, d0, beta = problem
    psi0 = P.psi_of(pb.theta)
    sd = np.array([[0.2, 0.3, 0.5], [0.2, 0.3, 0.5]])
    pbf = dataclasses.replace(pb, psi_prior=(psi0 + 0.05, sd))
    psi = psi0 + np.array([[0.03, -0.05, 0.02], [-0.04, 0.06, -0.01]])
    value, _, details = O.value_and_grad(pbf, d0, beta, psi=psi)
    assert value == pytest.approx(O.value(pbf, d0, beta, psi=psi), rel=1e-12)

    def direct_psi(p):
        return direct(dataclasses.replace(pb, theta=P.with_psi(pb.theta, p)), d0)

    g = np.asarray(jax.grad(direct_psi)(jnp.asarray(psi)))
    z = (psi - (psi0 + 0.05)) / sd
    np.testing.assert_allclose(details["grad_psi"], g - z / sd, rtol=1e-6, atol=1e-8 * np.abs(g).max())
    assert np.abs(g[:, :2]).max() > 0 and np.abs(g[:, 2]).max() > 0  # beam and gain both reach the data


def test_gauss_newton_recovers_sonar_parameters(problem):
    r"""From a wrong beam and gain at the true depths, GN restores the brightness shape across range.

    Data are simulated with little speckle (:math:`L = 50`) at theta's :math:`\psi`. On a level
    seabed seen from one altitude, range and beam angle go together, so the gain and the beam
    trade off; what the data fix is their product, :math:`\log g + \log S` up to a constant
    per ping-side (which :math:`\kappa` absorbs). That is what is checked.
    """
    import dataclasses
    import itertools

    from bbf.inference.gauss_newton import gauss_newton
    from bbf.inference.loop import jacobians
    pb, d0, beta = problem
    d_true = d0 - 0.3 * jnp.asarray(np.sin(pb.mesh.vertices[:, 0] / 5))
    th = dataclasses.replace(pb.theta, L=50.0)
    rng = np.random.default_rng(3)
    ys = [rng.gamma(th.L, (s + 1e-7) / th.L) for s in O.render(pb, d_true)]
    labels = [np.stack([[P.block_labels(y[t, c], th, c, pb.dr) for c in (0, 1)] for t in range(len(y))]) for y in ys]
    psi_true = P.psi_of(th)
    wrong = psi_true + np.array([[np.radians(6), np.radians(-8), 0.05], [np.radians(-6), np.radians(8), -0.05]])
    sd = np.array([[np.radians(15), np.radians(20), 0.2]] * 2)
    pbf = dataclasses.replace(pb, theta=th, y=ys, labels=labels, refl=None, psi_prior=(wrong, sd))  # prior on the wrong values

    def shape(psi):
        """log g + log S per ping-side over its echo bins, centred."""
        thp = P.with_psi(th, psi)
        out = []
        for Sp in O.render(pbf, d_true, psi):
            for c in (0, 1):
                ok = Sp[:, c] > 1e-2 * Sp[:, c].max()
                v = np.where(ok, np.log(np.where(ok, Sp[:, c], 1.0)) + np.asarray(P.log_gain(P.bin_ranges(pb.n_bins, pb.dr), thp.gain_db[c])), np.nan)
                out.append(v - np.nanmean(v, axis=1, keepdims=True))
        return np.concatenate(out)

    ref = shape(psi_true)
    Jb = O.beam_jacobian(pbf, d_true, wrong)
    _, _, psi, _, hist = gauss_newton(pbf, d_true, beta, jacobians(pbf, d_true), Jb=Jb, psi=wrong, max_iter=8,
                                      log=lambda _: None)
    assert all(b["ell"] >= a["ell"] for a, b in itertools.pairwise(hist))
    err0 = np.sqrt(np.nanmean((shape(wrong) - ref) ** 2))
    err = np.sqrt(np.nanmean((shape(psi) - ref) ** 2))
    assert err < 0.3 * err0, (err, err0, np.degrees(psi[:, :2]), psi[:, 2])


def test_ggn_matches_weighted_jacobian_products(problem):
    r""":math:`H z = J_r^\top J_r z` + priors, with :math:`J_r` from JAX through the full render.

    :math:`r` stacks the weighted predictions of every term at the assembly point: ping bins
    :math:`\sqrt{W_b} \log\bar I_b` and dark blocks :math:`p_j / (1 - p_j^0)`; the weights are taken
    from `ggn` at that point, so this checks every derivative and every block of the assembly,
    including :math:`\xi` (which reaches every row through the render), :math:`u` and :math:`\psi`.
    """
    import dataclasses

    from bbf.inference import ggn
    from bbf.inference.loop import jacobians
    pb, d0, beta = problem
    xi = xi_of(pb)
    psi = P.psi_of(pb.theta) + np.array([[0.05, -0.04, 0.02], [-0.03, 0.06, -0.01]])
    sd = np.array([[0.2, 0.3, 0.5], [0.25, 0.35, 0.4]])
    pbf = dataclasses.replace(pb, psi_prior=(psi, sd))
    _, _, det = O.value_and_grad(pbf, d0, beta, psi=psi, xi=xi)
    S, u_hat = det["S"], det["u"]
    th = P.with_psi(pb.theta, psi)
    J = jacobians(dataclasses.replace(pbf, theta=th), d0, xi)
    Jb = O.beam_jacobian(pbf, d0, psi, xi)
    H = ggn.assemble(pbf, J, S, u_hat, Jb=Jb, psi=psi)
    pw = ggn.ping_weights(dataclasses.replace(pbf, theta=th), S, u_hat, pbf.uses())
    rkey = O._key(pbf)
    uses = pbf.uses()
    dark = [~np.asarray(lab, bool) for lab in pbf.labels]

    def lit(Sp, u, thp, side):
        return jax.vmap(lambda s, uu: P.lit_probability(s, uu, thp, side))(Sp, u)

    p0 = [[np.asarray(lit(jnp.asarray(S[p][:, c]), jnp.asarray(u_hat[p][:, c]), th, c)) for c in (0, 1)]
          for p in range(2)]

    def r(x):
        thp = P.with_psi(pb.theta, x["psi"])
        Sx = [O._render_chunk(rkey, x["d"], jnp.asarray(pp), x["psi"], x["xi"]) for pp in pbf.poses]
        out = []
        for p in range(2):
            W = pw[p][0]
            for c in (0, 1):
                Ib, _ = P.predicted(Sx[p][:, c], x["u"][p][:, c][:, None], thp, c, pbf.dr)
                out.append((jnp.sqrt(W[:, c]) * jnp.where(uses[p][:, c], jnp.log(Ib), 0.0)).ravel())
                pj = lit(Sx[p][:, c], jnp.asarray(u_hat[p][:, c]), thp, c)  # kappa held at its mode
                nb = pj.shape[-1]
                dk = dark[p][:, c, :nb]
                out.append(jnp.where(dk, pj / np.where(dk, 1 - p0[p][c], 1.0), 0.0).ravel())  # lit blocks may have p = 1
        return jnp.concatenate(out)

    x0 = {"d": jnp.asarray(d0), "xi": xi, "beta": jnp.asarray(beta), "u": [jnp.asarray(u) for u in u_hat],
          "psi": jnp.asarray(psi)}
    lay = ggn.layout(pbf)
    n, nf, ib = lay["n"], lay["nf"], lay["beta"]
    assert nf == 2 * n and H.shape == (lay["size"],) * 2
    rng = np.random.default_rng(5)
    for _ in range(2):
        z = jax.tree.map(lambda a: jnp.asarray(rng.standard_normal(np.shape(a))), x0)
        _, Jz = jax.jvp(r, (x0,), (z,))
        _, vjp = jax.vjp(r, x0)
        (JtJz,) = vjp(Jz)
        flat = np.zeros(lay["size"])
        zf = np.zeros(lay["size"])
        for name, sl in (("d", slice(0, n)), ("xi", slice(lay["xi"], lay["xi"] + n)), ("beta", slice(ib, ib + 3)),
                         ("psi", slice(lay["psi"], lay["size"]))):
            flat[sl] = np.ravel(JtJz[name])
            zf[sl] = np.ravel(z[name])
        for p in range(2):
            o = lay["u_pass"][p]
            flat[o:o + u_hat[p].size] = np.ravel(JtJz["u"][p])
            zf[o:o + u_hat[p].size] = np.ravel(z["u"][p])
        # priors: (d, beta), xi, u, psi
        db = np.concatenate([np.arange(n), ib + np.arange(3)])
        flat[db] += pbf.prior.joint_precision() @ zf[db]
        assert pbf.refl is not None
        flat[lay["xi"]:lay["xi"] + n] += pbf.refl.Lambda @ zf[lay["xi"]:lay["xi"] + n]
        for p in range(2):
            o = lay["u_pass"][p]
            std = np.tile([th.at(0).log_kappa_std, th.at(1).log_kappa_std], len(pbf.poses[p]))
            flat[o:o + 2 * len(pbf.poses[p])] += zf[o:o + 2 * len(pbf.poses[p])] / std**2
        flat[lay["psi"]:] += zf[lay["psi"]:] / sd.ravel() ** 2
        Hz = H @ zf
        np.testing.assert_allclose(Hz, flat, rtol=1e-6, atol=1e-8 * np.abs(flat).max())
    # without the field the same assembly has no xi block
    plain = dataclasses.replace(pbf, refl=None)
    assert ggn.layout(plain)["nf"] == n
    with pytest.raises(ValueError):
        ggn.assemble(plain, J, S, u_hat, Jb=Jb, psi=psi)


def test_beam_jacobian_matches_finite_differences(problem):
    pb, d0, _ = problem
    psi = P.psi_of(pb.theta)
    Jb = O.beam_jacobian(pb, d0, psi)
    h = 1e-6
    for k in (0, 1):
        t = np.zeros_like(psi)
        t[:, k] = h
        Sp, Sm = O.render(pb, d0, psi + t), O.render(pb, d0, psi - t)
        for p in range(2):
            fd = (Sp[p] - Sm[p]) / (2 * h)
            np.testing.assert_allclose(Jb[p][..., k], fd, rtol=1e-5, atol=1e-6 * np.abs(fd).max())


def test_gauss_newton_refuses_a_non_finite_start(problem):
    from bbf.inference.gauss_newton import gauss_newton
    from bbf.inference.loop import jacobians
    pb, d0, beta = problem
    d = np.asarray(d0).copy()
    J = jacobians(pb, d)
    d[0] = np.nan
    with pytest.raises(FloatingPointError):
        gauss_newton(pb, d, beta, J, max_iter=1, log=lambda _: None)


def test_laplace_reports_psi_uncertainty(problem):
    import dataclasses

    from bbf.inference.laplace import laplace
    from bbf.inference.loop import jacobians
    pb, d0, beta = problem
    psi = P.psi_of(pb.theta)
    sd = np.array([[0.2, 0.3, 0.5]] * 2)
    pbf = dataclasses.replace(pb, psi_prior=(psi, sd))
    out = laplace(pbf, d0, beta, jacobians(pbf, d0), n_samples=50, rng=0, Jb=O.beam_jacobian(pbf, d0, psi))
    assert out["sd_psi"].shape == (2, 3) and out["sd_xi"].shape == d0.shape
    assert np.all(out["sd_xi"] > 0) and out["sd_xi"].min() < 0.7 * pbf.refl.sigma  # the data narrow the field under the swath
    assert np.all(out["sd_psi"] > 0) and np.all(out["sd_psi"] < sd)  # the data narrow every parameter
