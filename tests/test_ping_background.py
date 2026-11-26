r"""The background law :math:`N_c (r / r_N)^{\nu} e^{\gamma o}` (`Theta.noise_slope`, `Theta.noise_power`) in every term that uses it."""
import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bbf import mesh as M
from bbf import ping as P
from bbf import prior as PR
from bbf.inference import ggn
from bbf.inference import objective as O
from bbf.inference.calibrate import calibrate
from bbf.inference.init import first_echo
from bbf.inference.loop import beyond, labels_of
from bbf.io import Pass
from bbf.ping import factor as F
from bbf.ping import shadow as Sh
from bbf.ping.theta import NOISE_REF, STARBOARD, Theta

N_BINS, DR = 400, 0.1
NU, GAMMA = -2.0, 1.0


def theta(**kw):
    base = {"noise": 1e-7, "L": 0.6, "eps": 0.02, "eta": 1e-6, "zeta": 1e-6, "ell": 1.0, "log_kappa_mean": 0.0,
            "log_kappa_std": 1.0, "gain_db": [[0.0, 0.1, 2.0], [0.0, 0.05, 3.0]]}
    return Theta(**{**base, **kw})


def test_background_default_is_constant_and_law_follows_range_and_power():
    r = F.bin_ranges(N_BINS, DR)
    np.testing.assert_allclose(np.asarray(F.background(theta(), 0, r, 0.7)), 1e-7)
    th = theta(noise=[1e-7, 3e-7], noise_slope=NU, noise_power=GAMMA)
    off = np.log(0.1)
    for c in (0, 1):
        expect = th.at(c).noise * (np.asarray(r) / NOISE_REF) ** NU * 0.1
        np.testing.assert_allclose(np.asarray(F.background(th, c, r, off)), expect, rtol=1e-12)
        # with no echo the predicted mean is the gain times the background
        I_bar, _ = F.predicted(jnp.zeros(N_BINS), 0.0, th, c, DR, off)
        np.testing.assert_allclose(np.asarray(I_bar), np.exp(np.asarray(F.log_gain(r, th.at(c).gain_db))) * expect, rtol=1e-12)
    # an echo's part does not move with the offset here: it sits in u
    S = jnp.full(N_BINS, 1e-3)
    a, b = (np.asarray(F.predicted(S, 0.0, th, 0, DR, o)[0]) for o in (0.0, off))
    assert np.all(a > b) and np.allclose(a - b, np.asarray(F.predicted(jnp.zeros(N_BINS), 0.0, th, 0, DR)[0]) * 0.9)


def test_labels_keep_their_false_alarm_rate_under_the_law():
    th = theta(eps=1e-12, noise_slope=NU, noise_power=GAMMA)
    rng = np.random.default_rng(0)
    off = np.log(np.array([0.1, 0.5, 1.0]))[:, None]
    n = 8 * 6000
    r = F.bin_ranges(n, DR)
    mean = np.exp(np.asarray(F.log_gain(r, th.at(STARBOARD).gain_db))) * np.asarray(F.background(th, STARBOARD, r, off))
    y = rng.gamma(th.L, mean / th.L)
    rate = Sh.block_labels(y, th, STARBOARD, DR, off[:, 0]).mean(axis=1)
    np.testing.assert_allclose(rate, th.label_fa, atol=4 * np.sqrt(th.label_fa / 6000))
    # without the power the weak pings look dark and the labels of the strong ones are unchanged
    plain = Sh.block_labels(y, dataclasses.replace(th, noise_power=0.0), STARBOARD, DR).mean(axis=1)
    assert plain[0] < 0.2 * th.label_fa and plain[2] == pytest.approx(rate[2])


def test_lit_probability_uses_the_background_of_each_block():
    th = theta(noise_slope=NU, noise_power=GAMMA)
    S = jnp.full(N_BINS, 3e-7)
    with pytest.raises(ValueError):
        Sh.lit_probability(S, 0.0, th, STARBOARD)
    off = np.log(0.5)
    p = np.asarray(Sh.lit_probability(S, 0.0, th, STARBOARD, DR, off))
    # block by block it is the constant-background probability at that block's background
    r = np.asarray(F.bin_ranges(N_BINS, DR)).reshape(-1, th.block)
    for j in (3, 20, 45):
        nj = float(np.mean(np.asarray(F.background(th, STARBOARD, r[j], off))))
        flat = dataclasses.replace(th, noise=nj, noise_slope=0.0, noise_power=0.0)
        assert p[j] == pytest.approx(float(Sh.lit_probability(S, 0.0, flat, STARBOARD)[j]), rel=1e-9)
    assert np.all(np.diff(p[5:]) > 0)  # the background falls with range, so the same echo stands out more


@pytest.fixture(scope="module")
def problem():
    m = M.build(-45.0, 45.0, -6.0, 6.0, h=1.0, origin=(0.0, 0.0))
    v = m.vertices
    d_true = jnp.asarray(20.0 - 3.0 * np.exp(-((v[:, 0] - 2) ** 2 + v[:, 1] ** 2) / (2 * 4.0**2)))
    th = Theta(noise=1e-7, L=0.6, eps=0.02, eta=1e-6, zeta=1e-6, ell=1.0, log_kappa_mean=0.0, log_kappa_std=1.0,
               noise_slope=NU, noise_power=GAMMA)
    u = P.u_grid(N_BINS, DR, du=0.05)
    rng = np.random.default_rng(0)
    poses, ys, labels, offsets = [], [], [], []
    for x0, power in ((-6.0, [1.0, 0.2, 1.0]), (6.0, [0.5, 0.5, 0.1])):
        yy = np.linspace(-2, 2, 3)
        pp = np.column_stack([np.full(3, x0), yy, np.full(3, -5.0), np.zeros((3, 3))])
        off = np.log(np.array(power))
        S = np.asarray(P.unit_intensities(m, d_true, pp, th, u, N_BINS, DR))
        mean = np.stack([np.asarray(F.predicted(jnp.asarray(S[:, c]), off[:, None], th, c, DR, off[:, None])[0])
                         for c in (0, 1)], axis=1)
        y = rng.gamma(th.L, mean / th.L)
        poses.append(pp)
        ys.append(y)
        offsets.append(off)
        labels.append(np.stack([Sh.block_labels(y[:, c], th, c, DR, off) for c in (0, 1)], axis=1))
    prior = PR.Prior(m, 5.0, 2.0, mu_beta=np.array([0.0, 0.0, 20.0]), Sigma_beta=np.diag([0.01, 0.01, 25.0]))
    pb = O.Problem(mesh=m, prior=prior, theta=th, poses=poses, y=ys, labels=labels, u=u, dr=DR, chunk=4,
                   kappa_offset=offsets)
    return pb, d_true + 0.3 * jnp.asarray(np.sin(v[:, 0] / 5)), np.array([0.0, 0.0, 20.0])


def direct(pb, d):
    """The data term factor by factor, with each ping's offset."""
    total = 0.0
    uses = pb.uses()
    for p in range(2):
        S = P.unit_intensities(pb.mesh, d, pb.poses[p], pb.theta, pb.u, pb.n_bins, pb.dr)
        for t in range(len(pb.poses[p])):
            off = float(pb.kappa_offset[p][t])
            for c in (0, 1):
                y, s, use = jnp.asarray(pb.y[p][t, c]), S[t, c], jnp.asarray(uses[p][t, c])
                val, u_hat = F.laplace_ping_side(pb.theta, c, pb.dr, y, s, use, offset=off)
                total = total + val + Sh.log_shadow_side(pb.theta, c, pb.labels[p][t, c], s, u_hat, None, pb.dr, off)
    return total


def test_objective_carries_the_law_and_the_offsets(problem):
    pb, d0, beta = problem
    value, _, det = O.value_and_grad(pb, d0, beta)
    assert value == pytest.approx(pb.prior.log_density(np.asarray(d0), beta) + float(direct(pb, d0)), rel=1e-10)
    g_direct = np.asarray(jax.grad(lambda d: direct(pb, d))(d0))
    np.testing.assert_allclose(det["grad_d_data"], g_direct, rtol=1e-6, atol=1e-8 * np.abs(g_direct).max())
    # the law matters: with a constant background of the same level at 20 m the value differs
    flat = dataclasses.replace(pb, theta=dataclasses.replace(pb.theta, noise_slope=0.0, noise_power=0.0))
    assert abs(O.value(flat, d0, beta) - value) > 1.0
    # the Gauss-Newton weights differentiate the same predicted mean
    S, u_hat = det["S"], det["u"]
    pw = ggn.ping_weights(pb, S, u_hat, pb.uses())
    for p in range(2):
        off = jnp.asarray(pb.kappa_offset[p])[:, None]
        for c in (0, 1):
            def log_mean(s, u, c=c, off=off):
                return jnp.log(F.predicted(s, u, pb.theta, c, pb.dr, off)[0])
            s, uu = jnp.asarray(S[p][:, c]), jnp.asarray(u_hat[p][:, c])[:, None]
            _, dS = jax.jvp(lambda x, uu=uu: log_mean(x, uu), (s,), (jnp.ones_like(s),))
            _, du = jax.jvp(lambda x, s=s: log_mean(s, x), (uu,), (jnp.ones_like(uu),))
            np.testing.assert_allclose(pw[p][1][:, c], np.asarray(dS), rtol=1e-9)
            np.testing.assert_allclose(pw[p][2][:, c], np.asarray(du), rtol=1e-9)
    # and the shadow rows are finite and nonzero on dark blocks that predict an echo
    rows = ggn.shadow_rows(pb, S, u_hat)
    assert all(np.all(np.isfinite(r)) for r in rows) and any(np.any(r != 0) for r in rows)


def test_calibrate_and_first_echo_under_the_law():
    rng = np.random.default_rng(2)
    th = theta(eps=0.1, eta=3e-6, zeta=1e-12, noise_slope=NU, noise_power=GAMMA, gain_db=np.zeros((2, 3)))
    m = 30
    r = np.asarray(F.bin_ranges(N_BINS, DR))
    off = np.log(rng.choice([0.1, 0.5], m))
    S = np.tile(np.where(r > 15, 1e-5, 0.0), (m, 2, 1))
    u = rng.normal(0.0, 0.3, (m, 2)) + off[:, None]
    y = np.zeros((m, 2, N_BINS))
    for c in (0, 1):
        I_bar, I_sp = (np.asarray(a) for a in F.predicted(jnp.asarray(S[:, c]), u[:, c, None], th, c, DR, off[:, None]))
        spike = rng.random((m, N_BINS)) < th.at(c).eps
        y[:, c] = np.where(spike, rng.exponential(I_sp), rng.gamma(th.L, I_bar / th.L))
    fit, info = calibrate(y, S, np.ones_like(y, bool), dataclasses.replace(th, eta=1e-5), DR, offset=off)
    assert info["converged"]
    np.testing.assert_allclose(np.log(np.asarray(fit.eta)), np.log(3e-6), atol=0.4)
    assert np.sqrt(np.mean((info["u"] - u) ** 2)) < 0.15
    # without the power in the background the weak pings' water column reads as far too dark: the fit is worse
    _, plain = calibrate(y, S, np.ones_like(y, bool), dataclasses.replace(th, eta=1e-5, noise_power=0.0), DR, offset=off)
    assert plain["nll"] > info["nll"] + 100.0
    # the water column stays mostly dark on pings of either power (spikes light a share of the blocks)
    ps = Pass(name="law", poses=np.zeros((m, 6)), y=y, dr=DR, t=np.arange(m) * 0.1, power=100 * np.exp(off))
    lab = labels_of(ps, th)
    assert lab[:, :, : int(14 / (th.block * DR))].mean() < 0.4 and lab[:, :, int(16 / (th.block * DR)):].mean() > 0.95


def test_first_echo_under_the_law():
    """The detector normalises each ping by its own background: echoes at 15 and 18 m are found at either power."""
    rng = np.random.default_rng(3)
    th = theta(eps=1e-9, noise_slope=NU, noise_power=GAMMA, gain_db=np.zeros((2, 3)))
    m = 40
    r = np.asarray(F.bin_ranges(N_BINS, DR))
    off = np.log(rng.choice([0.1, 0.5], m))
    S = np.stack([np.where(r > r_echo, 1e-5, 0.0) for r_echo in (15.0, 18.0)])
    y = np.stack([rng.gamma(th.L, np.asarray(F.predicted(jnp.asarray(S[c]), off[:, None], th, c, DR, off[:, None])[0]) / th.L)
                  for c in (0, 1)], axis=1)
    ps = Pass(name="law", poses=np.zeros((m, 6)), y=y, dr=DR, t=np.arange(m) * 0.1, power=100 * np.exp(off))
    r0 = first_echo(ps, th)
    assert np.isfinite(r0).mean() > 0.95
    assert np.nanmedian(np.abs(r0 - [15.0, 18.0])) <= th.block * DR


def test_beyond_masks_near_blocks():
    np.testing.assert_array_equal(beyond(6, 2.0, 5.0), [False, False, True, True, True, True])
    assert beyond(4, 2.0, 0.0).all()
