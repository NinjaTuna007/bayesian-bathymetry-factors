import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.stats import gamma as gamma_dist

from bbf import mesh as M
from bbf.ping import factor as F
from bbf.ping import render as Rn
from bbf.ping import shadow as Sh
from bbf.ping.theta import STARBOARD, Theta

DEPTH, SONAR_Z = 20.0, -5.0
N_BINS, DR = 800, 0.05
U = Rn.u_grid(N_BINS, DR, du=0.01)
LEVEL = jnp.array([0.0, 0.0, SONAR_Z, 0.0, 0.0, 0.0])


def make_theta(eps=0.02):
    return Theta(noise=1e-7, L=0.5, eps=eps, eta=1e-3, zeta=1e-3, ell=1.0,
                 log_kappa_mean=0.0, log_kappa_std=1.0, gain_db=[[0.0, 0.1, 2.0], [0.0, 0.05, 3.0]])


def simulate(theta, side, S, log_kappa, rng):
    I_bar, I_sp = (np.asarray(a) for a in F.predicted(jnp.asarray(S), log_kappa, theta, side, DR))
    spike = rng.random(S.shape) < theta.at(side).eps
    main = gamma_dist.rvs(theta.L, scale=I_bar / theta.L, random_state=rng)
    return np.where(spike, gamma_dist.rvs(1.0, scale=I_sp, random_state=rng), main)


def test_false_alarm_rate_on_noise():
    th = make_theta(eps=1e-12)
    rng = np.random.default_rng(0)
    n = 8 * 20000
    y = simulate(th, STARBOARD, np.zeros(n), 0.0, rng)
    rate = Sh.block_labels(y, th, STARBOARD, DR).mean()
    assert rate == pytest.approx(th.label_fa, abs=4 * np.sqrt(th.label_fa / 20000))


def test_detection_matches_simulation_and_rises_with_snr():
    """With S constant within blocks the moment-matched block mean is exact."""
    th = make_theta()
    rng = np.random.default_rng(1)
    n_blocks = 4000
    for snr in (0.0, 0.5, 2.0, 8.0):
        S = np.full(8 * n_blocks, snr * th.at(STARBOARD).noise)  # :math:`\kappa = 1`
        y = simulate(th, STARBOARD, S, 0.0, rng)
        observed = Sh.block_labels(y, th, STARBOARD, DR).mean()
        predicted = float(Sh.lit_probability(jnp.asarray(S), 0.0, th, STARBOARD).mean())
        assert observed == pytest.approx(predicted, abs=0.03)
    p = [float(Sh.lit_probability(jnp.full(8, s * 1e-7), 0.0, th, STARBOARD)[0]) for s in (0, 1, 3, 10, 30)]
    assert np.all(np.diff(p) > 0)


@pytest.fixture(scope="module")
def wall_scene():
    """True seabed: a wall up to a crest 3 m below the sonar, then a drop back down, so the far
    side lies in shadow. Model: the wall up to a plateau, which would be lit there."""
    m = M.build(-40.0, 60.0, -30.0, 30.0, h=1.0, origin=(0.0, 0.0))
    x = m.vertices[:, 0]
    wall = np.clip((x - 5.0) / 5.0, 0.0, 1.0)  # 0 before x = 5, 1 after x = 10
    d_model = DEPTH - 12.0 * wall
    behind = (x > 12.0)
    d_true = np.where(behind, DEPTH, d_model)
    return m, jnp.asarray(d_true), jnp.asarray(d_model), behind


def test_shadow_factor_lowers_seabed_behind_crest(wall_scene):
    """Dark blocks where the model predicts an echo: the factor prefers the drop.

    Locally, less echo can come from raising the plateau towards grazing as well as from
    lowering it into shadow (the 12 m drop is far outside the linear range), so only the
    values are compared; the gradient must be finite and reach the seabed behind the wall.
    """
    m, d_true, d_model, behind = wall_scene
    th = make_theta()
    rng = np.random.default_rng(2)
    labels, lk = [], jnp.zeros(2)
    for side in (0, 1):
        S = np.asarray(Rn.unit_intensity(m, d_true, LEVEL, side, th, U, N_BINS, DR))
        labels.append(Sh.block_labels(simulate(th, side, S, 0.0, rng), th, side, DR))
    labels = np.stack(labels)

    def f(d):
        return Sh.log_shadow_factor(m, d, LEVEL, labels, th, U, DR, N_BINS, lk)

    assert float(f(d_true)) > float(f(d_model))
    g = np.asarray(jax.grad(f)(d_model))
    assert np.all(np.isfinite(g)) and np.abs(g[behind]).max() > 0


def test_ping_use_combines_labels_and_valid():
    labels = np.array([[True, False, True], [False, True, True]])
    use = Sh.ping_use(labels, 26, 8)
    assert use.shape == (2, 26)
    assert use[0, :8].all() and not Sh.ping_use(labels, 26, 8, valid=~labels).any()
    assert not use[0, 8:16].any() and use[0, 16:24].all() and not use[:, 24:].any()
    assert not use[1, :8].any() and use[1, 8:24].all()


def test_lit_probability_default_equals_spike_formula():
    """Without a measured rate, p = eps_B + (1 - eps_B) D(m), the earlier form."""
    from jax.scipy.special import gammainc
    th = make_theta()
    S = jnp.asarray(np.linspace(0, 5e-7, 80))
    p = np.asarray(Sh.lit_probability(S, 0.0, th, STARBOARD))
    k = th.block * th.L
    m = 1 + np.asarray(S).reshape(-1, th.block).mean(axis=1) / th.at(STARBOARD).noise
    eps_b = 1 - (1 - th.at(STARBOARD).eps) ** th.block
    D = 1 - np.asarray(gammainc(k, k * Sh.threshold(th) / m))
    np.testing.assert_allclose(p, eps_b + (1 - eps_b) * D, rtol=1e-12)


def test_lit_probability_with_measured_rate():
    import dataclasses
    th = dataclasses.replace(make_theta(), wc_lit=np.stack([np.linspace(0.5, 0.0, 10), np.full(10, 0.03)]))
    dark = np.asarray(Sh.lit_probability(jnp.zeros(80), 0.0, th, 0))
    np.testing.assert_allclose(dark, np.linspace(0.5, 0.0, 10), atol=1e-12)
    bright = np.asarray(Sh.lit_probability(jnp.full(80, 1e-3), 0.0, th, 1))
    assert bright.min() > 0.999
    # beyond the measured range the last rate continues
    assert np.asarray(Sh.lit_probability(jnp.zeros(120), 0.0, th, 1))[-1] == pytest.approx(0.03)


def test_water_column_lit_rate_recovers_profile():
    rng = np.random.default_rng(6)
    n, nb, bdr = 400, 100, 0.4
    rb = (np.arange(nb) + 0.5) * bdr
    true = np.where(rb < 4, 0.6, 0.02)  # ringing near the sonar, quiet water beyond
    r0 = np.stack([rng.uniform(15, 25, n), rng.uniform(15, 25, n)], axis=1)
    lab = rng.random((n, 2, nb)) < true
    lab |= rb[None, None, :] >= r0[..., None]  # seabed beyond the first echo is lit
    rate = Sh.water_column_lit_rate([lab], [r0], bdr, onset=1.0, smooth=0.4)
    assert rate.shape == (2, nb)
    np.testing.assert_allclose(rate[:, (rb > 0.5) & (rb < 3.5)], 0.6, atol=0.06)
    np.testing.assert_allclose(rate[:, (rb > 6) & (rb < 12)], 0.02, atol=0.02)
    np.testing.assert_allclose(rate[:, rb > 30], 0.02, atol=0.02)  # beyond any water column: the quiet level
    valid = Sh.onset_valid(r0, nb, bdr, onset=1.0)
    t = 0
    zone = (rb >= r0[t, 0] - 1.0) & (rb < r0[t, 0])
    assert not valid[t, 0, zone].any() and valid[t, 0, ~zone].all()


def test_seabed_above_the_waterline_returns_nothing():
    import dataclasses
    m = M.build(-30.0, 30.0, -10.0, 10.0, h=1.0, origin=(0.0, 0.0))
    d = jnp.full(m.n_vertices, DEPTH)  # seabed at z = -20 (z_ref 0)
    th = make_theta()
    wet = np.asarray(Rn.unit_intensity(m, d, LEVEL, STARBOARD, th, U, N_BINS, DR))
    dry = np.asarray(Rn.unit_intensity(m, d, LEVEL, STARBOARD, dataclasses.replace(th, z_water=-25.0), U, N_BINS, DR))
    assert wet.sum() > 0 and dry.sum() == 0


def test_measured_floor_recovers_a_falling_noise_profile():
    """Noise that falls with range (as port's water column does) is measured, and labels against it hit p_fa."""
    import dataclasses
    th = make_theta(eps=1e-12)
    rng = np.random.default_rng(7)
    n, nb, B = 400, 100, th.block
    r_bin = np.arange(nb * B) * DR
    floor_true = 1e9 * np.exp(-r_bin / 2.0) + 2e8  # intensity units, no gain applied
    r0 = np.stack([rng.uniform(3.5, 4.8, n), rng.uniform(3.5, 4.8, n)], axis=1)
    y = rng.gamma(th.L, floor_true / th.L, size=(n, 2, nb * B))
    y = np.where(r_bin[None, None, :] >= r0[..., None], y + 1e11, y)  # seabed beyond the first echo
    F = Sh.measured_floor([y], [r0], th, DR, onset=0.5, smooth=0.4)
    blk_true = floor_true[: nb * B].reshape(nb, B).mean(1)
    rb_end = (np.arange(nb) + 1) * B * DR
    seen = rb_end < 3.0  # water column of every ping
    np.testing.assert_allclose(F[:, seen], np.broadcast_to(blk_true[seen], (2, seen.sum())), rtol=0.12)
    thf = dataclasses.replace(th, floor=F)
    noise = rng.gamma(th.L, floor_true / th.L, size=(4000, nb * B))[:, : 3 * B]
    rate = Sh.block_labels(noise, thf, 0, DR).mean()
    assert rate == pytest.approx(th.label_fa, abs=0.01)
    # labels no longer move with the gain
    steep = dataclasses.replace(thf, gain_db=np.array([[0.0, 2.0, 0.0], [0.0, 2.0, 0.0]]))
    np.testing.assert_array_equal(Sh.block_labels(y[:5, 0], thf, 0, DR), Sh.block_labels(y[:5, 0], steep, 0, DR))


def test_lit_probability_against_the_floor():
    """With the floor, no echo gives p0, a strong echo gives 1, and the block mean is matched as simulated."""
    import dataclasses
    th = dataclasses.replace(make_theta(), floor=np.full((2, 100), 3e-7))
    p_dark = np.asarray(Sh.lit_probability(jnp.zeros(80), 0.0, th, 0, DR))
    np.testing.assert_allclose(p_dark, Sh.noise_lit_rate(th, 0, 10), atol=1e-12)
    assert np.asarray(Sh.lit_probability(jnp.full(80, 1e-3), 0.0, th, 0, DR)).min() > 0.999
    with pytest.raises(ValueError):
        Sh.lit_probability(jnp.zeros(80), 0.0, th, 0)
    rng = np.random.default_rng(8)
    st = th.at(STARBOARD)
    S = np.full(8 * 3000, 3e-8)
    g = np.exp(np.asarray(F.log_gain(F.bin_ranges(8 * 3000, DR), st.gain_db)))
    # data whose noise is the floor itself (not g N), plus the gained echo
    y = rng.gamma(th.L, (g * S + 3e-7) / th.L)
    observed = Sh.block_labels(y, th, STARBOARD, DR).mean()
    predicted = float(np.asarray(Sh.lit_probability(jnp.asarray(S), 0.0, th, STARBOARD, DR)).mean())
    assert 0.1 < predicted < 0.95  # the range where the comparison says something
    assert observed == pytest.approx(predicted, abs=0.03)


def test_gain_ratio_scales_the_predicted_block_mean():
    """A gain above the one the labels were thresholded with predicts brighter blocks: dark ones get less likely."""
    th = make_theta()
    S = jnp.asarray(np.linspace(0, 5e-7, 80))
    p1 = np.asarray(Sh.lit_probability(S, 0.0, th, STARBOARD))
    p2 = np.asarray(Sh.lit_probability(S, 0.0, th, STARBOARD, gain_ratio=jnp.full(10, 1.5)))
    assert (p2 > p1).all()
    labels = np.zeros(10, bool)
    assert float(Sh.log_shadow_side(th, STARBOARD, labels, S, 0.0, gain_ratio=jnp.full(10, 1.5))) < float(
        Sh.log_shadow_side(th, STARBOARD, labels, S, 0.0))
    # differentiating through log-kappa is what `stop=False` adds
    g = jax.grad(lambda u, stop: Sh.log_shadow_side(th, STARBOARD, labels, S, u, stop=stop))
    assert float(g(0.0, True)) == 0.0 and float(g(0.0, False)) < 0.0
