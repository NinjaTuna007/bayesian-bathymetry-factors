import math

import jax.numpy as jnp
import numpy as np
import pytest

from bbf import mesh as M
from bbf.ping import first_return as FR

DEPTH, SONAR_Z = 20.0, -5.0
D = DEPTH + SONAR_Z  # sonar height above the seabed at nadir
SLOPE = 0.3  # the seabed rises by 0.3 m per metre to the east (starboard at yaw 0)
U = np.arange(0.025, 30.0, 0.05)


@pytest.fixture(scope="module")
def m():
    return M.build(-40.0, 40.0, -40.0, 40.0, h=1.0, origin=(0.0, 0.0))


def pose(pitch=0.0):
    return jnp.array([0.0, 0.0, SONAR_Z, 0.0, pitch, 0.0])


def test_nearest_range_on_a_slope_and_flat(m):
    r"""Over a slope the nearest point is up it, at :math:`D / \sqrt{1 + a^2}`; over a flat seabed it is the nadir, :math:`D`."""
    sloped = jnp.asarray(DEPTH - SLOPE * m.vertices[:, 0])
    r, ok, _, _ = FR.nearest(m, sloped, pose(), U, FR.TAU)
    assert bool(ok)
    assert float(r) == pytest.approx(D / math.sqrt(1 + SLOPE**2), abs=FR.TAU)
    r, _, _, _ = FR.nearest(m, jnp.full(m.n_vertices, DEPTH), pose(), U, FR.TAU)
    assert float(r) == pytest.approx(D, abs=FR.TAU)  # nadir from both sides: a stationary minimum again


def test_per_ping_takes_the_earlier_side():
    fr = FR.FirstReturn([np.array([[10.0, 10.4], [np.nan, 12.0], [np.nan, np.nan]])])
    np.testing.assert_allclose(fr.per_ping[0], [10.0, 12.0, np.nan])


@pytest.mark.parametrize("pitch", [0.0, 0.05])
def test_rows_match_finite_differences(m, pitch):
    """The sparse rows predict the change of the soft-min range for a small depth perturbation."""
    rng = np.random.default_rng(0)
    d = DEPTH - SLOPE * m.vertices[:, 0] + 0.3 * rng.standard_normal(m.n_vertices)
    delta = 1e-4 * rng.standard_normal(m.n_vertices)
    r1, _, vid, coef = FR.nearest(m, jnp.asarray(d), pose(pitch), U, FR.TAU)
    r2, _, _, _ = FR.nearest(m, jnp.asarray(d + delta), pose(pitch), U, FR.TAU)
    pred = float(np.sum(np.asarray(coef) * delta[np.asarray(vid)]))
    assert float(r2 - r1) == pytest.approx(pred, rel=0.05, abs=1e-8)


def test_term_gradient_and_weights(m):
    """The summed term's gradient matches finite differences; a wild first echo gets the wide weight."""
    rng = np.random.default_rng(1)
    d = DEPTH + 0.2 * rng.standard_normal(m.n_vertices)
    poses = [np.array([[x, y, SONAR_Z, 0.0, 0.0, 0.0] for x, y in rng.uniform(-10, 10, (6, 2))])]
    r0 = [np.full((6, 2), D) + 0.1 * rng.standard_normal((6, 2))]
    r0[0][2] = np.nan  # no echo found on either side
    r0[0][3] = 9.0  # a fish
    fr = FR.FirstReturn(r0)
    val, grad, A, W = FR.first_return_term(m, d, poses, fr)
    assert A.shape == (5, m.n_vertices) and W.shape == (5,)
    assert W.min() == pytest.approx(1 / fr.sigma_wide**2, rel=1e-3)  # the fish row
    assert W.max() > 0.5 / fr.sigma**2
    delta = 1e-5 * rng.standard_normal(m.n_vertices)
    v2 = FR.first_return_term(m, d + delta, poses, fr)[0]
    assert v2 - val == pytest.approx(grad @ delta, rel=0.05)


def test_pulls_a_runaway_vertex_back(m):
    """A vertex under the track raised far towards the sonar keeps a gradient pushing it back down."""
    poses = [np.array([[0.2, 0.3, SONAR_Z, 0.0, 0.0, 0.0]])]
    fr = FR.FirstReturn([np.array([[D, D]])])
    d = np.full(m.n_vertices, DEPTH)
    k = np.argmin(np.linalg.norm(m.vertices - [0.2, 0.3], axis=1))
    d[k] -= 12.0
    val, grad, _, _ = FR.first_return_term(m, d, poses, fr)
    assert val < FR.first_return_term(m, np.full(m.n_vertices, DEPTH), poses, fr)[0] - 3
    assert grad[k] > 0.3  # per metre, deeper is better: a uniform floor would leave about 0 here
