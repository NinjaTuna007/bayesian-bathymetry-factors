import math

import jax.numpy as jnp
import numpy as np
import pytest

from bbf import mesh as M
from bbf.ping import first_return as FR
from bbf.ping.theta import PORT, STARBOARD

DEPTH, SONAR_Z = 20.0, -5.0
D = DEPTH + SONAR_Z  # sonar height above the seabed at nadir
SLOPE = 0.3  # the seabed rises by 0.3 m per metre to the east (starboard at yaw 0)
U = np.arange(0.025, 30.0, 0.05)


@pytest.fixture(scope="module")
def m():
    return M.build(-40.0, 40.0, -40.0, 40.0, h=1.0, origin=(0.0, 0.0))


def pose(pitch=0.0):
    return jnp.array([0.0, 0.0, SONAR_Z, 0.0, pitch, 0.0])


def test_nearest_range_on_a_slope(m):
    r"""Up the slope the nearest point is off nadir at :math:`D / \sqrt{1 + a^2}`; down the slope it is at nadir, :math:`D`."""
    d = jnp.asarray(DEPTH - SLOPE * m.vertices[:, 0])
    up, ok, _, _ = FR.nearest(m, d, pose(), STARBOARD, U, FR.TAU)
    down, _, _, _ = FR.nearest(m, d, pose(), PORT, U, FR.TAU)
    assert bool(ok)
    assert float(up) == pytest.approx(D / math.sqrt(1 + SLOPE**2), abs=FR.TAU)
    assert float(down) == pytest.approx(D, abs=2 * FR.TAU)  # a minimum at the curve's end: bias up to :math:`\tau`


@pytest.mark.parametrize("pitch", [0.0, 0.05])
def test_rows_match_finite_differences(m, pitch):
    """The sparse rows predict the change of the soft-min range for a small depth perturbation."""
    rng = np.random.default_rng(0)
    d = DEPTH - SLOPE * m.vertices[:, 0] + 0.3 * rng.standard_normal(m.n_vertices)
    delta = 1e-4 * rng.standard_normal(m.n_vertices)
    for side in (PORT, STARBOARD):
        r1, _, vid, coef = FR.nearest(m, jnp.asarray(d), pose(pitch), side, U, FR.TAU)
        r2, _, _, _ = FR.nearest(m, jnp.asarray(d + delta), pose(pitch), side, U, FR.TAU)
        pred = float(np.sum(np.asarray(coef) * delta[np.asarray(vid)]))
        assert float(r2 - r1) == pytest.approx(pred, rel=0.05, abs=1e-8)


def test_term_gradient_and_weights(m):
    """The summed term's gradient matches finite differences; a wild first echo is all but ignored."""
    rng = np.random.default_rng(1)
    d = DEPTH + 0.2 * rng.standard_normal(m.n_vertices)
    poses = [np.array([[x, y, SONAR_Z, 0.0, 0.0, 0.0] for x, y in rng.uniform(-10, 10, (6, 2))])]
    r0 = [np.full((6, 2), D) + 0.1 * rng.standard_normal((6, 2))]
    r0[0][2, 1] = np.nan  # no echo found
    r0[0][3, 0] = 4.0  # a fish
    r0[0][4, 0] = D + 1.5  # a detector error of a metre or two: the wide component
    fr = FR.FirstReturn(r0)
    val, grad, A, W = FR.first_return_term(m, d, poses, fr)
    assert A.shape == (11, m.n_vertices) and W.shape == (11,)
    assert W.min() < 1e-3 / fr.sigma**2  # the fish: the uniform component
    assert sorted(W)[1] == pytest.approx(1 / fr.sigma_wide**2, rel=0.05)  # the wide one
    assert W.max() > 0.5 / fr.sigma**2
    delta = 1e-5 * rng.standard_normal(m.n_vertices)
    v2 = FR.first_return_term(m, d + delta, poses, fr)[0]
    assert v2 - val == pytest.approx(grad @ delta, rel=0.05)


def test_pulls_a_raised_vertex_back(m):
    """A vertex under the track raised a few metres towards the sonar keeps a gradient pushing it back down."""
    poses = [np.array([[0.2, 0.3, SONAR_Z, 0.0, 0.0, 0.0]])]
    fr = FR.FirstReturn([np.array([[D, D]])])
    d = np.full(m.n_vertices, DEPTH)
    k = np.argmin(np.linalg.norm(m.vertices - [0.2, 0.3], axis=1))
    d[k] -= 3.0
    val, grad, _, _ = FR.first_return_term(m, d, poses, fr)
    assert val < FR.first_return_term(m, np.full(m.n_vertices, DEPTH), poses, fr)[0] - 3
    assert grad[k] > 0.3  # per metre, deeper is better
