r"""Offset between passes from their textures (`bbf.match.register`)."""
import numpy as np
import pytest

from bbf.io import Pass
from bbf.match import register as Rg

N_BINS, DR = 300, 0.15
Z_SEABED, SONAR_Z = -20.0, 0.0


_RNG = np.random.default_rng(42)
_K = _RNG.normal(0.0, 2.0, (40, 2))  # wave vectors (rad/m): structure at 1 to 3 m, no common period
_PHASE = _RNG.uniform(0, 2 * np.pi, 40)


def ground_texture(q):
    """A fixed log-reflectivity pattern of the ground: a sum of 40 plane waves, standard deviation about 0.7."""
    return np.sqrt(2.0 / 40) * 0.7 * np.cos(q @ _K.T + _PHASE).sum(axis=-1)


def a_pass(name, start, heading, n, rng, moved=(0.0, 0.0), pitch=0.0):
    """Pings along a straight track over the textured level seabed; the recorded positions are the true ones plus `moved`."""
    fwd = np.array([np.sin(heading), np.cos(heading)])
    xy = np.asarray(start) + np.arange(n)[:, None] * 0.25 * fwd
    poses = np.column_stack([xy, np.full(n, SONAR_Z), np.zeros(n), np.full(n, pitch), np.full(n, heading)])
    true = Pass(name=name, poses=poses, y=np.ones((n, 2, N_BINS)), dr=DR, t=np.arange(n) * 0.1)
    q = Rg.level_footprints(true, Z_SEABED)
    r = np.arange(N_BINS) * DR
    mean = np.where(np.isfinite(q).all(axis=-1), np.exp(ground_texture(np.nan_to_num(q))) * (1.0 + 30.0 / np.maximum(r, 1.0)), 0.01)
    y = rng.gamma(2.0, mean / 2.0)
    rec = poses.copy()
    rec[:, :2] += moved
    return Pass(name=name, poses=rec, y=y, dr=DR, t=true.t)


def test_level_footprints_on_a_level_ping():
    ps = Pass(name="p", poses=np.array([[1.0, 2.0, 0.0, 0.0, 0.0, 0.0]]), y=np.ones((1, 2, N_BINS)), dr=DR, t=np.zeros(1))
    q = Rg.level_footprints(ps, Z_SEABED)
    r = np.arange(N_BINS) * DR
    reach = r >= 20.0
    assert np.isnan(q[0, :, ~reach]).all() and np.isfinite(q[0, :, reach]).all()
    x = np.sqrt(r[reach] ** 2 - 20.0**2)
    # heading north: port is west, starboard east, both abeam
    np.testing.assert_allclose(q[0, 0, reach], np.column_stack([1.0 - x, np.full(x.shape, 2.0)]), atol=1e-9)
    np.testing.assert_allclose(q[0, 1, reach], np.column_stack([1.0 + x, np.full(x.shape, 2.0)]), atol=1e-9)
    # nose up by 0.1 rad: the ping plane meets the seabed 20 tan(0.1) m ahead... the sonar's forward axis tilts up, so astern
    up = Pass(name="p", poses=np.array([[1.0, 2.0, 0.0, 0.0, 0.1, 0.0]]), y=ps.y, dr=DR, t=np.zeros(1))
    qu = Rg.level_footprints(up, Z_SEABED)
    ok = np.isfinite(qu[0, 0, :, 1])
    np.testing.assert_allclose(np.abs(qu[0, 0, ok, 1] - 2.0), 20.0 * np.tan(0.1), atol=1e-9)


def test_move_surface_puts_a_known_shift_at_its_entry():
    rng = np.random.default_rng(0)
    b = rng.standard_normal((60, 50))
    a = np.full_like(b, np.nan)
    a[3:, :-2] = b[:-3, 2:]  # a(x) = b(x - (3, -2)): b moved by (3, -2) cells gives a
    surf = Rg.move_surface(a, b, max_cells=5, min_cells=100)
    i, j = np.unravel_index(np.nanargmax(surf), surf.shape)
    assert (i - 5, j - 5) == (3, -2) and surf[i, j] == pytest.approx(1.0)


@pytest.mark.parametrize("moved", [(-2.5, 0.5), (1.5, -1.0)])
def test_offset_between_recovers_a_navigation_offset(moved):
    """Two passes on opposite headings over the same ground; the second's recorded positions are off by `moved`."""
    rng = np.random.default_rng(1)
    a = a_pass("a", (-30.0, -8.0), np.pi / 2, 240, rng)                       # east, looking north on port
    b = a_pass("b", (30.0, 8.0), -np.pi / 2, 240, rng, moved=moved)           # west, looking north on starboard
    r0 = np.full((240, 2), 20.0)
    off = Rg.offset_between(a, b, r0, r0, Z_SEABED, cell=0.5, max_move=4.0, min_cells=300)
    assert off is not None
    # the move to give b undoes the error of its positions
    np.testing.assert_allclose(off.move, -np.asarray(moved), atol=0.3)
    assert off.corr > 0.3 and off.corr > off.corr_zero + 0.2 and off.z > 5
    assert off.sides in ((Rg.PORT, Rg.STARBOARD), (Rg.STARBOARD, Rg.PORT))
    fixed = Rg.offset_between(a, b.shifted(*off.move), r0, r0, Z_SEABED, cell=0.5, max_move=4.0, min_cells=300)
    assert fixed is not None and np.abs(fixed.move).max() < 0.3


def test_same_look_pairs_and_no_overlap():
    rng = np.random.default_rng(2)
    a = a_pass("a", (-10.0, 0.0), np.pi / 2, 40, rng)
    b = a_pass("b", (-10.0, 5.0), np.pi / 2, 40, rng)
    assert Rg.same_look_pairs(a, b) == [(0, 0), (1, 1)]
    far = a_pass("c", (500.0, 500.0), -np.pi / 2, 40, rng)
    assert Rg.same_look_pairs(a, far) == [(0, 1), (1, 0)]
    r0 = np.full((40, 2), 20.0)
    assert Rg.offset_between(a, far, r0, r0, Z_SEABED) is None
