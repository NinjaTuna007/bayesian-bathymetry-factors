import numpy as np
import pytest

from bbf import mesh as M
from bbf.io import Pass, load_depths, load_patch, save_depths


def write_cache(path, rng):
    n, B = 5, 7
    arrays = {}
    for name in ("A", "B"):
        arrays |= {
            f"{name}/t": np.arange(n) * 0.1,
            f"{name}/port": rng.uniform(0, 10, (n, B)).astype(np.float32),
            f"{name}/starboard": rng.uniform(0, 10, (n, B)).astype(np.float32),
            f"{name}/x": rng.normal(size=n), f"{name}/y": rng.normal(size=n), f"{name}/alt": np.full(n, 56.0),
            f"{name}/roll": np.zeros(n), f"{name}/pitch": np.zeros(n), f"{name}/yaw": np.linspace(0, 1, n),
            f"{name}/range_res": np.array(0.0132),
            f"{name}/bathy_ping": np.array([0, 0, 3], np.int32),
            f"{name}/bathy_x": np.zeros(3, np.float32), f"{name}/bathy_y": np.ones(3, np.float32),
            f"{name}/bathy_z": -np.ones(3, np.float32),
        }
    np.savez(path, **arrays)
    return arrays


def test_load_patch(tmp_path):
    raw = write_cache(tmp_path / "c.npz", np.random.default_rng(0))
    p = load_patch(tmp_path / "c.npz", "B")
    assert isinstance(p, Pass) and p.name == "B"
    assert (p.n_pings, p.n_bins) == (5, 7)
    np.testing.assert_allclose(p.y[:, 0], raw["B/port"].astype(float) ** 2)
    np.testing.assert_allclose(p.y[:, 1], raw["B/starboard"].astype(float) ** 2)
    np.testing.assert_allclose(p.poses[:, 5], raw["B/yaw"])
    assert p.dr == pytest.approx(0.0132)
    assert not hasattr(p, "bathy_xyz")  # the 3DSS bathymetry stays out of the passes
    np.testing.assert_allclose(p.local_poses((1.0, 2.0))[:, :2], p.poses[:, :2] - [1.0, 2.0])
    np.testing.assert_allclose(p.local_poses((1.0, 2.0))[:, 2:], p.poses[:, 2:])


def test_missing_patch(tmp_path):
    write_cache(tmp_path / "c.npz", np.random.default_rng(0))
    with pytest.raises(KeyError):
        load_patch(tmp_path / "c.npz", "C")


def test_depths_roundtrip(tmp_path):
    m = M.build(0.0, 10.0, 0.0, 5.0, h=1.0, z_ref=56.4)
    d = np.random.default_rng(0).normal(size=m.n_vertices)
    save_depths(tmp_path / "d.npz", m, d, beta=np.array([1.0, 2.0, 3.0]))
    m2, d2, extra = load_depths(tmp_path / "d.npz")
    assert m2 == m
    np.testing.assert_array_equal(d2, d)
    np.testing.assert_array_equal(extra["beta"], [1.0, 2.0, 3.0])
