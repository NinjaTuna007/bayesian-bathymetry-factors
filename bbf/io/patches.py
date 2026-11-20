r"""Passes of side-scan pings with their poses, read from the prototype's patch cache.

The cache (``<bag>_patches.npz``) holds one or more named time windows (patches) of a bag,
each a set of per-ping arrays keyed ``<name>/<field>``: raw ``port`` and ``starboard``
amplitudes (n, B), bin 0 nearest the sonar; navigation ``x``, ``y`` (east and north of a
local origin), ``alt`` (height), ``roll``, ``pitch``, ``yaw`` (NED/FRD, yaw clockwise
from north) interpolated at ``t``, the bag's record time of each ping; the sonar's header
stamps ``stamp_ns``; and the bin spacing ``range_res``. Record times are bursty, so the
pings are retimed by their stamps (`retime`).
The cache also holds the 3DSS bathymetry (``bathy_*``); `Pass` leaves it out, so inference
cannot use it (diagnostic scripts read it themselves, `scripts/seabed_3dss.py`).
"""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Pass:
    r"""A consecutive stretch of pings: one pass of the trajectory.

    poses: (n, 6) world poses :math:`(x, y, z, \phi_r, \phi_p, \psi)`; y: (n, 2, B)
    intensities (amplitude squared), sides port and starboard; dr: bin spacing
    :math:`\Delta r` (m); t: (n,) ping times (s).
    """
    name: str
    poses: np.ndarray
    y: np.ndarray
    dr: float
    t: np.ndarray

    @property
    def n_pings(self) -> int:
        return self.poses.shape[0]

    @property
    def n_bins(self) -> int:
        return self.y.shape[-1]

    def local_poses(self, origin) -> np.ndarray:
        """Poses with x, y relative to `origin` (e.g. a mesh's), for the mesh-local frame."""
        out = self.poses.copy()
        out[:, :2] -= np.asarray(origin, dtype=float)
        return out


def retime(t, stamp_ns, poses) -> tuple[np.ndarray, np.ndarray]:
    r"""Ping times from the sonar's own stamps, and the poses re-interpolated at them.

    The cache's times ``t`` are the bag's record times, which reach the recorder in bursts
    (jitter of tenths of a second here) and, in a replayed bag, run at the replay rate. The
    sonar's header stamps are regular. They are mapped onto the record clock (which the
    navigation shares) by a least-squares line :math:`t \approx a + b\, s`, keeping the average
    alignment the cache's navigation offset was fitted to, and the poses, which are the
    navigation sampled at ``t``, are interpolated at the mapped times (yaw unwrapped).
    """
    t = np.asarray(t, dtype=float)
    s = (np.asarray(stamp_ns, dtype=np.int64) - int(stamp_ns[0])) * 1e-9
    b, a = np.polyfit(s, t, 1)
    t_new = a + b * s
    order = np.argsort(t, kind="stable")
    tt, pp = t[order], np.asarray(poses, dtype=float)[order].copy()
    pp[:, 5] = np.unwrap(pp[:, 5])
    new = np.column_stack([np.interp(t_new, tt, pp[:, k]) for k in range(pp.shape[1])])
    new[:, 5] = (new[:, 5] + np.pi) % (2 * np.pi) - np.pi
    return t_new, new


def load_patch(path, name: str, use_stamps: bool = True) -> Pass:
    """Read patch `name` of a patch cache as a `Pass`.

    use_stamps: time the pings by the sonar's header stamps (`retime`) where the cache has
    them, instead of the bag's record times.
    """
    z = np.load(path)
    f = {k.split("/", 1)[1]: z[k] for k in z.files if k.startswith(f"{name}/")}
    if not f:
        raise KeyError(f"no patch {name!r} in {path}")
    poses = np.column_stack([f["x"], f["y"], f["alt"], f["roll"], f["pitch"], f["yaw"]]).astype(float)
    t = f["t"].astype(float)
    if use_stamps and "stamp_ns" in f:
        t, poses = retime(t, f["stamp_ns"], poses)
    y = np.stack([f["port"], f["starboard"]], axis=1).astype(float) ** 2
    return Pass(name=name, poses=poses, y=y, dr=float(f["range_res"]), t=t)
