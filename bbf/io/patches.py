r"""Passes of side-scan pings with their poses, read from the prototype's patch cache.

The cache (``<bag>_patches.npz``) holds one or more named time windows (patches) of a bag,
each a set of per-ping arrays keyed ``<name>/<field>``: raw ``port`` and ``starboard``
amplitudes (n, B), bin 0 nearest the sonar; navigation ``x``, ``y`` (east and north of a
local origin), ``alt`` (height), ``roll``, ``pitch``, ``yaw`` (NED/FRD, yaw clockwise
from north) shifted onto ping time; ping times ``t``; and the bin spacing ``range_res``.
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


def load_patch(path, name: str) -> Pass:
    """Read patch `name` of a patch cache as a `Pass`."""
    z = np.load(path)
    f = {k.split("/", 1)[1]: z[k] for k in z.files if k.startswith(f"{name}/")}
    if not f:
        raise KeyError(f"no patch {name!r} in {path}")
    poses = np.column_stack([f["x"], f["y"], f["alt"], f["roll"], f["pitch"], f["yaw"]]).astype(float)
    y = np.stack([f["port"], f["starboard"]], axis=1).astype(float) ** 2
    return Pass(name=name, poses=poses, y=y, dr=float(f["range_res"]), t=f["t"].astype(float))
