r"""Passes of side-scan pings with their poses, read from the prototype's patch cache.

The cache (``<bag>_patches.npz``) holds one or more named time windows (patches) of a bag,
each a set of per-ping arrays keyed ``<name>/<field>``: raw ``port`` and ``starboard``
amplitudes (n, B), bin 0 nearest the sonar; navigation ``x``, ``y`` (east and north of a
local origin), ``alt`` (height), ``roll``, ``pitch``, ``yaw`` (NED/FRD, yaw clockwise
from north) shifted onto ping time; ping times ``t``; the bin spacing ``range_res``; and
the 3DSS bathymetry points ``bathy_*`` in the sonar frame (forward, port, up).
"""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Pass:
    r"""A consecutive stretch of pings: one pass of the trajectory.

    poses: (n, 6) world poses :math:`(x, y, z, \phi_r, \phi_p, \psi)`; y: (n, 2, B)
    intensities (amplitude squared), sides port and starboard; dr: bin spacing
    :math:`\Delta r` (m); t: (n,) ping times (s). bathy_ping (m,) and bathy_xyz (m, 3) are
    the sonar's own bathymetry points, for diagnostics only (empty if absent).
    """
    name: str
    poses: np.ndarray
    y: np.ndarray
    dr: float
    t: np.ndarray
    bathy_ping: np.ndarray
    bathy_xyz: np.ndarray

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
    if "bathy_ping" in f:
        bathy_ping = f["bathy_ping"].astype(np.int64)
        bathy_xyz = np.column_stack([f["bathy_x"], f["bathy_y"], f["bathy_z"]]).astype(float)
    else:
        bathy_ping, bathy_xyz = np.zeros(0, np.int64), np.zeros((0, 3))
    return Pass(name=name, poses=poses, y=y, dr=float(f["range_res"]), t=f["t"].astype(float),
                bathy_ping=bathy_ping, bathy_xyz=bathy_xyz)
