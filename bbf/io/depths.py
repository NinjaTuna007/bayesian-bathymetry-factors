r"""Saving and loading vertex depths together with the mesh they live on."""
import numpy as np

from bbf import mesh as M


def save_depths(path, mesh: M.Mesh, d, **extra) -> None:
    """Write the mesh parameters, depths d (n,) and any extra arrays to an .npz file."""
    np.savez(path, h=mesh.h, x0=mesh.x0, y0=mesh.y0, n_rows=mesh.n_rows, n_cols=mesh.n_cols,
             origin=np.asarray(mesh.origin), z_ref=mesh.z_ref, d=np.asarray(d), **extra)


def load_depths(path) -> tuple[M.Mesh, np.ndarray, dict[str, np.ndarray]]:
    """Read (mesh, depths, extra arrays) written by `save_depths`."""
    z = np.load(path)
    mesh = M.Mesh(h=float(z["h"]), x0=float(z["x0"]), y0=float(z["y0"]), n_rows=int(z["n_rows"]),
                  n_cols=int(z["n_cols"]), origin=(float(z["origin"][0]), float(z["origin"][1])),
                  z_ref=float(z["z_ref"]))
    keys = {"h", "x0", "y0", "n_rows", "n_cols", "origin", "z_ref", "d"}
    return mesh, z["d"], {k: z[k] for k in z.files if k not in keys}
