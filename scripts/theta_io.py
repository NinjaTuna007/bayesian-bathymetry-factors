r"""The sonar parameters of a run as arrays in its saved state (`bbf.io.save_depths`), and back."""
import numpy as np

from bbf.ping import Theta

FIELDS = ("noise", "L", "eps", "eta", "zeta", "ell", "log_kappa_mean", "log_kappa_std", "gain_db", "beta0", "omega",
          "alpha", "block", "label_fa", "noise_slope", "noise_power")


def theta_arrays(theta: Theta) -> dict[str, np.ndarray]:
    """theta's fields as arrays keyed `theta_<field>` (the measured water-column lit rate included when set)."""
    out = {f"theta_{k}": np.asarray(getattr(theta, k)) for k in FIELDS}
    if theta.wc_lit is not None:
        out["theta_wc_lit"] = np.asarray(theta.wc_lit)
    return out


def theta_from(extra: dict) -> Theta:
    """The theta that `theta_arrays` saved, from a loaded state's extra arrays."""
    kw = {k: extra[f"theta_{k}"] for k in FIELDS}
    for k in ("L", "alpha", "label_fa", "noise_slope", "noise_power"):
        kw[k] = float(kw[k])
    kw["block"] = int(kw["block"])
    return Theta(**kw, wc_lit=extra.get("theta_wc_lit"))
