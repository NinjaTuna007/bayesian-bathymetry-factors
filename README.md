# Bayesian bathymetry factors

Public placeholder for the implementation of *Bayesian Bathymetry Factors from Side-scan Sonar* (manuscript in preparation).

The manuscript is not in this repository. This tree follows the pieces the paper specifies.

Side-scan sonar records intensity against slant range. The elevation of each echo is not in the reading. The method puts a depth on every vertex of a triangle mesh and scores those depths with three factors:

| Factor | What it scores |
|---|---|
| Prior | Vertex depths, jointly, as a Gaussian Markov random field with a planar trend |
| Ping | Port and starboard intensities of one transmission, against the intensities the mesh predicts |
| Match | Overlapping bins from two passes that see the same patch of seabed |

Poses stay outside the graph. Navigation uncertainty is accounted for after the depths are fit, rather than by adding the poses as variables.

## Layout

| Path | Holds |
|---|---|
| [`bbf/mesh/`](bbf/mesh/README.md) | Survey mesh, hat functions, depth interpolation |
| [`bbf/prior/`](bbf/prior/README.md) | Trend prior and the Matérn precision on the vertices |
| [`bbf/ping/`](bbf/ping/README.md) | Ray geometry, predicted intensity, gamma mixture, ping factor |
| [`bbf/match/`](bbf/match/README.md) | Overlap, matched bins, match factor |
| [`bbf/inference/`](bbf/inference/README.md) | Outer loop over visibility and pairs, Laplace fit, pose uncertainty |
| [`bbf/io/`](bbf/io/README.md) | Pings and a navigated trajectory in, depth and covariance out |

## Data

Set `SSS_DATA` to the directory holding the `rosbag2_*` survey bags, e.g. in an untracked `.env`:

    SSS_DATA=/path/to/bags

`.envrc` loads it under direnv. Without direnv, export it yourself.

## Status

Mesh and prior implemented; the ping and match factors, inference and io are still placeholders. A tagged release will be the first version that runs.
