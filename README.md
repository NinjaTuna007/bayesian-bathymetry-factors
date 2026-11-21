# Bayesian bathymetry factors

Public placeholder for the implementation of *Bayesian Bathymetry Factors from Side-scan Sonar* (manuscript in preparation).

The manuscript is not in this repository. This tree follows the pieces the paper specifies.

Side-scan sonar records intensity against slant range. The elevation of each echo is not in the reading. The method puts a depth on every vertex of a triangle mesh and scores those depths with three factors:

| Factor | What it scores |
|---|---|
| Prior | Vertex depths, jointly, as a Gaussian Markov random field with a planar trend |
| Ping | Port and starboard intensities of one transmission, against the intensities the mesh predicts |
| Match | A log-reflectivity field on the mesh that every pass renders through, so passes over the same seabed must agree |

Poses stay outside the graph. Navigation uncertainty is accounted for after the depths are fit, rather than by adding the poses as variables.

## Layout

| Path | Holds |
|---|---|
| [`bbf/mesh/`](bbf/mesh/README.md) | Survey mesh, hat functions, depth interpolation |
| [`bbf/prior/`](bbf/prior/README.md) | Trend prior and the Matérn precision on the vertices |
| [`bbf/ping/`](bbf/ping/README.md) | Ray geometry, predicted intensity, gamma mixture, ping factor |
| [`bbf/match/`](bbf/match/README.md) | Shared reflectivity field, cross-pass check on ground cells |
| [`bbf/inference/`](bbf/inference/README.md) | Outer loop over visibility, Laplace fit, pose uncertainty |
| [`bbf/io/`](bbf/io/README.md) | Pings and a navigated trajectory in, depth and covariance out |

## Data

Set `SSS_DATA` to the directory holding the `rosbag2_*` survey bags, e.g. in an untracked `.env`:

    SSS_DATA=/path/to/bags

`.envrc` loads it under direnv. Without direnv, export it yourself.

## Status

Mesh, prior, ping, shadow and match factors implemented, with a reader for the prototype's patch cache; inference is still a placeholder. A tagged release will be the first version that runs.
