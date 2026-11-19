# Inference

Placeholder. No code yet.

The trajectory is an input, not a set of variables. The unknowns are the vertex depths and the planar-trend coefficients.

The fit alternates two steps. At the current depths, recompute which seabed each bin sees and which bins are matched. With that visibility and those pairs held fixed, maximise the log posterior of the depths and the trend. Stop when the visibility and the pairs stop changing.

The initial depths are a plane through the first-echo depths. A Laplace approximation at the mode supplies a covariance. Navigation uncertainty on the poses is folded in afterwards by a sensitivity of that mode to the trajectory, times the pose covariance. The poses themselves are not re-estimated.

This directory will run that loop. The factor definitions stay in `prior/`, `ping/`, and `match/`.
