# Match factor

`bbf.match`: `Reflectivity` is the prior of a log-reflectivity field on the mesh vertices, and `footprints`, `cell_sums`, `cell_residual` and `ratio_score` check on ground cells how well two passes agree.

A pass is a consecutive stretch of the trajectory. Its pings cover a strip on each side of the track. Two passes overlap where those strips overlap.

The seabed has one reflectivity at each location, whichever pass looks at it. That reflectivity is a field on the mesh, with a Matérn prior, and it multiplies the echo inside the render of `bbf.ping`. Every ping of every pass renders through the same field, so the ratio of two passes' echoes from the same ground is free of the ground's reflectivity and depends on the depths through the two viewing geometries. The reflectivity left to each ping and side is its gain.

Nothing is paired. Which ground a bin covers is decided by the render at the current depths, every bin stays in its ping factor, and any number of passes share the field. The field is an unknown of the graph next to the depths.

The cell check is a diagnostic. It sums each pass's bins over ground cells and scores the log-ratio of the two sums against the predicted one; it does not enter the posterior.

This directory holds the field's prior and the check. It does not optimise the depths.
