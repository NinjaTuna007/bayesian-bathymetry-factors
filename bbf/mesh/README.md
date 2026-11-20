# Mesh

Placeholder. No code yet.

A triangle mesh over the survey area plus a buffer, so the prior does not set the boundary of the map. Vertices sit a fixed spacing apart. The depth at a point in the domain is the hat-function interpolation of the vertex depths, so only the triangle that contains the point contributes.

This directory will build that mesh and evaluate the interpolation. It will not score a measurement.
