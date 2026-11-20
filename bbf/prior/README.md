# Prior

Placeholder. No code yet.

The vertex depths are a Gaussian Markov random field: a planar trend plus a Matérn field written in the hat basis (Lindgren, Rue, and Lindström, 2011). The trend coefficients are one Gaussian factor. The depths, given the trend, are another, with a sparse precision that couples neighbouring vertices.

The mesh spacing and the Matérn range and scale are fixed. They are not variables in the graph.

This directory will assemble those two factors. It will not read a ping.
