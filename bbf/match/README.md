# Match factor

Placeholder. No code yet.

A pass is a consecutive stretch of the trajectory. Its pings cover a strip on each side of the track. Two passes overlap where those strips overlap. Inside the overlap, a location that both passes see becomes a matched pair of bins.

The two bins are treated as observing the same seabed reflectivity. That reflectivity is integrated out, which leaves a factor on the pair. Texture finer than a bin footprint is outside this approximation. Each bin belongs to at most one pair, so every bin enters the graph through exactly one factor.

Which pairs exist depends on the depths. The inference loop freezes the pairs while the depths move, then rebuilds them.

This directory will build the pairs and evaluate the factor. It will not optimise the depths.
