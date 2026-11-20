# Match factor

`bbf.match`: `draw_overlap`, `project` and `select_pairs` choose the matched pairs at the current depths (outer loop), `matched_mask` gives the bins to leave out of each ping factor, and `log_match_factor` scores the pairs with the side offset profiled out. Predictions are the rendered unit intensities of `bbf.ping`, so the factor is differentiable in the depths.

A pass is a consecutive stretch of the trajectory. Its pings cover a strip on each side of the track. Two passes overlap where those strips overlap. Inside the overlap, a location that both passes see becomes a matched pair of bins.

The two bins are treated as observing the same seabed reflectivity. That reflectivity is integrated out, which leaves a factor on the pair. Texture finer than a bin footprint is outside this approximation. Each bin belongs to at most one pair, so every bin enters the graph through exactly one factor.

Which pairs exist depends on the depths. The inference loop freezes the pairs while the depths move, then rebuilds them.

This directory builds the pairs and evaluates the factor. It does not optimise the depths.
