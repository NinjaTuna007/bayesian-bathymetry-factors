# Input and output

`bbf.io`: `Pass` holds a pass of pings (poses, intensities, bin spacing); `load_patch` reads one from the prototype's patch cache. The covariance on the trajectory and the outputs are not implemented yet.

In: the port and starboard intensities of each ping, and a navigated pose per ping, with a covariance on the trajectory.

Out: a depth at each mesh vertex, the trend coefficients, and a covariance that includes the pose uncertainty.

The only reader so far is for the prototype's patch cache. Nothing in this directory implements the factors.
