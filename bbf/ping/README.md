# Ping factor

Placeholder. No code yet.

One transmission gives a port return and a starboard return. Each side is a vector of bin intensities. The mesh and the pose at that ping predict an intensity per bin: spreading, absorption, the beam pattern, and the local incidence angle on the interpolated seabed.

Measured intensity is a two-component gamma mixture. One component is speckle around the prediction. The other is a brighter spike for fish, strong reflectors, and transmit ringing. The unknown seabed reflectivity of that ping and side is integrated out, so the factor depends on the vertex depths.

This directory will evaluate that factor for one ping. It will not match bins across passes.
