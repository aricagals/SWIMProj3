# Needed libraries
import numpy as np

# Global constants
r = 8.8131
Cv = 75.2629
N_MESH = 101
X_mesh, Y_mesh = np.meshgrid(np.linspace(-2.0, 2.0, N_MESH), np.linspace(-2.0, 2.0, N_MESH))

x0, y0 = [-0.95919093, -1.44157428]
x1, y1 = [ 0.67480179, -0.23938919]
t0 = 0.0
t1 = 0.0

# Anysotropic function
def anysotropic_FMM_fast(x0, y0, t0, x1, y1, t1, Cv, r):
    """
    Computes analytical solution to the anisotropic Eikonal equation with two sources.

    Parameters:
        x0, y0, t0: Coordinates and activation time of the first source
        x1, y1, t1: Coordinates and activation time of the second source
        Cv: conduction velocity along fiber direction
        r: anisotropy ratio = (Cv / Cv_perp)^2

    Returns:
        T: 2D array of activation times
    """
    # Cross-fiber speed
    Cv_perp = Cv / np.sqrt(r)

    # Arrival time from first source
    T0 = t0 + np.sqrt(((X_mesh - x0) / Cv)**2 + ((Y_mesh - y0) / Cv_perp)**2)

    # Arrival time from second source
    T1 = t1 + np.sqrt(((X_mesh - x1) / Cv)**2 + ((Y_mesh - y1) / Cv_perp)**2)

    # Combine both using minimum (earliest arrival)
    T = np.minimum(T0, T1)

    return T