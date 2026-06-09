import jax
import jax.numpy as np
import sys
sys.path.append('../')
sys.path.append('../utils')
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)


"""
Gets the amplitude of the Gaussian beam across the back focal plane
Params:
    mag: the magnification
    w_0: the beam waist [mm]
Returns:
    gaussian_amplitude: the amplitude of the gaussian beam at each value of s_perp
"""
def gaussian_amplitude_s_perp(mag, w_0, f, n, s_perp):
    s_waist = (mag * w_0) / (f * n)
    gaussian_amplitude = np.exp(-1 *((s_perp / s_waist)**2))
    return gaussian_amplitude


"""
Gets the strength of the angular component of the Richards-Wolf integral. 
Params:
    theta: the altitudinal angle (0 at the center, increases towards the edge)
    phi: the azimuthal angle
Returns:
    a_x: the x-component of the angular strength
    a_y: the y-component of the angular strength
    a_z: the z-component of the angular strength
"""
def strength_angular(theta, phi):
    cosT = np.cos(theta)
    sinT = np.sin(theta)
    cosP = np.cos(phi)
    sinP = np.sin(phi)

    a_x = cosT + (1 - cosT) * sinP**2
    a_y = (cosT - 1) * cosP * sinP
    a_z = -sinT * cosP

    return a_x, a_y, a_z


def get_bfp_grid(L_bfp, grid_bfp):
    dxy_bfp = L_bfp / grid_bfp
    x_bfp_1d = (np.arange(grid_bfp) - grid_bfp / 2) * dxy_bfp
    y_bfp_1d = (np.arange(grid_bfp) - grid_bfp / 2) * dxy_bfp
    x_bfp, y_bfp = np.meshgrid(x_bfp_1d, y_bfp_1d, indexing="ij")
    return dxy_bfp, x_bfp, y_bfp


def bfp_coord_convert(f, n, alpha, x_bfp, y_bfp):
    sx = x_bfp / (f * n)
    sy = y_bfp / (f * n)

    s_perp2 = sx**2 + sy**2
    s_max2 = np.sin(alpha)**2

    mask = s_perp2 <= s_max2

    #avoid masked assignment.
    #outside the pupil, set sz = 1 and theta = 0.
    sz_inside = np.sqrt(np.maximum(1.0 - s_perp2, 0.0))
    sz = np.where(mask, sz_inside, 1.0)

    theta = np.where(mask, np.arccos(sz), 0.0)
    phi = np.mod(np.arctan2(sy, sx), 2 * np.pi)

    return mask, theta, phi, sx, sy, sz


def make_rw_cache(
    L_ffp_x, L_ffp_y,
    grid_ffp_x, grid_ffp_y,
    x_offset, y_offset,
    alpha, k, f, n, L_bfp,
    grid_bfp
):
    """
    Precomputes all geometry-dependent quantities for rw_fast_jax_cached.

    Call this once if these parameters are fixed:
        L_ffp_x, L_ffp_y
        grid_ffp_x, grid_ffp_y
        x_offset, y_offset
        alpha, k, f, n, L_bfp
        grid_bfp
    """

    #BFP grid is a square with length L_bfp
    dxy_bfp, x_bfp, y_bfp = get_bfp_grid(L_bfp, grid_bfp)

    mask, theta, phi, sx, sy, sz = bfp_coord_convert(
        f, n, alpha, x_bfp, y_bfp
    )

    #angular strength factors
    a_x, a_y, a_z = strength_angular(theta, phi)

    #rectangular FFP grid
    dx_ffp = L_ffp_x / grid_ffp_x
    dy_ffp = L_ffp_y / grid_ffp_y

    x_ffp = x_offset + (np.arange(grid_ffp_x) - grid_ffp_x / 2) * dx_ffp
    y_ffp = y_offset + (np.arange(grid_ffp_y) - grid_ffp_y / 2) * dy_ffp

    #BFP angular coordinate axes
    sx_1d = sx[:, 0]
    sy_1d = sy[0, :]

    #cache these expensive phase matrices
    Ax = np.exp(1j * k * np.outer(x_ffp, sx_1d))
    Ay = np.exp(1j * k * np.outer(sy_1d, y_ffp))

    return {
        "dxy_bfp": dxy_bfp,
        "theta": theta,
        "phi": phi,
        "sx": sx,
        "sy": sy,
        "sz": sz,
        "mask": mask,
        "a_x": a_x,
        "a_y": a_y,
        "a_z": a_z,
        "x_ffp": x_ffp,
        "y_ffp": y_ffp,
        "Ax": Ax,
        "Ay": Ay,
    }


@jax.jit
def rw_kernel_cached(
    theta, sx, sy, sz, mask,
    a_x, a_y, a_z,
    Ax, Ay,
    phase_aberration,
    alpha, k, f, n, mag, w_0, L_bfp,
    dxy_bfp, N_order, z
):
    """
    JIT-compiled numerical kernel.

    This assumes all geometry and Ax/Ay matrices have already been cached.
    """

    # Gaussian BFP amplitude

    gauss = gaussian_amplitude_s_perp(
        mag,
        w_0,
        f,
        n,
        np.sqrt(sx**2 + sy**2),
    )

    #complex BFP field
    U = gauss * np.exp(1j * phase_aberration)
    
    U = np.where(mask, U, 0.0 + 0.0j)

    #safe sz outside pupil
    safe_sz = np.where(mask, sz, 1.0)

    #z propagation phase
    z_phase = np.exp(1j * k * z * safe_sz)

    #apodization / RW factor
    inv_sqrt_sz = 1.0 / np.sqrt(safe_sz)

    #pupil integrands
    P_x = np.where(mask, U * a_x * inv_sqrt_sz * z_phase, 0.0 + 0.0j)
    P_y = np.where(mask, U * a_y * inv_sqrt_sz * z_phase, 0.0 + 0.0j)
    P_z = np.where(mask, U * a_z * inv_sqrt_sz * z_phase, 0.0 + 0.0j)

    #prefactors
    C = -1j * k * f / (2 * np.pi)

    dsx = dxy_bfp / (f * n)
    dsy = dxy_bfp / (f * n)
    scale = C * dsx * dsy

    #compute RW integral
    E_x = scale * (Ax @ P_x @ Ay)
    E_y = scale * (Ax @ P_y @ Ay)
    E_z = scale * (Ax @ P_z @ Ay)

    #intensity
    I1 = np.abs(E_x)**2 + np.abs(E_y)**2 + np.abs(E_z)**2
    I = I1**N_order

    return np.flip(I.T)


def rw_fast_jax_cached(
    cache,
    alpha, k, f, n, mag, w_0, L_bfp,
    aberration,
    N_order,
    z
):
    """
    Cached JAX version of rw_fast.

    Use this after calling make_rw_cache(...).

    Same conceptual output as rw_fast:
        x_ffp, y_ffp, I
    """

    z_map = aberration.construct_map(alpha)

    theta = cache["theta"]
    phi = cache["phi"]
    mask = cache["mask"]

    #aberration phase over the full BFP grid
    phase_aberration = z_map(theta, phi)
    phase_aberration = np.where(mask, phase_aberration, 0.0)

    I = rw_kernel_cached(
        cache["theta"],
        cache["sx"],
        cache["sy"],
        cache["sz"],
        cache["mask"],
        cache["a_x"],
        cache["a_y"],
        cache["a_z"],
        cache["Ax"],
        cache["Ay"],
        phase_aberration,
        alpha, k, f, n, mag, w_0, L_bfp,
        cache["dxy_bfp"],
        N_order,
        z,
    )

    return cache["x_ffp"], cache["y_ffp"], I