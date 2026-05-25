import numpy as np
import sys
sys.path.append('../')
sys.path.append('../utils')
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

"""
Gets the amplitude of the Gaussian beam across the back focal plane
Params:
    theta: the set of altitudinal angles (0 at the center, increases towards the edge) to compute the amplitude of the beam over
    alpha: the maximum value of theta 
    mag: the magnification
    w_0: the beam waist [mm]
    R_BFP: the overall radius of the back focal plane
Returns:
    gaussian_amplitude: the amplitude of the gaussian beam at each value of theta
"""
def gaussian_amplitude_theta(theta, alpha, mag, w_0, R_BFP):
    r_bfp = R_BFP * (np.sin(theta)/np.sin(alpha))
    gaussian_amplitude = np.exp(-1 *((r_bfp / (mag * w_0))**2))
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

def get_bfp_grid(R_BFP, grid_bfp):
    dxy_bfp = (2 * R_BFP) / grid_bfp
    x_bfp_1d = (np.arange(grid_bfp) - grid_bfp / 2) * dxy_bfp
    y_bfp_1d = (np.arange(grid_bfp) - grid_bfp / 2) * dxy_bfp
    x_bfp, y_bfp = np.meshgrid(x_bfp_1d, y_bfp_1d, indexing="ij")
    return dxy_bfp, x_bfp, y_bfp

def bfp_coord_convert(R_BFP, alpha, x_bfp, y_bfp):
    sx = (x_bfp / R_BFP) * np.sin(alpha)
    sy = (y_bfp / R_BFP) * np.sin(alpha)

    s_perp2 = sx**2 + sy**2
    s_max2 = np.sin(alpha)**2

    mask = s_perp2 <= s_max2

    sz = np.zeros_like(sx)
    sz[mask] = np.sqrt(1.0 - s_perp2[mask])

    theta = np.zeros_like(sx)
    theta[mask] = np.arccos(sz[mask])

    phi = np.mod(np.arctan2(sy, sx), 2 * np.pi)
    return mask, theta, phi, sx, sy, sz


def rw_fast(L_ffp_x, L_ffp_y, grid_ffp_x, grid_ffp_y, x_offset, y_offset,
    alpha, k, f, mag, w_0, R_BFP, aberration,
    grid_bfp, N_order, z):

    z_map = aberration.construct_map(alpha)

    #BFP grid stays square/circular
    dxy_bfp, x_bfp, y_bfp = get_bfp_grid(R_BFP, grid_bfp)

    mask, theta, phi, sx, sy, sz = bfp_coord_convert(
        R_BFP, alpha, x_bfp, y_bfp
    )

    #build the BFP field U
    U = np.zeros_like(x_bfp, dtype=complex)

    gauss = gaussian_amplitude_theta(
        theta[mask],
        alpha,
        mag,
        w_0,
        R_BFP,
    )

    phase = np.exp(1j * z_map(theta[mask], phi[mask]))
    U[mask] = gauss * phase

    #angular strength factors
    a_x, a_y, a_z = strength_angular(theta, phi)

    #z propagation phase
    z_phase = np.ones_like(U, dtype=complex)
    z_phase[mask] = np.exp(1j * k * z * sz[mask])

    #set up pupil integrands
    P_x = np.zeros_like(U, dtype=complex)
    P_y = np.zeros_like(U, dtype=complex)
    P_z = np.zeros_like(U, dtype=complex)

    inv_sqrt_sz = np.zeros_like(sz)
    inv_sqrt_sz[mask] = 1.0 / np.sqrt(sz[mask])

    P_x[mask] = U[mask] * a_x[mask] * inv_sqrt_sz[mask] * z_phase[mask]
    P_y[mask] = U[mask] * a_y[mask] * inv_sqrt_sz[mask] * z_phase[mask]
    P_z[mask] = U[mask] * a_z[mask] * inv_sqrt_sz[mask] * z_phase[mask]

    #rectangular FFP grid
    dx_ffp = L_ffp_x / grid_ffp_x
    dy_ffp = L_ffp_y / grid_ffp_y

    x_ffp = x_offset + (np.arange(grid_ffp_x) - grid_ffp_x / 2) * dx_ffp
    y_ffp = y_offset + (np.arange(grid_ffp_y) - grid_ffp_y / 2) * dy_ffp

    #BFP angular coordinate axes
    sx_1d = sx[:, 0]   # length grid_bfp
    sy_1d = sy[0, :]   # length grid_bfp

    #Fourier/Richards-Wolf phase factors
    Ax = np.exp(1j * k * np.outer(x_ffp, sx_1d))
    Ay = np.exp(1j * k * np.outer(sy_1d, y_ffp))

    #prefactors
    C = -1j * k * f / (2 * np.pi)

    ds_xy = (np.sin(alpha) / R_BFP) * dxy_bfp
    scale = C * ds_xy**2

    # ompute integral
    E_x = scale * (Ax @ P_x @ Ay)
    E_y = scale * (Ax @ P_y @ Ay)
    E_z = scale * (Ax @ P_z @ Ay)

    #intensity
    I1 = np.abs(E_x)**2 + np.abs(E_y)**2 + np.abs(E_z)**2
    I = I1**N_order

    return x_ffp, y_ffp, np.flip(I.T)