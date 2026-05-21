from multiprocessing import Pool, cpu_count, get_context
from scipy.fft import fft2, ifft2, fftshift
import numpy as np
from pathlib import Path
import subprocess
import sys
sys.path.append('../')
sys.path.append('../utils')
import Zernike_helpers
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)


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
Gets the phase offset which goes into the exponential in the integral, neglecting external aberrations
Params:
    k: the wavefector
    x: the x-value [mm] of the point in the FFP we are evaluating the field at 
    y: the x-value [mm] of the point in the FFP we are evaluating the field at 
    z: the x-value [mm] of the point in the FFP we are evaluating the field at 
    theta: the altitudinal angle (0 at the center, increases towards the edge) of the point in the BFP to be integrated over
    phi: the azimuthal angle of the point in the BFP to be integrated over
Returns:
    abb_free_phase: the phase term in the fourier integral, neglecting any external aberrations
"""
def aberration_free_phase(k, x, y, z, theta, phi):
    abb_free_phase = \
        k * (x * np.sin(theta) * np.cos(phi)
        + y * np.sin(theta) * np.sin(phi)
        + z * np.cos(theta))
    return abb_free_phase
"""
Gets the kernel by which to propagate the wavefront in the angular spectrum approximation:
Params:
    N: the length of the grid
    d: the distance between grid elements in real sace
    k: the wavevector
    z: the distance to be propagated
Returns:
    H: the propagation kernel
"""
def prop_kernel(N, d, k, z):

    fx = np.fft.fftfreq(N, d=d)
    fy = np.fft.fftfreq(N, d=d)

    kx = 2 * np.pi * fx[:, None]
    ky = 2 * np.pi * fy[None, :]

    k_perp = np.sqrt(kx**2 + ky**2)
    kz = np.sqrt(np.maximum(k**2 - k_perp**2, 0.0))

    H  = np.exp(1j * kz * z) * (k_perp**2 <= k**2)

    return H

"""
Computes and propagates the wavefront associated with an aberrated gaussian beam
Params:
    aberration_map: a function which takes in (theta, phi) and returns a complex phase
    R_BFP: the radius of the back focal plane
    alpha: the maximum angle theta in the back focal plane
    mag: the magnification
    w_0: the Gaussian beam waist [mm]
    k: the wavevector
    prop_distance: the distance [mm] to propagate the aberrated wavefront over
Returns:
    U_xy: the propagated wavefront (aberration * gaussian beam) on an x-y grid
    dx: the physical distance between points in U_xy
"""
def propagate_wavefront(aberration_map, R_BFP, alpha, mag, w_0, k, prop_distance):
    N = 512
    dx = (2*R_BFP) / N
    x = (np.arange(N) - N/2) * dx
    y = (np.arange(N) - N/2) * dx
    x, y = np.meshgrid(x, y, indexing="ij")
    rho =   np.sqrt(x**2 + y**2) / R_BFP
    #sometimes this will complain and throw a runtime error for the points outside of the mask. however, this does not affect the calculations,
    #since these points are disregarded anyway
    theta = np.arcsin(np.sin(alpha) * rho)
    phi =   np.mod(np.arctan2(y, x), 2*np.pi)

    phase = np.exp(1j*aberration_map(theta, phi))
    gauss_amplitude = gaussian_amplitude_theta(theta, alpha, mag, w_0, R_BFP)
    U_xy = phase * gauss_amplitude
    mask = np.where(x**2 + y**2 >= R_BFP**2)
    U_xy[mask] = 0

    if prop_distance == 0:
        return U_xy, dx

    U_k = fft2(U_xy)
    H   = prop_kernel(U_xy.shape[0], dx, k, prop_distance)  
    U_xy  = ifft2(U_k * H)
    mask = np.where(x**2 + y**2 >= R_BFP**2)
    U_xy[mask] = 0
    return U_xy, dx

"""
Samples a map on an x-y grid onto a user-provided theta/phi grid.
Params:
    U_xy: the map on an x-y grid. the grid is assumed to have equal dimensions/spacing on both axes
    dx: the physical spacing between grid points in U_xy
    theta_grid: the grid of theta values to sample over
    phi_grid: the grid of phi values to sample over
    alpha: the maximum allowed value of theta
    R_BFP: the radius of the back focal plane
Returns:
    interpolated_values: the samples of the x-y grid on the theta/phi grid
"""
def sample_xy_to_theta_phi(U_xy, dx, theta_grid, phi_grid, alpha, R_BFP):
    N = U_xy.shape[0]
    #find the corresponding x/y values of the theta/phi values
    r = R_BFP * (np.sin(theta_grid) / np.sin(alpha))
    x = r * np.cos(phi_grid)
    y = r * np.sin(phi_grid)
    #convert x/y to dimensionless values, u and v
    u = x / dx + N/2
    v = y / dx + N/2
    #round u/v to the nearest indices on the grid, i0, i1, j0, j1
    i0 = np.floor(u).astype(int) % N
    j0 = np.floor(v).astype(int) % N
    i1 = (i0 + 1) % N
    j1 = (j0 + 1) % N
    #find the weights of the points at the nearest indices
    du = u - np.floor(u)
    dv = v - np.floor(v)
    #find the actual values of the points at the nearest indices
    U00 = U_xy[i0, j0]
    U10 = U_xy[i1, j0]
    U01 = U_xy[i0, j1]
    U11 = U_xy[i1, j1]
    #compute a weighted average of the four points
    interpolated_values = (1-du)*(1-dv)*U00 + du*(1-dv)*U10 + (1-du)*dv*U01 + du*dv*U11
    
    return interpolated_values

"""
Caches values of importance for the Richards-Wolf integration, to be passed to the integrator
Params:
    alpha: the maximum value of theta 
    k: the wavevector value
    mag: the magnification
    w_0: the beam waist [mm]
    R_BFP: the overall radius of the back focal plane
    prop_dsitance: the distance [mm] to propagate the aberration by before computing the PSF
    n_theta: the number of theta grid points to compute the integration over 
    n_phi: the number of phi grid points to compute the integration over 
    aberration_map: a function which takes in (theta, phi) and returns a complex phase
Returns:
    cache: a set of a values of importance for the Richards-Wolf integration
"""
def make_pupil_cache(alpha, k, f, mag, w_0, R_BFP, prop_distance,
                     n_theta=128, n_phi=128,
                     aberration_map=None):
    #define the grid of allowable thetas and phis
    theta = np.linspace(0, alpha, n_theta, endpoint=False)
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)

    #get the riemann integration element
    d_theta = theta[1] - theta[0]
    d_phi = phi[1] - phi[0]

    #get the pairs of theta, phi to integrate over
    theta_grid, phi_grid = np.meshgrid(theta, phi, indexing="ij")

    #helpful trig things for later
    cosT = np.cos(theta_grid)
    sinT = np.sin(theta_grid)

    #get the factors out in front
    pre_factor = -1j * k * f / (2 * np.pi)
    
    #get the actual wavefront. the propagation will not do anything if prop_distance == 0 (default)
    Uxy, dx = propagate_wavefront(aberration_map, R_BFP, alpha, mag, w_0, k, prop_distance)
    wavefront = sample_xy_to_theta_phi(Uxy, dx, theta_grid, phi_grid, alpha, R_BFP)

    inside_factor = np.sqrt(cosT) * sinT

    #precompute strength factors
    a_x, a_y, a_z = strength_angular(theta_grid, phi_grid)


    cache =  {
        "theta_grid": theta_grid,
        "phi_grid": phi_grid,
        "d_theta": d_theta,
        "d_phi": d_phi,
        "pre_factor": pre_factor,
        "inside_factor": inside_factor,
        "wavefront": wavefront,
        "a_x": a_x,
        "a_y": a_y,
        "a_z": a_z,
    }
    return cache

"""
Uses a set of cached values to integrate the PSF at a point x, y, z
Params:
    x: the x-value [mm] of the point in the FFP we are evaluating the field at 
    y: the x-value [mm] of the point in the FFP we are evaluating the field at 
    z: the x-value [mm] of the point in the FFP we are evaluating the field at 
    k: the wavefector
    cache: a set of precomputed values that facilitate the integration
Returns:
    E_x: the value of the electric field at (x,y,z) along the x-direction
    E_y: the value of the electric field at (x,y,z) along the y-direction
    E_x: the value of the electric field at (x,y,z) along the z-direction
"""
def E_integrate_cached(x, y, z, k, cache):
    theta_grid = cache["theta_grid"]
    phi_grid = cache["phi_grid"]
    default_phase = np.exp(1j * aberration_free_phase(k, x, y, z, theta_grid, phi_grid))
    
    # take the integral
    E_x = cache["pre_factor"] * np.sum(cache["inside_factor"] * cache["wavefront"] * cache["a_x"] * default_phase) * cache["d_theta"] * cache["d_phi"]
    E_y = cache["pre_factor"] * np.sum(cache["inside_factor"] * cache["wavefront"] * cache["a_y"] * default_phase) * cache["d_theta"] * cache["d_phi"]
    E_z = cache["pre_factor"] * np.sum(cache["inside_factor"] * cache["wavefront"] * cache["a_z"] * default_phase) * cache["d_theta"] * cache["d_phi"]
    return E_x, E_y, E_z


def _init_worker(aberration_kind, params, alpha, k, f, w_0, mag, R_BFP, theta_grid_size, prop_distance):
    global _aberration_map, _pupil_cache

    if aberration_kind == "Zernike":
        _aberration_map = Zernike_helpers.create_zernike_function(*params)
    elif aberration_kind == "Random":
        _aberration_map = Zernike_helpers.generate_random_aberration(*params)
    else:
        _aberration_map = None

    #precompute cache for each worker
    _pupil_cache = make_pupil_cache(alpha=alpha, k=k, f=f, w_0 = w_0, mag=mag, R_BFP = R_BFP,
        prop_distance = prop_distance,
        n_theta=theta_grid_size, n_phi=theta_grid_size,
        aberration_map=_aberration_map)

def row_intensity_helper(args):
    (x_i, y_array, z, k, N_order) = args

    row = np.empty_like(y_array, dtype=float)

    for idx, y_j in enumerate(y_array):
        E_x, E_y, E_z = E_integrate_cached(x_i, y_j, z, k, _pupil_cache)
        I1 = np.abs(E_x) ** 2 + np.abs(E_y) ** 2 + np.abs(E_z) ** 2
        row[idx] = I1 ** N_order

    return row

def intensity_grid_parallel(
    L_ffp, grid_ffp, x_offset, y_offset,
    alpha, k, f, mag, w_0, R_BFP,
    theta_grid_size, N_order, z = 0, prop_distance = 0,
    aberration_kind=None,
    n_procs=None,
    params=None):

    #we will always need this to construct a zernike map
    if params is not None:
        params.append(alpha)

    x = np.linspace(x_offset - L_ffp / 2, x_offset + L_ffp / 2, grid_ffp)
    y = np.linspace(y_offset - L_ffp / 2, y_offset + L_ffp / 2, grid_ffp)

    #one task per row
    tasks = [
        (x_i, y, z, k, N_order)
        for x_i in x
    ]

    if n_procs is None:
        n_procs = cpu_count()

    ctx = get_context("spawn")
    with ctx.Pool(
        processes=n_procs,
        initializer=_init_worker,
        initargs=(aberration_kind, params, alpha, k, f, mag, w_0, R_BFP, theta_grid_size, prop_distance)
    ) as pool:
        intensities_rows = pool.map(row_intensity_helper, tasks)

    #tack rows into a 2D array: shape (len(x), len(y))
    intensity_map = np.vstack(intensities_rows)

    return x, y, np.flip(intensity_map.T)

def parallel_grid_wrapper(L_ffp, grid_ffp, x_offset, y_offset, alpha, k, f, mag, w_0, R_BFP, theta_grid_size, N_order,
                          z = 0, prop_distance = 0, aberration_kind=None, output= Path(__file__).resolve().parent / "parallel_output/psf_output.npz", params=None, python_executable="python",
                          script_path = Path(__file__).resolve().parent / "RW_run_parallel.py"):
    cmd = [
        python_executable, script_path,
        "--L-ffp", str(L_ffp),
        "--grid-ffp", str(grid_ffp),
        "--x-offset", str(x_offset),
        "--y-offset", str(y_offset),
        "--alpha", str(alpha),
        "--k", str(k),
        "--f", str(f),
        "--mag", str(mag),
        "--w_0", str(w_0),
        "--R_BFP", str(R_BFP),
        "--theta-grid-size", str(theta_grid_size),
        "--N-order", str(N_order),
        "--output", output,
        "--params", str(params),
        "--z", str(z),
        "--prop_distance", str(prop_distance)
    ]

    if aberration_kind is not None:
        cmd += ["--aberration-kind", str(aberration_kind)]

    subprocess.run(cmd, check=True)

    output_files = np.load(output)
    return output_files["x"], output_files["y"], output_files["I"]