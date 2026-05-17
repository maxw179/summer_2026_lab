from multiprocessing import Pool, cpu_count, get_context
from scipy.fft import fft2, ifft2, fftshift
import numpy as np
import subprocess
import Zernike_helpers

"""
Gets the strength of the angular component of the Richards-Wolf integral. 
Params:
    theta: the altitudinal angle (0 at the center, increases towards the edge)
    phi: the azimuthal angle
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
    theta: the altitudinal angle (0 at the center, increases towards the edge)
    alpha: the maximum angle 
    mag: the magnification
    w_0: the beam waist [mm]
    R_BFP: the overall radius of the back focal plane
"""
def gaussian_amplitude_theta(theta, alpha, mag, w_0, R_BFP):
    r_bfp = R_BFP * (np.sin(theta)/np.sin(alpha))
    return np.exp(-1 *((r_bfp / (mag * w_0))**2))

"""
Gets the phase offset which goes into the exponential in the integral, neglecting external aberrations
Params:
    k: the wavefector
    x: the x-value of the point in the FFP we are evaluating the field at 
    y: the x-value of the point in the FFP we are evaluating the field at 
    z: the x-value of the point in the FFP we are evaluating the field at 
    theta: the altitudinal angle (0 at the center, increases towards the edge) of the point in the BFP to be integrated over
    phi: the azimuthal angle of the point in the BFP to be integrated over
"""
def aberration_free_phase(k, x, y, z, theta, phi):
    return \
        k * (x * np.sin(theta) * np.cos(phi)
        + y * np.sin(theta) * np.sin(phi)
        + z * np.cos(theta))
"""
Gets the kernel by which to propagate the wavefront in the angular spectrum approximation:
Params:
    N: the length of the grid
    d: the distance between grid elements in real sace
    k: the wavevector
    z: the distance to be propagated
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
"""
def propagate_wavefront(aberration_map, R_BFP, alpha, mag, w_0, k, prop_distance):
    N = 512
    dx = (2*R_BFP) / N
    x = (np.arange(N) - N/2) * dx
    y = (np.arange(N) - N/2) * dx
    x, y = np.meshgrid(x, y, indexing="ij")
    rho =   np.sqrt(x**2 + y**2) / R_BFP
    theta = np.asin(np.sin(alpha) * rho)
    phi =   np.mod(np.arctan2(y, x), 2*np.pi)

    phase = np.exp(1j*aberration_map(theta, phi))
    gauss_amplitude = gaussian_amplitude_theta(theta, alpha, mag, w_0, R_BFP)
    U0 = phase * gauss_amplitude
    mask = np.where(x**2 + y**2 >= R_BFP**2)
    U0[mask] = 0

    if prop_distance == 0:
        return U0, dx

    U_k = fft2(U0)
    H   = prop_kernel(U0.shape[0], dx, k, prop_distance)  
    Uz  = ifft2(U_k * H)
    mask = np.where(x**2 + y**2 >= R_BFP**2)
    Uz[mask] = 0
    return Uz, dx


def sample_xy_to_theta_phi(U_xy, dx, theta_grid, phi_grid, alpha, R_BFP):

    N = U_xy.shape[0]

    r = R_BFP * (np.sin(theta_grid) / np.sin(alpha))
    x = r * np.cos(phi_grid)
    y = r * np.sin(phi_grid)

    u = x / dx + N/2
    v = y / dx + N/2

    i0 = np.floor(u).astype(int) % N
    j0 = np.floor(v).astype(int) % N
    i1 = (i0 + 1) % N
    j1 = (j0 + 1) % N

    du = u - np.floor(u)
    dv = v - np.floor(v)

    U00 = U_xy[i0, j0]
    U10 = U_xy[i1, j0]
    U01 = U_xy[i0, j1]
    U11 = U_xy[i1, j1]

    return (
        (1-du)*(1-dv)*U00 +
        du*(1-dv)*U10 +
        (1-du)*dv*U01 +
        du*dv*U11
    )

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
    #get the actual wavefront
    #gaussian_amp = gaussian_amplitude(theta_grid, alpha, mag, w_0, R_BFP)
    #precompute aberration phase on the pupil grid
    aberration_phase = 0.0
    #if aberration_map is not None:
    #    aberration_phase = aberration_map(theta_grid, phi_grid)
    #propagate the wavefront
    #wavefront = gaussian_amp * np.exp(1j * aberration_phase)
    Uxy, dx = propagate_wavefront(aberration_map, R_BFP, alpha, mag, w_0, k, prop_distance)
    wavefront = sample_xy_to_theta_phi(Uxy, dx, theta_grid, phi_grid, alpha, R_BFP)

    inside_factor = np.sqrt(cosT) * sinT

    #precompute strength factors
    a_x, a_y, a_z = strength_angular(theta_grid, phi_grid)


    return {
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
        "aberration_phase": aberration_phase,
    }


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
    L_ffp, grid_ffp,
    alpha, k, f, mag, w_0, R_BFP,
    theta_grid_size, N_order, z = 0, prop_distance = 0,
    aberration_kind=None,
    n_procs=None,
    params=None):

    #we will always need this to construct a zernike map
    if params is not None:
        params.append(alpha)

    x = np.linspace(-L_ffp / 2, L_ffp / 2, grid_ffp)
    y = np.linspace(-L_ffp / 2, L_ffp / 2, grid_ffp)

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

def parallel_grid_wrapper(L_ffp, grid_ffp, alpha, k, f, mag, w_0, R_BFP, theta_grid_size, N_order,
                          z = 0, prop_distance = 0, aberration_kind=None, output="psf_output.npz", params=None, python_executable="python",
                          script_path="RW_run_parallel.py"):
    cmd = [
        python_executable, script_path,
        "--L-ffp", str(L_ffp),
        "--grid-ffp", str(grid_ffp),
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


# def intensity_grid(L_ffp, grid_ffp, alpha, k, f, w_0, mag, R_BFP, N_order, theta_grid_size, aberration_map=None):
#     x = np.linspace(-L_ffp / 2, L_ffp / 2, grid_ffp)
#     y = np.linspace(-L_ffp / 2, L_ffp / 2, grid_ffp)
#     intensity_map = np.zeros((len(x), len(y)))

#     #build cache once for the worker
#     cache = make_pupil_cache(
#         alpha=alpha, k=k, f=f, w_0 = w_0, mag=mag, R_BFP = R_BFP,
#         n_theta=theta_grid_size, n_phi=theta_grid_size,
#         aberration_map=aberration_map
#     )

#     for i in range(len(x)):
#         for j in range(len(y)):
#             x_p = x[i]
#             y_p = y[j]
#             #get the integrated electric field (with cached pupil)
#             E_x, E_y, E_z = E_integrate_cached(x_p, y_p, 0.0, k, cache)
#             I1 = np.abs(E_x) ** 2 + np.abs(E_y) ** 2 + np.abs(E_z) ** 2
#             I_np = I1 ** N_order
#             intensity_map[i, j] = I_np

#     return x, y, np.flip(intensity_map.T)