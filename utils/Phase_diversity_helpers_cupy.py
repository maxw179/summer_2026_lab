import sys
from pathlib import Path

ROOT = Path.cwd().resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.PSF_helpers import *
from utils.Plot_helpers import *
from utils.Zernike_helpers import *
from utils.Booth_helpers import *

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import subprocess

from scipy import fft as sp_fft

try:
    import cupy as cp
    from cupyx.scipy import fft as cp_fft
    USE_CUPY = True
except ImportError:
    cp = None
    cp_fft = None
    USE_CUPY = False
    print("Failed to import cupy.")


def _xp(x=None):
    if USE_CUPY and x is not None:
        return cp.get_array_module(x)
    return cp if USE_CUPY else np


def _asarray(x, dtype=None):
    if USE_CUPY:
        return cp.asarray(x, dtype=dtype)
    return np.asarray(x, dtype=dtype)


def _asnumpy(x):
    if USE_CUPY and isinstance(x, cp.ndarray):
        return cp.asnumpy(x)
    return x


def _scalar(x):
    x = _asnumpy(x)
    if hasattr(x, "item"):
        return x.item()
    return x

def _sync():
    if USE_CUPY:
        cp.cuda.Stream.null.synchronize()


def fft2(x, *args, **kwargs):
    if USE_CUPY and isinstance(x, cp.ndarray):
        kwargs.pop("workers", None)
        return cp_fft.fft2(x, *args, **kwargs)
    return sp_fft.fft2(x, *args, **kwargs)


def ifft2(x, *args, **kwargs):
    if USE_CUPY and isinstance(x, cp.ndarray):
        kwargs.pop("workers", None)
        return cp_fft.ifft2(x, *args, **kwargs)
    return sp_fft.ifft2(x, *args, **kwargs)


def fftshift(x, *args, **kwargs):
    if USE_CUPY and isinstance(x, cp.ndarray):
        return cp_fft.fftshift(x, *args, **kwargs)
    return sp_fft.fftshift(x, *args, **kwargs)


def ifftshift(x, *args, **kwargs):
    if USE_CUPY and isinstance(x, cp.ndarray):
        return cp_fft.ifftshift(x, *args, **kwargs)
    return sp_fft.ifftshift(x, *args, **kwargs)


def forward_H(microscope: Microscope, 
              aberration: Aberration):
    
    H = microscope.compute_pupil_function(aberration = aberration, 
                                          gaussian = False) 
    return H


def forward_h(microscope: Microscope, 
              aberration: Aberration):
    
    H = forward_H(microscope = microscope, 
                  aberration = aberration)
    H = _asarray(H, dtype=_xp().complex128 if USE_CUPY else np.complex128)
    h = ifft2(H, workers = -1)
    return h


def forward_PSF(microscope: Microscope, 
                aberration: Aberration):
    
    h = forward_h(microscope = microscope, 
                  aberration = aberration)
    xp = _xp(h)
    shifted_psf = xp.abs(h)**(2 * microscope.N_order)
    psf = fftshift(shifted_psf)
    return psf


def circular_convolve(f: np.array, 
                      psf: np.array):

    f = _asarray(f, dtype=_xp().float64 if USE_CUPY else np.float64)
    psf = _asarray(psf)
    xp = _xp(f)
    S = fft2(ifftshift(psf))
    F = fft2(f, workers = -1)
    convolved_img = xp.real(ifft2(S * F, workers = -1))
    return _asnumpy(convolved_img)


def forward_image(microscope: Microscope, 
                  f: np.array, 
                  aberration: Aberration):
    
    psf = forward_PSF(microscope = microscope,
                       aberration = aberration)
    
    final_img = circular_convolve(f = f, psf = psf)
    return final_img


def get_diversity_image(microscope: Microscope, 
                        f: np.array, 
                        true_aberration: Aberration, 
                        diversity_aberration: Aberration):
    
    img = forward_image(microscope = microscope, 
                         f = f, 
                         aberration = true_aberration + diversity_aberration)
    return img


def objective(cache,
              gamma: float):

    S_stack = cache["S_stack"]
    D_stack = cache["D_stack"]
    xp = _xp(S_stack)

    gamma = xp.asarray(gamma, dtype=xp.float64)
    first_term = xp.sum(xp.abs(D_stack)**2)

    numerator = xp.abs(xp.sum(xp.conj(S_stack) * D_stack, axis=0))**2
    denominator = gamma + xp.sum(xp.abs(S_stack)**2, axis=0)

    J = xp.real(first_term - xp.sum(numerator / denominator))
    return _scalar(J)


def cache_D_stack(d_stack: np.array):
    D_stack = fft2(_asarray(d_stack, dtype=_xp().float64 if USE_CUPY else np.float64), axes = (-2, -1), workers = -1)
    return D_stack.astype(_xp(D_stack).complex128)


def cache_Z_stack(microscope: Microscope,
                  modes_corrected: np.array):
    Z_stack = _asarray([microscope.compute_phase_map(Aberration([m], [1.0])) for m in modes_corrected])
    return Z_stack.astype(_xp(Z_stack).float64)



def pre_compute_cache(microscope: Microscope,
                      D_stack: np.array,
                      Z_stack: np.array,
                      a_guess: Aberration,
                      a_stack: np.array):
    H_stack = _asarray([forward_H(microscope, a + a_guess) for a in a_stack])
    xp = _xp(H_stack)

    H_stack = H_stack.astype(xp.complex128)
    h_stack = ifft2(H_stack, axes = (-2, -1), workers = -1).astype(xp.complex128)
    s_stack = (xp.abs(h_stack)**(2 * microscope.N_order)).astype(xp.float64)
    S_stack = fft2(s_stack, axes = (-2, -1), workers = -1).astype(xp.complex128)

    N = len(D_stack)
    j_idx, k_idx = xp.triu_indices(N, k = 1)

    cache = {"H_stack": H_stack,
        "h_stack": h_stack,
        "S_stack": S_stack,
        "s_stack": s_stack,
        "D_stack": D_stack.astype(xp.complex128, copy=False),
        "a_stack": a_stack,
        "Z_stack": Z_stack.astype(xp.float64, copy=False),
        "j_idx": j_idx,
        "k_idx": k_idx}
    
    return cache


#grid = grid along which the images were taken
#d_stack = list of phase diverse images
#a_guess = guess for the inherent sample aberration
#a_stack = list of aberrations applied along each phase diverse image
#gamma = regularization parameters
def estimate_object(cache: dict,
                    gamma: float):
    

    S_stack = cache["S_stack"]
    D_stack = cache["D_stack"]
    xp = _xp(S_stack)
    gamma = xp.asarray(gamma, dtype=xp.float64)
    
    #compute the estimator
    numerator = xp.sum(xp.conj(S_stack) * D_stack, axis = 0)
    denominator = gamma + xp.sum(xp.abs(S_stack)**2, axis = 0 )
    F = numerator/denominator 
    f = ifft2(F, workers = -1)
    return xp.real(f), F


#grid = grid along which the images were taken
#d_stack = list of phase diverse images
#a_stack = list of aberrations applied along each phase diverse image
#aberration_guesss = guess for the inherent sample aberration
#modes_corrected = list of modes to correct along
#gamma = regularization parameters
def estimate_aberration_gradient(microscope: Microscope,
                                cache: dict,
                                F_guess: np.array):
    
    D_stack = cache["D_stack"]
    S_stack = cache["S_stack"]
    h_stack = cache["h_stack"]
    H_stack = cache["H_stack"]
    Z_stack = cache["Z_stack"]
    xp = _xp(S_stack)

    F_guess = xp.asarray(F_guess, dtype=xp.complex128)
    V_stack = xp.conj(F_guess) * D_stack - xp.abs(F_guess)**2 * S_stack

    h_abs_power = xp.abs(h_stack)**(2 * microscope.N_order - 2)
    inner_product = h_abs_power * h_stack * xp.real(ifft2(V_stack, axes = (-2, -1), workers = -1))

    g_stack = xp.imag(xp.conj(H_stack) * fft2(inner_product, axes = (-2, -1), workers = -1)) 
    g = (-2 * microscope.N_order * xp.sum(g_stack, axis = 0)).astype(xp.float64)
    #get the component along each zernike
    g_c = xp.einsum("yx,myx->m", g, Z_stack, optimize=True).astype(xp.float64)
    return g, g_c


def get_Q(S_stack: np.array,
          gamma: float):
    xp = _xp(S_stack)
    gamma = xp.asarray(gamma, dtype=xp.float64)
    S_squared_stack = xp.abs(S_stack)**2

    return gamma + xp.sum(S_squared_stack, axis = 0)


def get_H_gn_phi_batch(Z_batch: np.array,
                       h_stack: np.array,
                       H_stack: np.array,
                       D_tilde_stack: np.array,
                       N_order: int,
                       j_idx: np.array,
                       k_idx: np.array):
    
    xp = _xp(h_stack)
    n_batch = len(Z_batch)
    Ny, Nx = h_stack.shape[-2:]

    if len(j_idx) == 0:
        return xp.zeros((n_batch, Ny, Nx), dtype=xp.float64)
    
    h_abs_power = xp.abs(h_stack)**(2 * N_order - 2)

    inner_stack = ifft2(H_stack[:, None] * Z_batch[None], axes = (-2, -1), workers = -1)
    fourier_stack = fft2(xp.imag(h_abs_power[:, None] * xp.conj(h_stack[:, None]) * inner_stack), axes = (-2, -1), workers = -1)

    U_tilde_jk = D_tilde_stack[j_idx, None] * fourier_stack[k_idx] - D_tilde_stack[k_idx, None] * fourier_stack[j_idx]

    inner_term_1 = h_abs_power[j_idx, None] * h_stack[j_idx, None] * ifft2(xp.conj(D_tilde_stack[k_idx, None]) * U_tilde_jk, axes = (-2, -1), workers = -1)
    term_1 = xp.conj(H_stack[j_idx, None]) * fft2(inner_term_1, axes = (-2, -1), workers = -1)

    inner_term_2 = h_abs_power[k_idx, None] * h_stack[k_idx, None] * ifft2(xp.conj(D_tilde_stack[j_idx, None]) * U_tilde_jk, axes = (-2, -1), workers = -1)
    term_2 = xp.conj(H_stack[k_idx, None]) * fft2(inner_term_2, axes = (-2, -1), workers = -1)

    total = xp.sum(xp.imag(term_1 - term_2), axis = 0)
    return (4 * N_order**2 * total).astype(xp.float64)


#kept for compatibility; get_hessian now uses the batched version above
def get_H_gn_phi_n(Z_n: np.array,
                   h_stack: np.array,
                   H_stack: np.array,
                   D_tilde_stack: np.array,
                   N_order: int):
    
    xp = _xp(h_stack)
    N = len(D_tilde_stack)
    j_idx, k_idx = xp.triu_indices(N, k = 1)
    return get_H_gn_phi_batch(Z_n[None],
                              h_stack = h_stack,
                              H_stack = H_stack,
                              D_tilde_stack = D_tilde_stack,
                              N_order = N_order,
                              j_idx = j_idx,
                              k_idx = k_idx)[0]



def get_hessian(microscope: Microscope,
                cache: dict,
                gamma: float,
                hessian_batch_size: int = 4):
    
    D_stack = cache["D_stack"]
    H_stack = cache["H_stack"]
    h_stack = cache["h_stack"]
    S_stack = cache["S_stack"]
    Z_stack = cache["Z_stack"]
    j_idx = cache["j_idx"]
    k_idx = cache["k_idx"]
    xp = _xp(S_stack)

    n_modes = len(Z_stack)
    hessian_projected = xp.zeros((n_modes, n_modes), dtype=xp.float64)
    N_order = microscope.N_order

    Q = get_Q(S_stack = S_stack,
              gamma = gamma)

    D_tilde_stack = (D_stack/xp.sqrt(Q)).astype(xp.complex128)
    hessian_batch_size = max(1, int(hessian_batch_size))
    
    for n0 in range(0, n_modes, hessian_batch_size):
        n1 = min(n0 + hessian_batch_size, n_modes)
        H_gn_phi_batch = get_H_gn_phi_batch(Z_batch = Z_stack[n0:n1],
                                            h_stack = h_stack,
                                            H_stack = H_stack,
                                            D_tilde_stack = D_tilde_stack,
                                            N_order = N_order,
                                            j_idx = j_idx,
                                            k_idx = k_idx)
        hessian_projected[n0:n1] = xp.einsum("byx,myx->bm", H_gn_phi_batch, Z_stack, optimize=True)

    hessian = xp.triu(hessian_projected) + xp.triu(hessian_projected, k = 1).T
    return hessian.astype(xp.float64)


def loop_optimize(n_loops: int,
                  microscope: Microscope,
                  d_stack: np.array,
                  a_stack: np.array,
                  modes_corrected: np.array,
                  params: dict,
                  debug: bool,
                  log: bool):
    
    try:
        gamma                   = params["gamma"]
        J_tol                   = params["J_tol"]
        max_step                = params["max_step"]
    except:
        print("Error: please provide all parameters (gamma, J_tol)")
        
    #initialize variables as necessary
    a_guess = EmptyAberration()
    F_guess = 0
    c_guess = np.zeros(len(modes_corrected), dtype=np.float64)

    #initialize logs
    if log:
        c_guess_log = np.zeros((n_loops, len(modes_corrected)), dtype=np.float64) 
        F_guess_log = np.zeros((n_loops, microscope.grid_bfp, microscope.grid_bfp), dtype=np.complex128) 

    J_log = np.zeros(n_loops, dtype=np.float64) if log else np.zeros(n_loops, dtype=np.float64)

    #cache the D stack and Z stack
    D_stack = cache_D_stack(d_stack)
    Z_stack = cache_Z_stack(microscope = microscope,
                           modes_corrected = modes_corrected)


    for i in tqdm(range(n_loops)):
        cache = pre_compute_cache(microscope = microscope,
                                  D_stack = D_stack,
                                  Z_stack = Z_stack,
                                  a_guess = a_guess,
                                  a_stack = a_stack)


        J = objective(cache,
                      gamma = gamma)
        if debug:
            print(f"Loop {i + 1}: J = {J}")
            print(f"Corrercted Zernike Mode: {modes_corrected}")
            print(f"Estimated Zernike Coefficients: {c_guess}")

        t = time.time()
        f_guess, F_guess = estimate_object(cache = cache,
                                           gamma = gamma)
        _sync()
        s = time.time()
        if debug:
            print(f"\t Estimated object in {np.round(s-t, 3)} seconds")


        t = time.time()
        g, g_c = estimate_aberration_gradient(microscope = microscope,
                                              cache = cache,
                                              F_guess = F_guess)
        _sync()
        s = time.time()
        if debug:
            print(f"\t Estimated gradient in {np.round(s-t, 3)} seconds")

        t = time.time()
        hessian = get_hessian(microscope = microscope,
                              cache = cache,
                              gamma = gamma,
                              hessian_batch_size = 16)
        _sync()
        s = time.time()
        if debug:
            print(f"\t Estimated Hessian in {np.round(s-t, 3)} seconds")
        
                
        
        
        if USE_CUPY:
            update = -cp.linalg.solve(hessian, g_c)
            step_norm = np.linalg.norm(update)
            if step_norm > max_step:
                update *= max_step / step_norm
            c_guess = c_guess + cp.asnumpy(update)
        else:
            update = -np.linalg.solve(hessian, g_c)
            step_norm = np.linalg.norm(update)
            if step_norm > max_step:
                update *= max_step / step_norm
            c_guess = c_guess + update



        a_guess = Aberration(modes_corrected, c_guess)

        if log:
            c_guess_log[i] = c_guess 
            if F_guess_log is not None:
                F_guess_log[i] = _asnumpy(F_guess) 
        J_log[i] = J

        #termination conditions
        if i > 0:
            #J increasing condition
            denom = np.abs(J_log[i-1] - J_log[0])
            J_stat = np.inf if denom == 0 else np.abs(J_log[i] - J_log[i-1])/denom
            print(f"Jdiff (should be neg): {J_log[i] - J_log[i-1]}")
            if J_stat < J_tol or J_log[i] - J_log[i-1] > 0:
                if log:
                    return c_guess, _asnumpy(F_guess), c_guess_log, F_guess_log, J_log 
                else:
                    return c_guess, _asnumpy(F_guess)
                        
    if log:
        return c_guess, _asnumpy(F_guess), c_guess_log, F_guess_log, J_log 
    else:
        return c_guess, _asnumpy(F_guess)
