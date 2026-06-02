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

from scipy.fft import fft2, ifft2
from scipy.fft import fftshift
from scipy.fft import ifftshift

def forward_H(microscope: Microscope, 
              aberration: Aberration):
    
    H = microscope.compute_pupil_function(aberration = aberration, 
                                          gaussian = False) 
    return H

def forward_h(microscope: Microscope, 
              aberration: Aberration):
    
    H = forward_H(microscope = microscope, 
                  aberration = aberration)
    h = ifft2(H, workers = -1)
    return h

def forward_PSF(microscope: Microscope, 
                aberration: Aberration):
    
    h = forward_h(microscope = microscope, 
                  aberration = aberration)
    shifted_psf = np.abs(h)**(2 * microscope.N_order)
    psf = fftshift(shifted_psf)
    return psf

def circular_convolve(f: np.array, 
                      psf: np.array):

    S = fft2(ifftshift(psf))
    F = fft2(f, workers = -1)
    convolved_img = np.real(ifft2(S * F, workers = -1))
    return convolved_img

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

    first_term = np.sum(np.abs(D_stack)**2)

    numerator = np.abs(np.sum(np.conj(S_stack) * D_stack, axis=0))**2
    denominator = gamma + np.sum(np.abs(S_stack)**2, axis=0)

    J = np.real(first_term - np.sum(numerator / denominator))
    return J

def cache_D_stack(d_stack: np.array):
    D_stack = fft2(np.asarray(d_stack), axes = (-2, -1), workers = -1)
    return D_stack

def cache_Z_stack(microscope: Microscope,
                  modes_corrected: np.array):
    Z_stack = np.array([microscope.compute_phase_map(Aberration([m], [1.0])) for m in modes_corrected])
    return Z_stack


def pre_compute_cache(microscope: Microscope,
                      D_stack: np.array,
                      Z_stack: np.array,
                      a_guess: Aberration,
                      a_stack: np.array):
    H_stack = np.asarray([forward_H(microscope, a + a_guess) for a in a_stack])
    #get the iffts 
    h_stack = ifft2(H_stack, axes = (-2, -1), workers = -1)
    #get the stack of PSFS
    s_stack = np.abs(h_stack)**(2 * microscope.N_order)
    #normalize the PSFs
    #take fourier transforms
    S_stack = fft2(s_stack, axes = (-2, -1), workers = -1)

    cache = {"H_stack": H_stack.astype(np.complex64),
        "h_stack": h_stack.astype(np.complex64),
        "S_stack": S_stack.astype(np.complex64),
        "s_stack": s_stack.astype(np.float32),
        "D_stack": D_stack.astype(np.complex64),
        "a_stack": a_stack,
        "Z_stack": Z_stack.astype(np.float32)}
    
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
    N = len(D_stack)
    
    #compute the estimator
    numerator = np.sum(np.conj(S_stack) * D_stack, axis = 0)
    denominator = gamma + np.sum(np.abs(S_stack)**2, axis = 0 )
    F = numerator/denominator 
    f = ifft2(F, workers = -1)
    return np.real(f), F

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
    N = len(D_stack)

    V_stack = np.conj(F_guess) * D_stack - np.abs(F_guess)**2 * S_stack

    inner_product = np.abs(h_stack)**(2 * microscope.N_order - 2) * h_stack *  np.real(ifft2(V_stack, axes = (-2, -1), workers = -1))

    g_stack = np.imag(np.conj(H_stack) * fft2(inner_product, axes = (-2, -1), workers = -1)) 
    g = -2 * microscope.N_order * np.sum(g_stack, axis = 0) 
    #get the component along each zernike
    g_c = np.array([np.sum(g * Z) for Z in Z_stack])
    return g, g_c

def get_Q(S_stack: np.array,
          gamma: float):
    S_squared_stack = np.abs(S_stack)**2

    return gamma + np.sum(S_squared_stack, axis = 0)

def get_H_gn_phi_n(Z_n: np.array,
                   h_stack: np.array,
                   H_stack: np.array,
                   D_tilde_stack: np.array,
                   N_order: int):
    
    L = len(Z_n)
    N = len(D_tilde_stack)
    
    inner_stack = ifft2(H_stack * Z_n, axes = (-2, -1), workers = -1)
    fourier_stack = fft2(np.imag(np.abs(h_stack)**(2 * N_order - 2) * np.conj(h_stack) * inner_stack), axes = (-2, -1), workers = -1)

    total = np.zeros((L, L))
    j_idx, k_idx = np.triu_indices(N, k = 1)

    if len(j_idx) > 0:
        U_tilde_jk = D_tilde_stack[j_idx] * fourier_stack[k_idx] - D_tilde_stack[k_idx] * fourier_stack[j_idx]

        inner_term_1 = np.abs(h_stack[j_idx])**(2 * N_order - 2) * h_stack[j_idx] * ifft2(np.conj(D_tilde_stack[k_idx]) * U_tilde_jk, axes = (-2, -1), workers = -1)
        term_1 = np.conj(H_stack[j_idx]) * fft2(inner_term_1, axes = (-2, -1), workers = -1)

        inner_term_2 = np.abs(h_stack[k_idx])**(2 * N_order - 2) * h_stack[k_idx] * ifft2(np.conj(D_tilde_stack[j_idx]) * U_tilde_jk, axes = (-2, -1), workers = -1)
        term_2 = np.conj(H_stack[k_idx]) * fft2(inner_term_2, axes = (-2, -1), workers = -1)

        total += np.sum(np.imag(term_1 - term_2), axis = 0)

    return 4 * N_order**2 * total


def get_hessian(microscope: Microscope,
                cache: dict,
                gamma: float):
    
    D_stack = cache["D_stack"]
    H_stack = cache["H_stack"]
    h_stack = cache["h_stack"]
    S_stack = cache["S_stack"]
    Z_stack = cache["Z_stack"]

    n_modes = len(Z_stack)
    hessian = np.zeros((n_modes, n_modes))
    N_order = microscope.N_order

    Q = get_Q(S_stack = S_stack,
              gamma = gamma)

    D_tilde_stack = D_stack/np.sqrt(Q)
    
    for n in range(n_modes):
        H_gn_phi_n = get_H_gn_phi_n(Z_n = Z_stack[n],
                                    h_stack = h_stack,
                                    H_stack = H_stack,
                                    D_tilde_stack = D_tilde_stack,
                                    N_order = N_order)
        
        for m in range(n, n_modes):
            Z_m = Z_stack[m]
            hessian_element = np.sum(H_gn_phi_n * Z_m)
            hessian[m,n] = hessian_element
            hessian[n,m] = hessian_element
            
    return hessian

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
    except:
        print("Error: please provide all parameters (gamma, J_tol)")
        

    #initialize variables as necessary
    a_guess = EmptyAberration()
    F_guess = 0
    c_guess = 0

    #initialize logs
    c_guess_log = np.zeros((n_loops, len(modes_corrected)))
    F_guess_log = np.zeros((n_loops, microscope.grid_bfp, microscope.grid_bfp))
    J_log = np.zeros(n_loops)

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
            print(f"Loop {i + 1}: J = {np.round(J * 1e13, 4)}")
            print(f"Corrercted Zernike Mode: {modes_corrected}")
            print(f"Estimated Zernike Coefficients: {c_guess}")

        
        t = time.time()
        f_guess, F_guess = estimate_object(cache = cache,
                                           gamma = gamma)
        s = time.time()
        if debug:
            print(f"\t Estimated object in {np.round(s-t, 3)} seconds")

        t = time.time()
        g, g_c = estimate_aberration_gradient(microscope = microscope,
                                              cache = cache,
                                              F_guess = F_guess)
        s = time.time()
        if debug:
            print(f"\t Estimated gradient in {np.round(s-t, 3)} seconds")

        t = time.time()
        hessian = get_hessian(microscope = microscope,
                                cache = cache,
                                gamma = gamma)
        
        s = time.time()
        if debug:
            print(f"\t Estimated Hessian in {np.round(s-t, 3)} seconds")
        update = -1 * np.matmul(np.linalg.inv(hessian), g_c)
  
        c_guess = c_guess + update
        a_guess = Aberration(modes_corrected, c_guess)

        c_guess_log[i] = c_guess 
        F_guess_log[i] = F_guess 
        J_log[i] = J

        #termination conditions
        if i > 0:
            #J increasing condition
            J_stat = np.abs(J_log[i] - J_log[i-1])/np.abs(J_log[i-1] - J_log[0])
            if J_stat < J_tol:

                if log:
                    return c_guess, F_guess, c_guess_log, F_guess_log, J_log 
                else:
                    return c_guess, F_guess
                            


    if log:
        return c_guess, F_guess, c_guess_log, F_guess_log, J_log 
    else:
        return c_guess, F_guess
