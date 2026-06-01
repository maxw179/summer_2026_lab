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

from scipy.fft import fft2
from scipy.fft import ifft2
from scipy.fft import fftshift
from scipy.fft import ifftshift

def forward_H(microscope: Microscope, 
              aberration: Aberration):
    
    H = microscope.compute_pupil_function(aberration = aberration, 
                                          gaussian = True) 
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

def objective(microscope: Microscope, 
              d_stack: np.array, 
              a_guess: Aberration, 
              a_stack: np.array, 
              gamma: float):
    
    H_stack = [forward_H(microscope, a_guess + a) for a in a_stack]
    h_stack = [ifft2(H, workers = -1) for H in H_stack]
    s_stack = [np.abs(h)**(2 * microscope.N_order) for h in h_stack]

    S_stack = np.stack([fft2(s, workers = -1) for s in s_stack], axis=0)
    D_stack = np.stack([fft2(d, workers = -1) for d in d_stack], axis=0)

    first_term = np.sum(np.abs(D_stack)**2)

    numerator = np.abs(np.sum(np.conj(S_stack) * D_stack, axis=0))**2
    denominator = gamma + np.sum(np.abs(S_stack)**2, axis=0)

    J = np.real(first_term - np.sum(numerator / denominator))
    return J

#grid = grid along which the images were taken
#d_stack = list of phase diverse images
#a_guess = guess for the inherent sample aberration
#a_stack = list of aberrations applied along each phase diverse image
#gamma = regularization parameters
def estimate_object(microscope: Microscope,
                    d_stack: np.array,
                    a_guess: Aberration,
                    a_stack: np.array,
                    gamma: float):
    N = len(d_stack)
    #get the pupil functions
    H_stack = [forward_H(microscope, a + a_guess) for a in a_stack]
    #get the iffts 
    h_stack = [ifft2(H, workers = -1) for H in H_stack]
    #get the stack of PSFS
    s_stack = [np.abs(h)**(2 * microscope.N_order) for h in h_stack]
    #normalize the PSFs
    #take fourier transforms
    S_stack = [fft2(s, workers = -1) for s in s_stack]
    D_stack = [fft2(d, workers = -1) for d in d_stack]
    #compute the estimator
    numerator = np.sum([np.conj(S_stack[j]) * D_stack[j] for j in range(N)], axis = 0)
    denominator = gamma + np.sum([np.abs(S_stack[j])**2 for j in range(N)], axis = 0 )
    F = numerator/denominator 
    f = ifft2(F)
    return np.real(f), F

#grid = grid along which the images were taken
#d_stack = list of phase diverse images
#a_stack = list of aberrations applied along each phase diverse image
#aberration_guesss = guess for the inherent sample aberration
#modes_corrected = list of modes to correct along
#gamma = regularization parameters
def estimate_aberration_gradient(microscope: Microscope,
                    d_stack: np.array,
                    a_guess: Aberration,
                    a_stack: np.array,
                    F_guess: np.array,
                    modes_corrected: np.array):
    N = len(d_stack)
    
    D_stack = [fft2(d) for d in d_stack]

    #get the pupil functions
    H_stack = [forward_H(microscope, a + a_guess) for a in a_stack]
    #get the iffts 
    h_stack = [ifft2(H, workers = -1) for H in H_stack]
    #get the stack of PSFS
    s_stack = [np.abs(h)**(2 * microscope.N_order) for h in h_stack]
    #normalize the PSFs
    #take fourier transforms
    S_stack = [fft2(s, workers = -1) for s in s_stack]

    V_stack = [np.conj(F_guess) * D_stack[i] - np.abs(F_guess)**2 * S_stack[i] for i in range(N)]

    inner_product = [np.abs(h_stack[i])**(2 * microscope.N_order - 2) * h_stack[i] *  np.real(ifft2(V_stack[i], workers = -1)) for i in range(N)]

    g_stack = [np.imag(np.conj(H_stack[i]) * fft2(inner_product[i], workers = -1)) for i in range(N) ] 
    g = -2 * microscope.N_order * np.sum(g_stack, axis = 0) 
    Z_stack = [microscope.compute_phase_map(Aberration([m], [1.0])) for m in modes_corrected]
    #get the component along each zernike
    g_c = np.array([np.sum(g * Z) for Z in Z_stack])
    return g, g_c

def get_Q(h_stack: np.array,
          gamma: float,
          N_order: int):
    
    s_stack = [np.abs(h)**(2 * N_order) for h in h_stack]
    #normalize the PSFs
    #take fourier transforms
    S_squared_stack = [np.abs(fft2(s, workers = -1))**2 for s in s_stack]

    return gamma + np.sum(S_squared_stack, axis = 0)

#indexed by j and k
def get_U_tilde(D_tilde_stack: np.array,
                   h_stack: np.array,
                   H_stack: np.array,
                   Z_n: np.array,
                   N_order: int):
    
    N = len(D_tilde_stack)
    L = len(Z_n)

    #precompute U_tilde_jk terms
    inner_stack = [ifft2(H * Z_n, workers = -1) for H in H_stack]
    fourier_stack = [fft2(np.imag(np.abs(h_stack[i])**(2 * N_order - 2) * np.conj(h_stack[i]) * inner_stack[i]), workers = -1) for i in range (N)]

    U_tilde = np.zeros((N, N, L, L), dtype=complex)
    for j in range(N):
        for k in range(N):
            U_tilde[j,k] = D_tilde_stack[j] * fourier_stack[k] - D_tilde_stack[k] * fourier_stack[j]
            

    return U_tilde

def get_H_gn_phi_n(Z_n: np.array,
                   h_stack: np.array,
                   H_stack: np.array,
                   D_tilde_stack: np.array,
                   U_tilde: np.array,
                   N_order: int):
    
    L = len(Z_n)
    N = len(D_tilde_stack)
    
    total = np.zeros((L, L))
    for k in range(N):
        for j in range(k):

            inner_term_1 = np.abs(h_stack[j])**(2 * N_order - 2) * h_stack[j] * ifft2(np.conj(D_tilde_stack[k]) * U_tilde[j,k], workers = -1)
            term_1 = np.conj(H_stack[j]) * fft2(inner_term_1, workers = -1)

            inner_term_2 = np.abs(h_stack[k])**(2 * N_order - 2) * h_stack[k] * ifft2(np.conj(D_tilde_stack[j]) * U_tilde[j,k], workers = -1)
            term_2 = np.conj(H_stack[k]) * fft2(inner_term_2, workers = -1)

            total += np.imag(term_1 - term_2)

    return 4 * N_order**2 * total


def get_hessian(microscope: Microscope,
                d_stack: np.array,
                a_guess: Aberration,
                a_stack: np.array,
                modes_corrected: np.array,
                gamma: float):
    

    n_modes = len(modes_corrected)
    hessian = np.zeros((n_modes, n_modes))

    N_order = microscope.N_order

    #fourier transform the images
    D_stack = np.array([fft2(d, workers = -1) for d in d_stack])
    #get the pupil functions
    H_stack = [forward_H(microscope, a + a_guess) for a in a_stack]
    #get the iffts 
    h_stack = [ifft2(H, workers = -1) for H in H_stack]
    #get Q
    Q = get_Q(h_stack = h_stack, 
              gamma = gamma,
              N_order = N_order)
    #compute D_tildes
    D_tilde_stack = D_stack/np.sqrt(Q)
    Z_stack = [microscope.compute_phase_map(Aberration([m], [1.0])) for m in modes_corrected]
    U_tilde_stack = [get_U_tilde(D_tilde_stack = D_tilde_stack, 
                                 h_stack = h_stack, 
                                 H_stack = H_stack, 
                                 Z_n = Z,
                                 N_order = N_order) for Z in Z_stack]

    H_gn_phi_ns = [get_H_gn_phi_n(Z_n = Z_stack[i],
                                h_stack = h_stack,
                                H_stack = H_stack,
                                D_tilde_stack = D_tilde_stack,
                                U_tilde = U_tilde_stack[i],
                                N_order = N_order) for i in range(n_modes)]
    
    for m in range(n_modes):
        Z_m = Z_stack[m]
        for n in range(m + 1):
            hessian_element = np.sum(H_gn_phi_ns[n] * Z_m)
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
        rate                    = params["rate"]
        threshold               = params["threshold"]
        threshold_damper        = params["threshold_damper"]
    except:
        print("Error: please provide all parameters (gamma, rate, threshold, threshold_damper)")
        

    #initialize variables as necessary
    a_guess = EmptyAberration()
    F_guess = 0
    c_guess = 0

    for i in tqdm(range(1, n_loops + 1)):

        t = time.time()
        f_guess, F_guess = estimate_object(microscope = microscope,
                                           d_stack = d_stack,
                                           a_guess = a_guess,
                                           a_stack = a_stack,
                                           gamma = gamma)
        s = time.time()
        if debug:
            print(f"\t Estimated object in {np.round(s-t, 3)} seconds")

        t = time.time()
        g, g_c = estimate_aberration_gradient(microscope = microscope,
                                              d_stack = d_stack,
                                              a_stack = a_stack,
                                              a_guess = a_guess,
                                              F_guess = F_guess,
                                              modes_corrected = modes_corrected)
        s = time.time()
        if debug:
            print(f"\t Estimated gradient in {np.round(s-t, 3)} seconds")

        t = time.time()
        hessian = get_hessian(microscope = microscope,
                              d_stack = d_stack,
                              a_guess = a_guess,
                              a_stack = a_stack,
                              modes_corrected = modes_corrected,
                              gamma = gamma)
        s = time.time()
        if debug:
            print(f"\t Estimated Hessian in {np.round(s-t, 3)} seconds")
        update = -1 * np.matmul(np.linalg.inv(hessian), g_c) * rate
        #threshold the update
        update = np.clip(update, -threshold, threshold)
        c_guess = c_guess + update
        a_guess = Aberration(modes_corrected, c_guess)

        J = objective(microscope = microscope,
                      d_stack = d_stack, 
                      a_guess = a_guess, 
                      a_stack = a_stack,
                      gamma = gamma)
        if debug:
            print(f"Loop {i}: J = {np.round(J * 1e13, 4)}")
            print(f"Corrercted Zernike Mode: {modes_corrected}")
            print(f"Estimated Zernike Coefficients: {c_guess}")

        c_guess_log = np.zeros((n_loops, len(modes_corrected)))
        F_guess_log = np.zeros((n_loops, 256, 256))
        J_log = np.zeros(n_loops)

        threshold *= threshold_damper

    if log:
        return c_guess, F_guess, c_guess_log, F_guess_log, J_log 
    else:
        return c_guess, F_guess