from utils.RW_helpers import *
from utils.Zernike_helpers import *
import numpy as np
from scipy.signal import fftconvolve

class Grid():
    #L_ffp: FFP size [mm]
    #grid_ffp: number of grid points along each axis
    def __init__(self, L_ffp, grid_ffp, x_offset, y_offset, z_level):
        self.L_ffp = L_ffp 
        self.grid_ffp = grid_ffp
        self.x_offset = x_offset
        self.y_offset = y_offset
        self.z_level = z_level

    def get_xy(self):
        x = np.linspace(-self.L_ffp / 2, self.L_ffp / 2, self.grid_ffp)
        y = np.linspace(-self.L_ffp / 2, self.L_ffp / 2, self.grid_ffp)
        return x,y
    
    def get_grid(self):
        x, y = self.get_xy()
        return np.zeros((len(x), len(y)))


class PSF_Grid(Grid):
    def __init__(self, L_ffp, grid_ffp, z_level):
        super().__init__(L_ffp, grid_ffp, 0.0, 0.0, z_level)

class Aberration():
    def __init__(self, modes, strengths, alpha):
        self.modes = modes 
        self.strengths = strengths
        self.alpha = alpha
    def __str__(self):
        s = ""
        for i, mode in enumerate(self.modes):
            s += f"m={mode[0]}, n={mode[1]}: {np.round(self.strengths[i], 3)}"
        return s
    
    def __add__(self, other):
        return Aberration(self.modes, self.strengths + other.strengths, self.alpha)

    def construct_map(self):
        return create_zernike_function(self.modes, self.strengths, self.alpha)
    
class Microscope():
    def __init__(self, N_order, 
                 lambd, 
                 n, 
                 num_apt, 
                 f, 
                 mag, 
                 w_0,
                 L_bfp,
                 grid_bfp):
        self.N_order = N_order 
        self.lambd =lambd
        self.n = n
        self.num_apt = num_apt 
        self.f = f 
        self.mag = mag
        self.w_0 = w_0
        
        self.L_bfp = L_bfp
        self.grid_bfp = grid_bfp

        self.k = (2*n*np.pi)/lambd 
        r_pupil = f * num_apt
        self.r_bfp = L_bfp / 2

        na_eff = num_apt
        #effective NA can be limited by the size of the back focal plane
        if self.r_bfp < r_pupil:
            na_eff = self.r_bfp / f

        self.alpha = np.arcsin(na_eff / n)


    def compute_PSF(self, psf_grid, aberration):
        _, _, I = parallel_grid_wrapper(
            L_ffp = psf_grid.L_ffp,
            grid_ffp = psf_grid.grid_ffp,
            x_offset = psf_grid.x_offset,
            y_offset = psf_grid.y_offset,
            alpha = self.alpha,
            k = self.k,
            f = self.f,
            mag = self.mag, 
            w_0 = self.w_0,
            R_BFP = self.r_bfp, 
            theta_grid_size = self.grid_bfp,
            N_order = self.N_order,
            z = psf_grid.z_level,
            prop_distance = 0 ,
            aberration_kind = "Zernike",
            params = [aberration.modes,
                      aberration.strengths]
        )
        x, y = psf_grid.get_xy()
        return x, y, I

def bead_img(psf_grid, xs, ys, bead_sizes):
    length = psf_grid.L_ffp
    num_pixels = psf_grid.grid_ffp

    x = np.linspace(-length/2, length/2, num_pixels)
    y = np.linspace(-length/2, length/2, num_pixels)
    img = np.zeros((num_pixels, num_pixels))
    center = int(num_pixels/2)
    for k in range(len(xs)):
        x_center = xs[k]
        y_center = ys[k]
        bead_size = bead_sizes[k]
        for i in range(num_pixels):
            for j in range(num_pixels):
                x_img = (j - center) * (length/num_pixels)
                y_img = (i - center) * (length/num_pixels)

                distance = np.sqrt((x_center - x_img)**2 + (y_center - y_img)**2)
                if distance <= bead_size:
                    img[i, j] = 1
    return x, y, img

def psf_convolve(image, psf):
    return fftconvolve(image, psf/np.sum(psf), mode="same")

def quality(image):
    return np.mean(image**2)
