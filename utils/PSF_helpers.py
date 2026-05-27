from utils.RW_helpers import *
from utils.RW_fast import *
from utils.RW_fast_jax import *
from utils.Zernike_helpers import *
import numpy as np
from scipy.signal import fftconvolve

class Arbitrary_Grid():
    def __init__(self,
                 L_ffp_x: float,
                 L_ffp_y: float,
                 grid_ffp_x: int, 
                 grid_ffp_y: int, 
                 x_offset: float, 
                 y_offset: float,
                 z_level: float):
        self.L_ffp_x = L_ffp_x
        self.L_ffp_y = L_ffp_y
        self.grid_ffp_x = grid_ffp_x
        self.grid_ffp_y = grid_ffp_y
        self.x_offset = x_offset
        self.y_offset = y_offset
        self.z_level = z_level

    def get_xy(self):
        x = np.linspace(-self.L_ffp_x / 2, self.L_ffp_x / 2, self.grid_ffp_x)
        y = np.linspace(-self.L_ffp_y / 2, self.L_ffp_y / 2, self.grid_ffp_y)
        return x,y
    
    def get_grid(self):
        x, y = self.get_xy()
        return np.zeros((len(x), len(y)))

class Square_Grid(Arbitrary_Grid):
    def __init__(self, 
                 L_ffp: float, 
                 grid_ffp: int, 
                 x_offset: float, 
                 y_offset: float,
                 z_level: float):
        super().__init__(L_ffp, L_ffp, grid_ffp, grid_ffp, x_offset, y_offset, z_level)
    

class Centered_Square_Grid(Square_Grid):
    def __init__(self, L_ffp, grid_ffp, z_level):
        super().__init__(L_ffp, grid_ffp, 0.0, 0.0, z_level)

class Image_Mask(Arbitrary_Grid):   
    def __init__(self,
                 grid: Arbitrary_Grid,
                 image_mask: np.array):
        super().__init__(grid.L_ffp_x, 
                         grid.L_ffp_y, 
                         grid.grid_ffp_x, 
                         grid.grid_ffp_y,
                         grid.x_offset, 
                         grid.y_offset, 
                         grid.z_level)
        self.image_mask = image_mask 
        if np.shape(self.image_mask) != np.shape(self.get_grid()):
            raise RuntimeError("Shape of the image mask differs from the L_ffp, grid_ffp parameters.")

class Bead_Image(Image_Mask):
    def __init__(self, 
                 grid: Arbitrary_Grid,
                 xs: list, 
                 ys: list, 
                 bead_sizes: list):
        _, _, image_mask = bead_img(grid, xs, ys, bead_sizes)
        super().__init__(grid, image_mask)

    
    
    
class EmptyAberration(Aberration):
    def __init__(self, 
                 modes: list = [[0,0]]):
        super().__init__(modes, [0] * len(modes))
    
class Microscope():
    def __init__(self, 
                 N_order: int, 
                 lambd: float, 
                 n: float, 
                 num_apt: float, 
                 f: float, 
                 mag: float, 
                 w_0: float,
                 L_bfp: float,
                 grid_bfp: int):
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

        self.cache = 0


    def old_compute_PSF(self, 
                    grid: Arbitrary_Grid, 
                    aberration: Aberration):
        _, _, I = parallel_grid_wrapper(
            L_ffp_x = grid.L_ffp_x,
            L_ffp_y = grid.L_ffp_y,
            grid_ffp_x = grid.grid_ffp_x,
            grid_ffp_y = grid.grid_ffp_y,
            x_offset = grid.x_offset,
            y_offset = grid.y_offset,
            alpha = self.alpha,
            k = self.k,
            f = self.f,
            mag = self.mag, 
            w_0 = self.w_0,
            R_BFP = self.r_bfp, 
            theta_grid_size = self.grid_bfp,
            N_order = self.N_order,
            z = grid.z_level,
            prop_distance = 0 ,
            aberration_kind = "Zernike",
            params = [aberration.modes,
                      aberration.strengths]
        )
        x, y = grid.get_xy()
        return x, y, I
    
    def compute_PSF(self,
                    grid: Arbitrary_Grid, 
                    aberration: Aberration):
        
        _, _, I = rw_fast(
            L_ffp_x = grid.L_ffp_x,
            L_ffp_y = grid.L_ffp_y,
            grid_ffp_x = grid.grid_ffp_x,
            grid_ffp_y = grid.grid_ffp_y,
            x_offset = grid.x_offset,
            y_offset = grid.y_offset,
            alpha = self.alpha,
            k = self.k,
            f = self.f,
            mag = self.mag, 
            w_0 = self.w_0,
            R_BFP = self.r_bfp, 
            grid_bfp = self.grid_bfp,
            N_order = self.N_order,
            z = grid.z_level,
            aberration = aberration
        )
        x, y = grid.get_xy()
        return x, y, I
    
    def compute_PSF_jax(self,
                    grid: Arbitrary_Grid, 
                    aberration: Aberration,
                    force_cache = True):
        
        if self.cache == 0 or force_cache:
            self.cache = make_rw_cache(
            grid.L_ffp_x, grid.L_ffp_y,
            grid.grid_ffp_x, grid.grid_ffp_y,
            grid.x_offset, grid.y_offset,
            self.alpha, self.k, self.r_bfp,
            self.grid_bfp)
        
        
        _, _, I = rw_fast_jax_cached(
            self.cache,
            self.alpha, self.k, self.f, self.mag, self.w_0, self.r_bfp,
            aberration,
            self.N_order,
            grid.z_level
        )
        x, y = grid.get_xy()
        return x, y, I
    
    def compute_image(self, 
                      image: Image_Mask, 
                      aberration: Aberration):
        x, y, PSF = self.compute_PSF(image, aberration)
        return x, y, psf_convolve(image.image_mask, PSF)

def bead_img(grid: Arbitrary_Grid, 
             xs: list, 
             ys: list, 
             bead_sizes: list):
    if not (len(xs) == len(ys) == len(bead_sizes)):
        raise RuntimeError("xs, ys, and bead_sizes must have the same length.")

    length_x = grid.L_ffp_x
    num_pixels_x = grid.grid_ffp_x

    length_y = grid.L_ffp_y
    num_pixels_y = grid.grid_ffp_y

    x, y = grid.get_xy()
    img = np.zeros((num_pixels_x, num_pixels_y))
    center_x = int(num_pixels_x/2)
    center_y = int(num_pixels_y/2)
    for k in range(len(xs)):
        x_center = xs[k]
        y_center = ys[k]
        bead_size = bead_sizes[k]
        for i in range(num_pixels_x):
            for j in range(num_pixels_y):
                x_img = (i - center_x) * (length_x/num_pixels_x)
                y_img = (j - center_y) * (length_y/num_pixels_y)

                distance = np.sqrt((x_center - x_img)**2 + (y_center - y_img)**2)
                if distance <= bead_size:
                    img[i, j] = 1
    return x, y, img

def psf_convolve(image, psf):
    psf_sum = np.sum(psf)
    if psf_sum == 0:
        raise RuntimeError("Cannot normalize PSF because its sum is zero.")
    return fftconvolve(image, psf / psf_sum, mode="same")
