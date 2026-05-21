from utils.RW_helpers import *
from utils.Zernike_helpers import *
import numpy as np
from scipy.signal import fftconvolve

class Grid():
    def __init__(self, 
                 L_ffp: float, 
                 grid_ffp: int, 
                 x_offset: float, 
                 y_offset: float,
                 z_level: float):
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

class Centered_Grid(Grid):
    def __init__(self, L_ffp, grid_ffp, z_level):
        super().__init__(L_ffp, grid_ffp, 0.0, 0.0, z_level)

class Image_Mask(Grid):   
    def __init__(self,
                 grid: Grid,
                 image_mask: np.array):
        super().__init__(grid.L_ffp, grid.grid_ffp, grid.x_offset, grid.y_offset, grid.z_level)
        self.image_mask = image_mask 
        if np.shape(self.image_mask) != np.shape(self.get_grid()):
            raise RuntimeError("Shape of the image mask differs from the L_ffp, grid_ffp parameters.")

class Bead_Image(Image_Mask):
    def __init__(self, 
                 grid: Grid,
                 xs: list, 
                 ys: list, 
                 bead_sizes: list):
        _, _, image_mask = bead_img(grid, xs, ys, bead_sizes)
        super().__init__(grid, image_mask)

class Aberration:
    def __init__(self, 
                 modes: list, 
                 strengths: list):
        self.modes = modes
        self.strengths = strengths

    def __str__(self):
        s = ""
        for i, mode in enumerate(self.modes):
            s += f"n={mode[0]}, m={mode[1]}: {np.round(self.strengths[i], 3)}\n"
        return s.rstrip()
    
    def __add__(self, other):
        new_modes = np.array(self.modes)
        new_strengths = np.array(self.strengths)

        for i, mode in enumerate(np.array(other.modes)):
            matches = np.all(new_modes == mode, axis=1)

            if np.any(matches):
                new_strengths[matches] += other.strengths[i]
            else:
                new_modes = np.concatenate([new_modes, [mode]], axis=0)
                new_strengths = np.concatenate([new_strengths, [other.strengths[i]]])

        return Aberration(new_modes.tolist(), new_strengths.tolist())
    
    def __sub__(self, other):
        new_modes = np.array(self.modes)
        new_strengths = np.array(self.strengths)

        for i, mode in enumerate(np.array(other.modes)):
            matches = np.all(new_modes == mode, axis=1)

            if np.any(matches):
                new_strengths[matches] -= other.strengths[i]
            else:
                new_modes = np.concatenate([new_modes, [mode]], axis=0)
                new_strengths = np.concatenate([new_strengths, [-other.strengths[i]]])

        return Aberration(new_modes.tolist(), new_strengths.tolist())

    def __len__(self):
        return len(self.modes)

    def construct_map(self, 
                      alpha: float):
        return create_zernike_function(self.modes, self.strengths, alpha)
    
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


    def compute_PSF(self, 
                    grid: Grid, 
                    aberration: Aberration):
        _, _, I = parallel_grid_wrapper(
            L_ffp = grid.L_ffp,
            grid_ffp = grid.grid_ffp,
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
    
    def compute_image(self, 
                      image: Image_Mask, 
                      aberration: Aberration):
        x, y, PSF = self.compute_PSF(image, aberration)
        return x, y, psf_convolve(image.image_mask, PSF)

def bead_img(grid: Grid, 
             xs: list, 
             ys: list, 
             bead_sizes: list):
    if not (len(xs) == len(ys) == len(bead_sizes)):
        raise RuntimeError("xs, ys, and bead_sizes must have the same length.")

    length = grid.L_ffp
    num_pixels = grid.grid_ffp

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
    psf_sum = np.sum(psf)
    if psf_sum == 0:
        raise RuntimeError("Cannot normalize PSF because its sum is zero.")
    return fftconvolve(image, psf / psf_sum, mode="same")
