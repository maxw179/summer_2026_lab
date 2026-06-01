import numpy as np
import math
from PIL import Image

def radial_zernike(m, n, rho):
    #0 for odd n-m
    if (n - m) % 2 == 1:
        return 0
    #normal case with some radial dependence
    total = 0
    for k in range(0, int((n-m)/2 + 1)):
        numerator = (-1)**k * math.factorial(n-k)
        denominator = math.factorial(k) * math.factorial(int((n+m)/2 - k)) * math.factorial(int((n-m)/2 - k))

        total += (numerator/denominator) * rho**(n - 2 * k)
    return total

def zernike_mode(m, n, rho, phi):
    radial_component = radial_zernike(np.abs(m), n, rho)
    #no angular dependence
    if m == 0:
        return radial_component
    #odd zernikes
    elif m < 0:
        return radial_component * np.sin(np.abs(m) * phi)
    #even zernikes
    else:
        return radial_component * np.cos(np.abs(m) * phi)

#convert from the pupil coordinates into Zernike land
def pupil_polar_coords(theta_grid, phi_grid, alpha):
    rho = np.sin(theta_grid) / np.sin(alpha)
    psi = phi_grid
    return rho, psi

#modes is a list of lists [[m_1, n_1], [m_2, n_2], ...]
#strengths is a list [a_1, a_2,...]
def create_zernike_function(modes, strengths, alpha):
    #make completely immutable after creation
    modes = tuple(tuple(m) for m in modes)
    strengths = tuple(strengths)
    alpha = float(alpha)
    def total_zernike_map(theta_grid, phi_grid):
        rho, psi = pupil_polar_coords(theta_grid, phi_grid, alpha)
        #get ready to store the phases
        phase = np.zeros_like(theta_grid)
        #only add zernikes within the pupil
        mask = rho <= 1.0
        rho_inside = rho[mask]
        psi_inside = psi[mask]

        if rho_inside.size == 0:
            #return no phase
            return phase
    
        slm_waves = np.zeros_like(rho_inside)
        #iterate over each zernike mode and add to the SLM
        for i in range(len(modes)):
            m = modes[i][0] 
            n = modes[i][1]
            strength = strengths[i]

            zernike = zernike_mode(m, n, rho_inside, psi_inside)
            slm_waves += strength * zernike 

        phase[mask] = 2 * np.pi * slm_waves
        return phase  
    
    return total_zernike_map

def get_allowed_modes(min_order, max_order):
    modes = []
    for n in range(min_order, max_order):
        if n == 0:
            modes.append([0,0])
        else:
            for m in range(-n, n + 1, 2):
                modes.append([m,n ])
    return modes

def zernike_RMS(a_1, a_2, alpha, n=100):
        z_1 = a_1.construct_map(alpha)
        z_2 = a_2.construct_map(alpha)
        x = np.linspace(-1, 1, n)
        y = np.linspace(-1, 1, n)

        x_grid, y_grid = np.meshgrid(x, y, indexing="xy")
        rho_grid = np.sqrt(x_grid**2 + y_grid**2)
        phi_grid = np.arctan2(y_grid, x_grid)

        mask = rho_grid <= 1.0
        theta_grid = np.arcsin(rho_grid * np.sin(alpha))

        z_1_out = z_1(theta_grid[mask], phi_grid[mask])
        z_2_out = z_2(theta_grid[mask], phi_grid[mask])

        # difference in "waves"
        dz = (z_1_out - z_2_out)

        ## wrap to [-0.5, 0.5) waves (shortest phase difference)
        #dz_wrapped = ((dz + 0.5) % 1.0) - 0.5

        return np.sqrt(np.mean(dz**2))

def generate_random_aberration(seed, num_orders, scaling):
    rng = np.random.default_rng(seed)
    #don't do tilt or piston, set min_mode = 2
    min_mode = 2
    modes = get_allowed_modes(min_mode, num_orders + min_mode)
    strengths = []
    for mode in modes:
        n = mode[1]
        #prevent a division by 0 error
        if n ==0: n = 1
        strengths.append(rng.uniform(-scaling/np.sqrt(n), scaling/np.sqrt(n)))

    return np.array(modes), np.array(strengths)

#z_map takes in theta, phi
def decompose_wavefront(z_map, alpha, decomp_order, n=100):
    x = np.linspace(-1, 1, n)
    y = np.linspace(-1, 1, n)

    x_grid, y_grid = np.meshgrid(x, y, indexing="xy")
    rho_grid = np.sqrt(x_grid**2 + y_grid**2)
    phi_grid = np.arctan2(y_grid, x_grid)

    mask = rho_grid <= 1.0
    theta_grid = np.arcsin(rho_grid * np.sin(alpha))
    wavefront = np.zeros_like(theta_grid)
    wavefront[mask] = z_map(theta_grid[mask], phi_grid[mask])

    modes = get_allowed_modes(0, decomp_order)
    strengths = []

    for mode in modes:
        Z = np.zeros_like(rho_grid)
        Z[mask] = zernike_mode(*mode, rho_grid[mask], phi_grid[mask])

        coeff = np.sum(wavefront[mask] * Z[mask]) / np.sum(Z[mask]**2)
        #return output in waves
        strengths.append(coeff/(2*np.pi))

    return modes, strengths

def img_to_array(img_path, alpha):
    img = Image.open(img_path)
    img_array = np.array(img)
    gray = img_array[:,:,0]
    gray[gray >0] = 1
    H, W = gray.shape

    # Return a closure z_map(theta, phi)
    def z_map(theta, phi):
        # theta->rho mapping for your pupil
        rho = np.sin(theta) / np.sin(alpha)

        # pupil Cartesian coords in [-1,1]
        x = rho * np.cos(phi)
        y = rho * np.sin(phi)

        # map (x,y) -> image coords (u,v) in [0, W-1]x[0, H-1]
        # (x,y)=(-1,-1) -> bottom-left, (1,1) -> top-right
        u = (x + 1) * 0.5 * (W - 1)
        v = (1 - (y + 1) * 0.5) * (H - 1)  # flip y for image coordinates

        # bilinear sampling
        u0 = np.floor(u).astype(int)
        v0 = np.floor(v).astype(int)
        u1 = np.clip(u0 + 1, 0, W - 1)
        v1 = np.clip(v0 + 1, 0, H - 1)
        u0 = np.clip(u0, 0, W - 1)
        v0 = np.clip(v0, 0, H - 1)

        du = u - u0
        dv = v - v0

        I00 = gray[v0, u0]
        I10 = gray[v0, u1]
        I01 = gray[v1, u0]
        I11 = gray[v1, u1]

        I = (1-du)*(1-dv)*I00 + du*(1-dv)*I10 + (1-du)*dv*I01 + du*dv*I11

        # Outside pupil disk, set to 0
        I = np.where(rho <= 1.0, I, 0.0)

        #cconvert intensity -> phase map (centered)
        I = (I/np.max(I)) * 2*np.pi
        return I

    return z_map
    
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