from utils.PSF_helpers import *
from utils.Zernike_helpers import *
from utils.Plot_helpers import *
from scipy.optimize import least_squares
import numpy as np
import matplotlib.pyplot as plt

class SLM():
    def __init__(self,
                  microscope: Microscope,
                  image_mask: Image_Mask,
                  innate_aberration: Aberration):
        self.microscope = microscope
        self.image_mask = image_mask
        self.x, self.y = self.image_mask.get_xy()
        self.grid = self.image_mask.get_grid()
        self.innate_aberration = innate_aberration 

        print("Getting unaberrated quality...")
        self.quality_0 = self.get_unaberrated_quality()

    def run_booth(self, 
                  num_iterations: int = 10, 
                  corrected_modes: list = None, 
                  init_bias: float = 0.2,
                  min_bias: float = 0.001,
                  bias_decay: float = 4/5):
        
        print("===== Running Booth 2002 Modal AO Algorithm =====")
        if corrected_modes is None:
            print("No specific modes inputted. Assuming all modes should be corrected.")
            corrected_modes = self.innate_aberration.modes
            
        else:
            print("Noted that only some modes should be corrected.")

        #prepare the corrected aberration data structure
        current_correction = Aberration(corrected_modes, np.zeros(len(corrected_modes)))

        print("Original Aberration:")
        print(self.innate_aberration)
        #each subarray is one iteration worth of adjustments to be made
        correction_log = np.zeros((num_iterations+ 1, len(current_correction)))
        quality_log = np.zeros(num_iterations + 1)
        bias_log = np.zeros(num_iterations)

        bias = init_bias
        for iteration in range(num_iterations):
            print(f"===== Iteration {iteration + 1}=====")
            #get unbiased image
            print("Getting raw image quality..")
            M_0 = self.get_corrected_quality(current_correction)
            correction_log[iteration] = current_correction.strengths
            quality_log[iteration] = M_0
            bias_log[iteration] = bias

            #print the image quality for the user
            print("Image Quality:", np.round(M_0/self.quality_0, 3))
            #initialize the steps to be taken in the next round
            steps = np.zeros(len(current_correction))
            #optimize each mode separately
            print(f"Applying biases: {np.round(bias, 3)} waves")
            for mode_index, mode in enumerate(corrected_modes):
                print(f"\t===Optimizing m={mode[0]}, n={mode[1]}===")
                #initialize the arrays with the biases
                plus_bias = np.copy(current_correction.strengths)
                minus_bias = np.copy(current_correction.strengths)
                plus_bias[mode_index] += bias 
                minus_bias[mode_index] -= bias

                plus_aberration = Aberration(corrected_modes, plus_bias)
                minus_aberration = Aberration(corrected_modes, minus_bias)

                print("\tExploring plus bias...")
                M_plus =  self.get_corrected_quality(plus_aberration)
                print("\tImage Quality:", np.round(M_plus/self.quality_0, 3))
                print("\tExploring minus bias...")
                M_minus =  self.get_corrected_quality(minus_aberration)
                print("\tImage Quality:", np.round(M_minus/self.quality_0, 3))

                print("\tEvaluating next step...")
                xfit = [-bias, 0, bias]
                yfit = [M_minus, M_0, M_plus]
                a, b, c = downwards_parabola(xfit, yfit)

                step = 0
                if abs(a) > 0:
                    step = -b/(2*a)
                
                print(f"\tComputed step to be: {np.round(step, 3)}")
                if step < -bias:
                    print(f"\t\tRounding up to: {np.round(-bias, 3)} ")
                    step = -bias
                elif step > bias:
                    print(f"\t\tRounding down to: {np.round(bias, 3)} ")
                    step = bias
                
                steps[mode_index] = step 
            #add the steps to the correction
            current_correction.strengths += steps
            #recompute the bias
            bias = np.max([min_bias, bias * bias_decay])
        final_correction = current_correction
        print("Getting final image at: " + str(final_correction))
        M_f = self.get_corrected_quality(final_correction)
        quality_log[-1] = M_f
        correction_log[-1] = final_correction.strengths
        print("Final image quality:", np.round(M_f/self.quality_0, 3))
        return correction_log, quality_log, bias_log, final_correction

    def get_unaberrated_img(self):
        _, _, img = self.microscope.compute_image(self.image_mask, EmptyAberration())
        return img
    
    def get_unaberrated_quality(self):
        img = self.get_unaberrated_img()
        return quality(img)
    
    def get_aberrated_img(self, mode = "vector"):
        _, _, img = self.microscope.compute_image(self.image_mask, self.innate_aberration, mode)
        return img

    def get_aberrated_quality(self):
        img = self.get_aberrated_img()
        return quality(img)
    
    def get_corrected_img(self, 
                          correction: Aberration,
                          mode = "vector"):
        _, _, img = self.microscope.compute_image(self.image_mask, self.innate_aberration + correction, mode)
        return img

    def get_corrected_quality(self,
                              correction: Aberration):
        img = self.get_corrected_img(correction)
        return quality(img)
    
    #takes in the input from the booth correction algorithm
    def produce_correction_graphic(self, quality_log, correction_log, final_correction):
        alpha = self.microscope.alpha
        fig, axs = plt.subplots(2, 2, sharex = "col")

        imgs = [self.get_aberrated_img(), 
                self.get_corrected_img(final_correction)]
        zmaps = [(self.innate_aberration).construct_map(alpha), 
                 (self.innate_aberration + final_correction).construct_map(alpha)]
        
        many_composite(fig, axs[:, 0], self.x, self.y,
                       imgs,
                       zmaps, 
                       alpha,
                       is_edge = True, is_horizontal=False)

        #iterations
        iteration_no = np.arange(0, len(quality_log))
        #get quality data for the first image taken in each iteration
        #get z_map for the correction in each iteration
        z_maps = [(self.innate_aberration + Aberration(final_correction.modes, correction_strengths)).construct_map(alpha)
                for correction_strengths in correction_log]
        errors = [zernike_RMS(z_map, EmptyAberration().construct_map(alpha), alpha) for z_map in z_maps]
        
        axs[0][1].plot(iteration_no, quality_log/self.quality_0, marker = "o")
        axs[0][1].set_ylabel("Image Quality")

        axs[1][1].plot(iteration_no, errors, marker = "o")
        axs[1][1].set_xlabel("Iteration No.")
        axs[1][1].set_ylabel("Residual Aberration RMS")
        axs[1][1].set_xticks(iteration_no)

def quality(img: np.array):
    p = np.percentile(img, 99) 
    return np.mean(img[img > p])

def parabola(x, a, b, c):
    return a*x*x + b*x + c

def downwards_parabola(xfit, yfit):
    xfit = np.asarray(xfit, dtype=float)
    yfit = np.asarray(yfit, dtype=float)

    def residual(mu):
        a, b, c = mu
        return parabola(xfit, a, b, c) - yfit

    # ordinary quadratic fit as initial guess
    a0, b0, c0 = np.polyfit(xfit, yfit, 2)

    # if initial guess opens upward, start with a small downward curvature
    if a0 > 0:
        a0 = -1e-12

    result = least_squares(
        residual,
        x0=[a0, b0, c0],
        bounds=([-np.inf, -np.inf, -np.inf],
                [0,       np.inf,  np.inf])
    )

    return result.x

