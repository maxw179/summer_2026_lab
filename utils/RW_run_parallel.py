#!/usr/bin/env python
import argparse
import numpy as np
import ast
import RW_helpers


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute PSF intensity grid using multiprocessing."
    )

    #basic grid / region params
    parser.add_argument(
        "--L-ffp",
        type=float,
        required=True,
        help="Field of view in the Fourier/focal plane (same units as x,y).",
    )
    parser.add_argument(
        "--grid-ffp",
        type=int,
        required=True,
        help="Number of points along each axis in the x-y grid.",
    )

    #optical parameters
    parser.add_argument(
        "--alpha",
        type=float,
        required=True,
        help="Half-angle of the objective (in radians).",
    )
    parser.add_argument(
        "--k",
        type=float,
        required=True,
        help="Wavenumber k = 2πn/λ (in 1/length).",
    )
    parser.add_argument(
        "--f",
        type=float,
        required=True,
        help="Focal length of the objective (same length units as x,y).",
    )
    parser.add_argument(
        "--mag",
        type=float,
        required=True,
        help="magnification of objective.",
    )
    parser.add_argument(
        "--w_0",
        type=float,
        required=True,
        help="beam waist.",
    )
    parser.add_argument(
        "--R_BFP",
        type=float,
        required=True,
        help="radius of back focal plane [mm].",
    )

    # Integration grid + nonlinearity order
    parser.add_argument(
        "--theta-grid-size",
        type=int,
        required=True,
        help="Number of theta (and phi) samples for the angular integration.",
    )
    parser.add_argument(
        "--N-order",
        type=int,
        required=True,
        help="Order of nonlinearity (2 for 2P, 3 for 3P, etc.).",
    )

    # Optional aberration map
    parser.add_argument(
        "--aberration-kind",
        type=str,
        default=None,
        help="Type of aberration map to use",
    )
    parser.add_argument(
        "--z",
        type=float,
        default=None,
        help="Z level to integrate at",
    )
    parser.add_argument(
        "--prop_distance",
        type=float,
        default=None,
        help="distance to propagate",
    )

    # Misc
    parser.add_argument(
        "--n-procs",
        type=int,
        default=None,
        help="Number of processes to use (default: cpu_count).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="psf_result.npz",
        help="Output .npz file to write x, y, I to (default: psf_result.npz).",
    )

    parser.add_argument(
        "--params",
        type=ast.literal_eval,
        default = [],
        help = "Extra parameters to the function (including parameters for the aberration map)"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    #call the parallel PSF computation
    x, y, I = RW_helpers.intensity_grid_parallel(
        L_ffp=args.L_ffp,
        grid_ffp=args.grid_ffp,
        alpha=args.alpha,
        k=args.k,
        f=args.f,
        mag=args.mag,
        R_BFP=args.R_BFP,
        w_0 = args.w_0,
        z = args.z,
        theta_grid_size=args.theta_grid_size,
        N_order=args.N_order,
        aberration_kind=args.aberration_kind,
        prop_distance= args.prop_distance,
        n_procs=args.n_procs,
        params=args.params
    )

    #save to disk for the notebook (or anything) to load later
    np.savez(args.output, x=x, y=y, I=I)

if __name__ == "__main__":
    main()
