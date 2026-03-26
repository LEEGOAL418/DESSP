#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from ase.io import read
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import argparse
from pathlib import Path


def find_ht_sites_ab(structure, tol=1e-2, plot=False, outdir=None, save_sites=True):
    """
    Find Hollow-Top (HT) sites in AB-stacked bilayer graphene.
    
    Parameters:
    structure: ASE Atoms object
    tol: Tolerance used to determine if a site is HT
    plot: Whether to generate a distribution plot of the sites
    outdir: Path to the output directory
    save_sites: Whether to save the site coordinates to a file
    
    Returns:
    ht_sites: Array of HT site coordinates
    """
    c_positions = np.array([a.position for a in structure if a.symbol == "C"])

    # Dynamically calculate z_middle so it lies exactly between the two layers
    z_mean = np.mean(c_positions[:, 2])

    top_layer = c_positions[c_positions[:, 2] > z_mean]
    bottom_layer = c_positions[c_positions[:, 2] < z_mean]

    tree_bottom = cKDTree(bottom_layer[:, :2])
    ht_sites = []
    for c_top in top_layer:
        # Query the nearest neighbor in the bottom layer using XY coordinates
        dist, _ = tree_bottom.query(c_top[:2], k=1)
        # In AB stacking, if a top atom has no bottom atom directly beneath it, it's an HT site
        if dist > tol:
            ht_sites.append([c_top[0], c_top[1], z_mean])

    ht_sites = np.array(ht_sites)
    
    # Set up output directory if saving sites or plotting
    if plot or save_sites:
        if outdir is None:
            outdir = Path.cwd()
        else:
            outdir = Path(outdir)
            outdir.mkdir(parents=True, exist_ok=True)
    
    # Save site coordinates
    if save_sites and len(ht_sites) > 0:
        formula = structure.get_chemical_formula(mode="reduce")
        sites_file = outdir / "Li-sites.txt"
        np.savetxt(sites_file, ht_sites, fmt="%.6f")
        print(f"HT sites saved to {sites_file}")
    
    # Generate visualization plot
    if plot and len(ht_sites) > 0:
        formula = structure.get_chemical_formula(mode="reduce")
        cell = structure.get_cell()
        x_len = np.linalg.norm(cell[0])
        y_len = np.linalg.norm(cell[1])
        aspect_ratio = y_len / x_len if x_len > 0 else 1.
        base_width = 6
        fig_height = min(base_width * aspect_ratio, base_width * 2)

        plt.figure(figsize=(base_width, fig_height))
        plt.scatter(c_positions[:, 0], c_positions[:, 1], c="black", s=15, label="C atoms")
        plt.scatter(ht_sites[:, 0], ht_sites[:, 1], c="red", s=35, marker="x", label="HT sites")
        plt.xlabel("x (Å)")
        plt.ylabel("y (Å)")
        plt.legend()
        plt.gca().set_aspect("equal", adjustable="box")
        plt.title(f"AB-stacked BLG ({formula}): C atoms & HT sites")

        image_file = outdir / f"HT_sites_{formula}.png"
        plt.tight_layout()
        plt.savefig(image_file, dpi=300)
        plt.close()
        print(f"HT site distribution plot saved to {image_file}")
    
    return ht_sites


def main():
    parser = argparse.ArgumentParser(description="Find and visualize HT sites in bilayer graphene.")
    parser.add_argument("--structure", "-s", required=True, help="Path to the structure file (e.g., C192.vasp).")
    parser.add_argument("--tol", type=float, default=1e-3, help="Tolerance for HT site detection.")
    parser.add_argument("--outdir", type=str, default="outputs_ht", help="Directory to save outputs.")
    parser.add_argument("--no-plot", action="store_true", help="Do not generate plot.")
    parser.add_argument("--no-save", action="store_true", help="Do not save site coordinates.")
    args = parser.parse_args()

    structure_path = Path(args.structure)
    if not structure_path.is_file():
        raise FileNotFoundError(f"Error: File not found at {structure_path}")

    # Read structure
    structure = read(structure_path, index=0)
    
    # Calculate HT sites
    ht_sites = find_ht_sites_ab(
        structure, 
        tol=args.tol, 
        plot=not args.no_plot, 
        outdir=args.outdir,
        save_sites=not args.no_save
    )
    
    # Output summary information
    formula = structure.get_chemical_formula(mode="reduce")
    print(f"Structure: {formula}")
    print(f"Total HT intercalation sites found: {len(ht_sites)}")


if __name__ == "__main__":
    main()