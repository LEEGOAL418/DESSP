#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse


def plot_ga_results(csv_path, output_dir="ga_outputs/figs", plot_convergence=True, plot_distribution=True):
    """
    Plots the convergence curve and offspring energy distribution for Genetic Algorithm results.
    
    Parameters:
    csv_path: Path to the CSV file.
    output_dir: Path to the output directory.
    plot_convergence: Whether to plot the convergence curve.
    plot_distribution: Whether to plot the offspring energy distribution plot.
    
    Returns:
    None
    """
    # Ensure the output directory exists
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # ===== Read CSV, handling cases where 'desc' might contain commas =====
    with open(csv_path, "r") as f:
        header = f.readline().strip().split(",")  # Read the first line as the header
        lines = f.readlines()

    records = []
    for line in lines:
        parts = line.strip().split(",")
        if len(parts) >= 7:
            # The first 6 columns are fixed; everything else is merged into 'desc'
            gen, gaid, raw_E, raw_score, op, parents = parts[:6]
            desc = ",".join(parts[6:])
            records.append([gen, gaid, raw_E, raw_score, op, parents, desc])

    df = pd.DataFrame(records, columns=header)

    # Convert numerical types
    df["gen"] = df["gen"].astype(int)
    df["raw_E(eV)"] = df["raw_E(eV)"].astype(float)
    df["raw_score"] = df["raw_score"].astype(float)

    # Best energy per generation
    best_by_gen = df.groupby("gen")["raw_E(eV)"].min()
    
    # ===== Plotting =====
    if plot_convergence:
        # (a) Best energy per generation
        plt.figure(figsize=(6,4))
        plt.plot(best_by_gen.index, best_by_gen.values, marker="o", label="Best energy per gen")
        plt.xlabel("Generation")
        plt.ylabel("Best Energy (eV)")
        plt.title("GA Convergence Curve")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_path / "convergence_curve.png", dpi=200)
        plt.close()
        print(f" Convergence curve saved to {output_path / 'convergence_curve.png'}")

    if plot_distribution:
        # (b) Scatter plot of all offspring energies + Line plot for best energy per generation
        plt.figure(figsize=(8,6))
        for g, group in df.groupby("gen"):
            plt.scatter([g]*len(group), group["raw_E(eV)"], s=15, alpha=0.6, label=f"Gen {g}" if g<=3 else "")

        # Add red line plot (best per generation)
        plt.plot(best_by_gen.index, best_by_gen.values, color="red", marker="o", linewidth=2, label="Best per gen")

        plt.xlabel("Generation")
        plt.ylabel("Energy (eV)")
        plt.title("Energy distribution of offspring across generations")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_path / "offspring_distribution.png", dpi=200)
        plt.close()
        print(f" Offspring distribution plot saved to {output_path / 'offspring_distribution.png'}")


def main():
    parser = argparse.ArgumentParser(description="Plot GA results from offspring_log.csv")
    parser.add_argument("--csv", "-c", default="ga_outputs/offspring_log.csv", help="Path to the CSV file")
    parser.add_argument("--outdir", "-o", default="ga_outputs/figs", help="Directory to save outputs")
    parser.add_argument("--no-convergence", action="store_true", help="Do not plot convergence curve")
    parser.add_argument("--no-distribution", action="store_true", help="Do not plot offspring distribution")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Error: File not found at {csv_path}")

    # Call the plotting function
    plot_ga_results(
        csv_path=csv_path,
        output_dir=args.outdir,
        plot_convergence=not args.no_convergence,
        plot_distribution=not args.no_distribution
    )


if __name__ == "__main__":
    main()