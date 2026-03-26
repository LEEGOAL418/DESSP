#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
initialize_population_soap.py (Refined with Kernel-SOAP)

Purpose:
  Generate initial GA populations for the LiₓC₁₂ system.
  [Upgrade Point] Introduced Kernel-SOAP descriptors for rigorous structural de-duplication, 
  replacing fragile coordinate hashing.
  Ensures that the initial generated structures are not only different in coordinates 
  but also significantly different in their geometric fingerprints.

Main Features:
  KernelSOAPComparator for de-duplication (threshold 0.995/0.9999)
  Automatic completion (generates sufficient structures even with few HT sites)
  Never-stuck logic (protected by maximum attempt count)
  Supports random seeds (--seed) for reproducibility

Usage Example:
  python initialize_population_soap.py \
      --structure C36.vasp \
      --db ga_test.db \
      --n_li 4 \
      --pop_size 20 \
      --seed 42
"""

import os
import json
import numpy as np
import random
from random import sample, uniform
from pathlib import Path
from ase import Atoms
from ase.io import read, write
from ase.ga.data import PrepareDB
from ase.geometry import wrap_positions
import math

# --- Import DScribe ---
try:
    from dscribe.descriptors import SOAP
except ImportError:
    raise ImportError("dscribe library is required: pip install dscribe")

# --- Import HT site finding tool ---
try:
    from HT_sites import find_ht_sites_ab
except ImportError:
    print("️ Warning: HT_sites.py not found. Ensure it is in the same directory.")
    # Placeholder to prevent IDE errors
    find_ht_sites_ab = None

# ==============================================================
# ================ 1. Kernel SOAP Comparator ==================
# ==============================================================

class KernelSOAPComparator:
    """
    Used for structural de-duplication during the initialization phase.
    """
    def __init__(self, species=["C", "Li"], r_cut=6.0, n_max=8, l_max=6, sigma=0.2, similarity_threshold=0.9999):
        # The threshold can be slightly relaxed (e.g., 0.995) during initialization to ensure generation efficiency
        self.similarity_threshold = similarity_threshold
        self.soap = SOAP(
            species=species,
            periodic=True,
            r_cut=r_cut,
            n_max=n_max,
            l_max=l_max,
            sigma=sigma,
            average="off", 
            sparse=False
        )
    
    def get_fingerprint(self, atoms):
        # Cache the fingerprint to speed up multiple comparisons
        if 'soap_local_fp' in atoms.info:
            return atoms.info['soap_local_fp']
        
        if not np.any(atoms.get_pbc()):
            atoms.set_pbc(True)
        
        try:
            fp_matrix = self.soap.create(atoms, n_jobs=1)
            norms = np.linalg.norm(fp_matrix, axis=1, keepdims=True)
            norms[norms < 1e-8] = 1.0 
            fp_matrix = fp_matrix / norms
        except Exception:
            return None
            
        atoms.info['soap_local_fp'] = fp_matrix
        return fp_matrix

    def looks_like(self, a1, a2):
        fp1 = self.get_fingerprint(a1)
        fp2 = self.get_fingerprint(a2)
        
        if fp1 is None or fp2 is None:
            return False, 0.0
            
        similarity_matrix = np.dot(fp1, fp2.T)
        best_match_a = np.max(similarity_matrix, axis=1) 
        best_match_b = np.max(similarity_matrix, axis=0)
        final_score = (np.mean(best_match_a) + np.mean(best_match_b)) / 2.0
        
        return final_score > self.similarity_threshold, final_score

# ==============================================================
# ================ 2. Perturbation & Generation Logic ==========
# ==============================================================

def apply_layer_perturbation(atoms, max_shift_xy=0.5, max_dz=0.3, max_slide=1.2):
    """Apply intra-layer/inter-layer perturbations"""
    atoms = atoms.copy()
    pos = atoms.get_positions()

    z_mean = np.mean(pos[:, 2])
    top_mask = pos[:, 2] > z_mean
    bottom_mask = pos[:, 2] <= z_mean

    # 1) Intra-layer translation (random jitter)
    shift_xy = np.array([
        uniform(-max_shift_xy, max_shift_xy),
        uniform(-max_shift_xy, max_shift_xy),
        0.0
    ])
    pos[top_mask] += shift_xy

    # 2) Relative inter-layer sliding (simulate stacking variations)
    slide_xy = np.array([
        uniform(-max_slide, max_slide),
        uniform(-max_slide, max_slide),
        0.0
    ])
    pos[top_mask] += slide_xy / 2
    pos[bottom_mask] -= slide_xy / 2

    # 3) Inter-layer spacing perturbation
    dz = uniform(-max_dz, max_dz)
    pos[top_mask, 2] += dz
    pos[bottom_mask, 2] -= dz / 2

    atoms.set_positions(wrap_positions(pos, atoms.get_cell(), pbc=True))
    return atoms

# ==============================================================
# ================ 3. Main Function ============================
# ==============================================================

def main(structure_file="C36.vasp", db_file="ga_test.db",
         n_li=4, pop_size=None, outdir="init_outputs",
         tol=1e-3, max_shift_xy=0.5, max_dz=0.3, max_slide=1.2, seed=None):

    # ===== Set random seed =====
    if seed is None:
        seed = np.random.randint(0, 10**6)
        print(f" Seed not specified, automatically generated seed: {seed}")
    else:
        print(f" Using specified seed: {seed}")
    np.random.seed(seed)
    random.seed(seed)

    structure_path = Path(structure_file)
    if not structure_path.exists():
        raise FileNotFoundError(f"Structure file not found: {structure_path}")
    base_structure = read(structure_path)

    # === Output directories ===
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    extxyz_dir = outdir / f"extxyz_Li{n_li}"
    poscar_dir = outdir / f"POSCAR_Li{n_li}"
    index_dir = outdir / f"indices_Li{n_li}"
    for d in [extxyz_dir, poscar_dir, index_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # === HT Sites ===
    if find_ht_sites_ab is None:
         raise ImportError("Missing HT_sites module")
         
    li_sites = find_ht_sites_ab(base_structure, tol=tol, plot=False)
    if li_sites is None or len(li_sites) == 0:
        raise RuntimeError("No HT sites found. Check if the input structure is AB-stacked BLG.")
    np.savetxt(outdir / "Li-sites.txt", li_sites, fmt="%.6f")
    
    n_sites = len(li_sites)
    print(f" Number of HT sites = {n_sites}")

    # === Estimate reasonable pop_size ===
    n_combo = int(math.comb(n_sites, min(n_li, n_sites)))
    n_perturb = 12
    total_space = n_combo * n_perturb

    if pop_size is None or pop_size <= 0:
        if total_space < 50:
            pop_size = min(total_space, 5)
        elif total_space < 500:
            pop_size = max(5, int(total_space * 0.1))
        else:
            pop_size = 10
        print(f" Automatic estimation: ~{total_space} candidates in config space → recommended pop_size = {pop_size}")
    else:
        print(f" Using specified pop_size = {pop_size}")

    # === Initialize Database ===
    if Path(db_file).exists():
        print(f"️ Old database {db_file} detected, deleting to regenerate...")
        os.remove(db_file)
        
    db = PrepareDB(
        db_file_name=db_file,
        population_size=pop_size,
        slab=base_structure,
        stoichiometry=["Li"] * n_li,
        extra_keys=["li_indices", "perturb_type"]
    )

    # === Initialize SOAP Comparator ===
    print(" Initializing Kernel-SOAP Comparator (Threshold=0.9999)...")
    soap_comp = KernelSOAPComparator(similarity_threshold=0.9999)
    
    # Store generated structure objects in memory for comparison
    accepted_structures = [] 

    all_indices = list(range(len(li_sites)))
    accepted = 0
    attempts = 0
    max_attempts = pop_size * 100 # Allow sufficient attempts

    # === Generation Loop ===
    print(f"\n Generating {pop_size} initial structures...")
    
    while accepted < pop_size and attempts < max_attempts:
        attempts += 1
        
        # 1. Randomly select sites
        li_idx_list = sorted(sample(all_indices, min(n_li, len(all_indices))))
        li_positions = li_sites[li_idx_list]
        li_atoms = Atoms("Li" * n_li, positions=li_positions)
        
        candidate = base_structure.copy()
        candidate.extend(li_atoms)

        # 2. Apply perturbation
        perturbed = apply_layer_perturbation(
            candidate,
            max_shift_xy=max_shift_xy,
            max_dz=max_dz,
            max_slide=max_slide
        )

        # 3. SOAP De-duplication
        is_duplicate = False
        # Compare against all accepted structures
        for existing in accepted_structures:
            # looks_like automatically calculates and caches fingerprints
            is_sim, score = soap_comp.looks_like(perturbed, existing)
            if is_sim:
                is_duplicate = True
                # Optional: print duplication info for debugging
                # print(f"  [Skip] Similar to existing structure #{accepted_structures.index(existing)+1} (Sim={score:.4f})")
                break
        
        if is_duplicate:
            continue

        # 4. Accept Structure
        accepted_structures.append(perturbed)
        
        # Add to database
        data = {"li_indices": li_idx_list, "perturb_type": "layer+shift+slide"}
        kvp = {"li_indices": json.dumps(li_idx_list), "perturb_type": "layer+shift+slide"}
        db.add_unrelaxed_candidate(perturbed, data=data, key_value_pairs=kvp)

        # Export files
        write(extxyz_dir / f"candidate_{accepted}.extxyz", perturbed)
        write(poscar_dir / f"POSCAR_{accepted}", perturbed, format="vasp")
        np.savetxt(index_dir / f"indices_{accepted}.txt", np.array(li_idx_list), fmt="%d")

        accepted += 1
        print(f"[{accepted}/{pop_size}]  New structure generated (Att {attempts}) | Li combination {li_idx_list}")

    # === Final Report ===
    print(f"\n Generation finished.")
    if accepted < pop_size:
        print(f" Warning: Population not filled (Target {pop_size}, Actual {accepted}). Config space might be exhausted or threshold is too high.")
    else:
        print(f" Successfully generated {pop_size} unique initial structures!")
    
    print(f" Database: {db_file}")
    print(f" Output directory: {outdir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate GA initial population with SOAP de-duplication")
    parser.add_argument("--structure", "-s", default="C36.vasp", help="Base structure file")
    parser.add_argument("--db", "-d", default="ga_test.db", help="Database file name")
    parser.add_argument("--n_li", "-n", type=int, default=4, help="Number of Li atoms")
    parser.add_argument("--pop_size", "-p", type=int, default=None, help="Population size")
    parser.add_argument("--outdir", "-o", default="init_outputs", help="Output directory")
    parser.add_argument("--tol", type=float, default=0.05, help="HT site tolerance")
    parser.add_argument("--max_shift_xy", type=float, default=0.5, help="Intra-layer translation")
    parser.add_argument("--max_dz", type=float, default=0.3, help="Inter-layer distance perturbation")
    parser.add_argument("--max_slide", type=float, default=1.2, help="Inter-layer slide")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")

    args = parser.parse_args()

    main(
        structure_file=args.structure,
        db_file=args.db,
        n_li=args.n_li,
        pop_size=args.pop_size,
        outdir=args.outdir,
        tol=args.tol,
        max_shift_xy=args.max_shift_xy,
        max_dz=args.max_dz,
        max_slide=args.max_slide,
        seed=args.seed
    )