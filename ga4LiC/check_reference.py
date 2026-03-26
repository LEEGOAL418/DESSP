#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import numpy as np
from ase.io import read

# ---------- Geometric Helpers ----------
def layer_distance(atoms):
    """Calculates the average z-coordinate difference between the top and bottom C layers."""
    C_pos = atoms[[i for i, s in enumerate(atoms.get_chemical_symbols()) if s == "C"]].positions
    if len(C_pos) == 0:
        return np.nan
    z_vals = C_pos[:, 2]
    z_sorted = np.sort(z_vals)
    half = len(z_sorted) // 2
    return z_sorted[half:].mean() - z_sorted[:half].mean()
    
def stacking_shift(atoms, percentile=80):
    """
    Calculates the interlayer slip scalar (Å) for a single configuration, 
    used for AA/AB/SP stacking discrimination.
    
    Method: For each top-layer C atom, calculate the 2D minimum image distance 
    to the "nearest bottom-layer C atom"; then take a given percentile (default p80) 
    as the slip amount.

    Returns:
        float: Slip amount (Å). AA ≈ 0, AB ≈ 1.42, SP is between the two.
    """
    # 1) Get C atom fractional coordinates and split layers by fz
    symbols = atoms.get_chemical_symbols()
    C_idx = [i for i, s in enumerate(symbols) if s == "C"]
    if not C_idx:
        return np.nan

    fcoords = atoms.get_scaled_positions(wrap=True)[C_idx]  # [0,1)×3
    order = np.argsort(fcoords[:, 2])
    fcoords = fcoords[order]
    half = len(fcoords) // 2
    bot_uv = fcoords[:half, :2]   # Bottom layer (u,v)
    top_uv = fcoords[half:, :2]   # Top layer (u,v)

    # 2) Lattice vectors and surface normal
    a_vec, b_vec, c_vec = atoms.cell.array
    n = c_vec / np.linalg.norm(c_vec)

    # 3) For each top atom, calculate the 2D minimum image distance to the nearest bottom atom
    dists = []
    for uv in top_uv:
        duv = uv[None, :] - bot_uv          # (Nb,2)
        duv -= np.round(duv)                # wrap to [-0.5, 0.5)
        vecs = duv[:, 0][True, None] * a_vec + duv[:, 1][True, None] * b_vec  # -> Cartesian
        # Strictly project into the plane
        vecs_inplane = vecs - (vecs @ n)[:, None] * n[None, :]
        dists.append(np.linalg.norm(vecs_inplane, axis=1).min())

    dists = np.asarray(dists)
    # 4) Use a high percentile to represent the "slip amount"
    return float(np.percentile(dists, percentile))

def rmsd_with_li_alignment(ref_atoms, cand_atoms, include_li=False):
    """
    Calculates RMSD using pymatgen's StructureMatcher.
    Suitable for fixed-lattice systems (e.g., Li-intercalated bilayer graphene), 
    automatically handles/ignores Li atoms as specified.

    Parameters:
        ref_atoms (ase.Atoms): Reference structure
        cand_atoms (ase.Atoms): Candidate structure
        include_li (bool): Whether to include Li atoms in the matching (default False)

    Returns:
        float: RMSD value (Å), returns 999.0 if matching fails.
    """
    import numpy as np
    from pymatgen.core import Structure
    from pymatgen.analysis.structure_matcher import StructureMatcher

    # === 1. Convert to pymatgen.Structure ===
    ref_struct = Structure(
        lattice=ref_atoms.cell.array,
        species=ref_atoms.get_chemical_symbols(),
        coords=ref_atoms.get_positions(),
        coords_are_cartesian=True
    )
    cand_struct = Structure(
        lattice=cand_atoms.cell.array,
        species=cand_atoms.get_chemical_symbols(),
        coords=cand_atoms.get_positions(),
        coords_are_cartesian=True
    )

    # === 2. Construct Matcher ===
    matcher = StructureMatcher(
        ltol=0.01,                          # Lattice length tolerance (fixed lattice -> very small)
        stol=0.2,                           # Atomic position tolerance
        angle_tol=1,                        # Lattice angle tolerance (fixed lattice -> very small)
        primitive_cell=False,               # Do not reduce to primitive cell
        scale=False,                        # Do not scale lattice volume
        attempt_supercell=False,            # Do not attempt supercell matching
        ignored_species=["Li"] if not include_li else []  # Automatically ignore Li atoms
    )

    # === 3. Calculate RMSD ===
    try:
        result = matcher.get_rms_dist(ref_struct, cand_struct)
        if result is None:
            return 999.0

        rms_norm, _ = result  # (Normalized RMSD, max distance)
        # Restore to physical units (Å)
        norm_factor = (ref_struct.volume / len(ref_struct)) ** (1 / 3)
        rmsd = rms_norm * norm_factor
        return float(rmsd)

    except Exception as e:
        print(f"[WARN] RMSD calculation failed: {e}")
        return 999.0


# ---------- Main Workflow ----------
def check_reference(ref_file, scan_dir, threshold=0.1, use_li_alignment=True,
                    layer_tol=None, shift_tol=None, include_all_atoms=True):
    """
    Scans structures in scan_dir to find configurations close to the ref structure.
    Criteria:
      1. RMSD < threshold;
      2. If layer_tol/shift_tol are provided, apply additional interlayer distance 
         and slip difference criteria.
    Final results are sorted by RMSD in ascending order.
    """
    ref = read(ref_file)
    scan_dir = Path(scan_dir)

    # ========= 1. Scan target files =========
    file_list = sorted(scan_dir.glob("*.extxyz"))
    mode_ext = True
    if not file_list:
        file_list = sorted(scan_dir.glob("gen_*.xyz"))
        mode_ext = False

    # ========= 2. Calculate reference structure information =========
    ref_layer = layer_distance(ref) if (layer_tol is not None) else None
    ref_shift = stacking_shift(ref) if (shift_tol is not None) else None

    found = []

    # ========= 3. Scan candidate structures =========
    for f in file_list:
        atoms_list = [read(f)] if mode_ext else read(f, index=":")
        for atoms in atoms_list:
            # --- RMSD Calculation ---
            # (Note: rmsd_direct is assumed to be an alternative method not defined in snippet)
            d_rmsd = rmsd_with_li_alignment(ref, atoms, include_all_atoms)
            
            if not np.isfinite(d_rmsd):
                continue

            ok = (d_rmsd < threshold)
            d_layer = d_shift = None

            # --- Interlayer distance and slip difference check ---
            if ok and (layer_tol is not None or shift_tol is not None):
                if layer_tol is not None:
                    cand_layer = layer_distance(atoms)
                    d_layer = abs(cand_layer - ref_layer)
                    ok = ok and (d_layer <= layer_tol)
                if shift_tol is not None:
                    cand_shift = stacking_shift(atoms)
                    d_shift = abs(cand_shift - ref_shift)
                    ok = ok and (d_shift <= shift_tol)

            if ok:
                found.append((f.name, atoms.info.get("confid", None), d_rmsd, d_layer, d_shift))

    # ========= 4. Output results =========
    if found:
        # Sort by RMSD ascending
        found.sort(key=lambda x: x[2])

        print(f"✅ Found {len(found)} candidates close to the reference:")
        print(f"{'file name':28s} | {'gaid':>6s} | {'RMSD(Å)':>12s} | {'Δd(Å)':>12s} | {'Δshift(Å)':>14s}")
        for fname, gaid, d_rmsd, d_layer, d_shift in found:
            d_rmsd_str = f"{d_rmsd:12.6f}" if d_rmsd is not None else f"{'—':>12s}"
            d_layer_str = f"{d_layer:12.6f}" if d_layer is not None else f"{'—':>12s}"
            d_shift_str = f"{d_shift:14.6f}" if d_shift is not None else f"{'—':>14s}"
            print(f"{fname:28s} | {str(gaid):>6s} | {d_rmsd_str} | {d_layer_str} | {d_shift_str}")
    else:
        print("No structures found close to the reference.")


def main():
    p = argparse.ArgumentParser(description="Check if a reference structure appears in a candidate set (RMSD with optional interlayer/stacking constraints).")
    p.add_argument("--ref", help="Reference structure (extxyz/xyz/POSCAR)")
    p.add_argument("--scan_dir", default="ga_outputs/gens_xyz", help="Directory to scan: e.g., gens_xyz or validation_candidates")
    p.add_argument("--threshold", type=float, default=0.10, help="RMSD threshold (Å)")
    p.add_argument("--no_li_align", action="store_true", help="Disable Li anchor alignment (enabled by default)")
    p.add_argument("--layer_tol", type=float, default=None, help="Interlayer distance difference threshold (Å)")
    p.add_argument("--shift_tol", type=float, default=None, help="Stacking slip threshold (Å)")
    p.add_argument("--carbon_only", action="store_true", help="Use only the carbon framework for RMSD (default includes all atoms)")
    args = p.parse_args()

    check_reference(
        args.ref, args.scan_dir, args.threshold,
        use_li_alignment=not args.no_li_align,
        layer_tol=args.layer_tol, shift_tol=args.shift_tol,
        include_all_atoms=not args.carbon_only
    )

if __name__ == "__main__":
    main()