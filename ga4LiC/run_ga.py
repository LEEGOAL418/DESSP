#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_ga.py

"""

import os
import sys
import warnings
import time
import argparse
import logging
import csv
import multiprocessing as mp
from pathlib import Path

# --- 1. Environment configuration (to prevent CPU oversubscription) ---
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")

# --- Resource-tracking issue (optional workaround) ---
try:
    from multiprocessing import resource_tracker
    def fix_register(name, rtype):
        if rtype == 'semaphore':
            return
        return resource_tracker._resource_tracker.register(name, rtype)
except ImportError:
    pass

import torch
import torch._dynamo
torch._dynamo.config.suppress_errors = True
import numpy as np

# --- ASE components ---
from ase.io import write
from ase.ga import set_raw_score
from ase.ga.data import DataConnection
from ase.ga.population import Population
from ase.ga.offspring_creator import OperationSelector
from ase.optimize import LBFGS
from ase.optimize.sciopt import SciPyFminCG
from ase.constraints import FixedLine

# --- DScribe ---
from dscribe.descriptors import SOAP

# --- Custom modules ---
from calculator import get_calculator
from ga_plotter import plot_ga_results
from ga_operators import (
    LiVacancySwapOperator,
    InplaneShiftOperator,
    OutplaneStretchOperator,
)

# ==============================================================
# ================= Structure Validator =========================
# ==============================================================

class StructureValidator:
    def __init__(self, min_distance=0.85, max_energy=-50.0):
        self.min_distance = min_distance
        self.max_energy = max_energy

    def is_valid(self, atoms, energy=None):
        # 1) distance-based rejection
        all_dists = atoms.get_all_distances(mic=True) + np.eye(len(atoms)) * 100
        min_dist = np.min(all_dists)
        if min_dist < self.min_distance:
            return False, f"Clash (MinDist={min_dist:.3f} < {self.min_distance})"

        # 2) energy-based rejection (optional)
        if energy is not None:
            if energy > self.max_energy:
                return False, f"High Energy (E={energy:.3f} > {self.max_energy})"

        return True, "OK"

# ==============================================================
# ================= Kernel SOAP Comparator ======================
# ==============================================================

class KernelSOAPComparator:
    def __init__(
        self,
        species=("C", "Li"),
        r_cut=6.0,
        n_max=8,
        l_max=6,
        sigma=0.2,
        similarity_threshold=0.999
    ):
        self.similarity_threshold = similarity_threshold
        self.soap = SOAP(
            species=list(species),
            periodic=True,
            r_cut=r_cut,
            n_max=n_max,
            l_max=l_max,
            sigma=sigma,
            average="off",
            sparse=False
        )

    def get_fingerprint(self, atoms):
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

        # Kernel similarity (dot product)
        similarity_matrix = np.dot(fp1, fp2.T)
        best_match_a = np.max(similarity_matrix, axis=1)
        best_match_b = np.max(similarity_matrix, axis=0)
        final_score = (np.mean(best_match_a) + np.mean(best_match_b)) / 2.0

        return final_score > self.similarity_threshold, float(final_score)

# ==============================================================
# ================= Worker Functions ============================
# ==============================================================

worker_calc = None

def relax_worker_init(calc_name, model_path, device):
    """Load the calculator once at worker startup to reduce repeated initialization overhead."""
    global worker_calc
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'

    try:
        if calc_name == "mace" and model_path:
            worker_calc = get_calculator(name=calc_name, model_path=model_path, device=device)
        else:
            worker_calc = get_calculator(name=calc_name, device=device)
    except Exception:
        worker_calc = None

def relax_worker_task(task_data):
    """
    task_data:
      atoms, fmax, steps, opt_name, meta_info, remove_constraints_after
    """
    atoms, fmax, steps, opt_name, meta_info, remove_constraints_after = task_data
    global worker_calc

    if worker_calc is None:
        return None, "Calc not init", 0, 0.0, meta_info

    try:
        atoms.calc = worker_calc
        start_time = time.time()

        if opt_name == "BFGS":
            dyn = LBFGS(atoms, trajectory=None, logfile=None)
        elif opt_name == "CG":
            dyn = SciPyFminCG(atoms, trajectory=None, logfile=None)
        else:
            dyn = LBFGS(atoms, trajectory=None, logfile=None)

        dyn.run(fmax=fmax, steps=steps)

        opt_time = time.time() - start_time
        opt_steps = dyn.get_number_of_steps()
        E = atoms.get_potential_energy()

        # Remove the calculator to avoid serializing calculator information when writing files
        atoms.calc = None

        # Optionally remove constraints after optimization to avoid extxyz write errors caused by list-style constraints
        if remove_constraints_after:
            atoms.set_constraint(None)

        # Clear the SOAP cache
        if 'soap_local_fp' in atoms.info:
            del atoms.info['soap_local_fp']

        return atoms, float(E), int(opt_steps), float(opt_time), meta_info

    except Exception as e:
        return None, f"Error: {str(e)}", 0, 0.0, meta_info

# ==============================================================
# ================= Main Program ================================
# ==============================================================

def setup_logging(log_dir="."):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    fh_debug = logging.FileHandler(os.path.join(log_dir, "run_debug.log"), mode='w')
    fh_debug.setLevel(logging.DEBUG)
    fh_debug.setFormatter(formatter)
    logger.addHandler(fh_debug)

    fh_info = logging.FileHandler(os.path.join(log_dir, "run.log"), mode='w')
    fh_info.setLevel(logging.INFO)
    fh_info.setFormatter(formatter)
    logger.addHandler(fh_info)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger

def parse_args():
    p = argparse.ArgumentParser(description="Run Kernel-SOAP GA (Two-Stage Relax + Early Stop)")

    # Basic
    p.add_argument("--db_file", type=str, default="ga_test.db", help="Database file name")
    p.add_argument("--li_sites", type=str, default="Li-sites.txt", help="Lithium-site definition file")
    p.add_argument("--num_gen", type=int, default=50, help="Maximum number of evolutionary generations (upper bound, used together with early stopping)")
    p.add_argument("--calc", type=str, default="orb", help="Force field / potential (mace/orb)")
    p.add_argument("--model_path", type=str, default=None, help="Path to the MACE model")
    p.add_argument("--pop_size", type=int, default=None, help="Population size (defaults to the population_size parameter stored in the database)")
    p.add_argument("--optimizer", type=str, default="BFGS", help="Optimizer (Search stage)")
    p.add_argument("--relax_jobs", type=int, default=4, help="Number of parallel relaxation workers")

    # Two-stage relaxation params
    p.add_argument("--search_steps", type=int, default=50, help="Search stage: maximum number of relaxation steps")
    p.add_argument("--search_fmax", type=float, default=0.10, help="Search stage: fmax")
    p.add_argument("--final_steps", type=int, default=100, help="Final stage: maximum number of relaxation steps (strict)")
    p.add_argument("--final_fmax", type=float, default=0.01, help="Final stage: fmax (strict)")
    p.add_argument("--final_topk", type=int, default=30, help="Final stage: Top-K count for strict re-relaxation")
    p.add_argument("--final_optimizer", type=str, default="BFGS", help="Final stage: optimizer (strict)")
    p.add_argument("--final_update_db", action="store_true", help="Whether to write Final strict energies back to the database (disabled by default)")

    # Constraint
    p.add_argument("--constrain_prob", type=float, default=0.2, help="Probability of applying Z-only constraints to offspring (0.0-1.0)")

    # SOAP / basin logic
    p.add_argument("--strict_sim", type=float, default=0.9998, help="Strict duplicate-detection threshold (generation stage)")
    p.add_argument("--loose_sim", type=float, default=0.9995, help="Loose similarity threshold (database insertion stage)")
    p.add_argument("--energy_diff_threshold", type=float, default=0.005, help="Energy-difference threshold for similar structures (used to classify intermediates)")

    # Early stop (convergence)
    p.add_argument("--early_stop", action="store_true", help="Enable built-in early stopping")
    p.add_argument("--min_gens", type=int, default=10, help="Minimum number of generations before early stopping is allowed (to prevent premature termination)")
    p.add_argument("--min_delta_E", type=float, default=0.003, help="Energy-improvement threshold (eV)")
    p.add_argument("--patience_E", type=int, default=10, help="Patience for energy stagnation")
    p.add_argument("--min_accept_ratio", type=float, default=0.15, help="Novelty threshold accepted/pop_size")
    p.add_argument("--patience_novelty", type=int, default=6, help="Patience for novelty stagnation")
    p.add_argument("--early_stop_mode", type=str, default="AND", choices=["AND", "OR"],
                   help="Early-stop trigger mode: AND = stop only when both criteria stall; OR = stop when either criterion stalls")

    # Final Top-K basin selection (diversity)
    p.add_argument("--final_diversity_sim", type=float, default=None,
                   help="SOAP similarity threshold for Final Top-K basin selection; None means loose_sim is reused")
    p.add_argument("--final_max_scan", type=int, default=2000,
                   help="Maximum number of candidates scanned during Final basin selection (in ascending energy order)")
    p.add_argument("--final_diag_pairs", type=int, default=120,
                   help="Number of sampled pairs for Final diversity diagnostics (0 disables diagnostics)")
    # Skip Final stage (for strict-search + early-stop workflow)
    p.add_argument("--skip_final", action="store_true",
                   help="Skip the Final Top-K strict re-relaxation stage after GA termination (while retaining GA search and early stopping)")

    return p.parse_args()

def _safe_get_raw_E(atoms):
    """
    Read raw_score from atoms and convert it to E
    set_raw_score(atoms, -E) 之后，raw_score = -E
    """
    kv = atoms.info.get("key_value_pairs", {})
    raw_score = kv.get("raw_score", None)
    if raw_score is None:
        # Some structures may not have raw_score recorded (defensive fallback)
        try:
            return float(atoms.get_potential_energy())
        except Exception:
            return np.nan
    return float(-1.0 * raw_score)

def _get_all_relaxed_candidates(db):
    """
    The ASE GA DataConnection API differs slightly across versions.
    A compatibility-oriented fallback is used here:
    """
    for fn in ["get_all_relaxed_candidates", "get_all_relaxed_steps"]:
        if hasattr(db, fn):
            return getattr(db, fn)()
    # Fallback: if the API is unavailable, return an empty list
    return []

# ==============================================================
# ===== Final diversity diagnostics (极轻量) =====================
# ==============================================================

def _kernel_soap_similarity(comp: KernelSOAPComparator, a1, a2):
    """
    Compute the Kernel SOAP similarity score (return the score only, without thresholding).
    Reuse the fingerprint cache in KernelSOAPComparator to avoid redundant computation.
    """
    fp1 = comp.get_fingerprint(a1)
    fp2 = comp.get_fingerprint(a2)
    if fp1 is None or fp2 is None:
        return np.nan
    sim_mat = np.dot(fp1, fp2.T)
    best_a = np.max(sim_mat, axis=1)
    best_b = np.max(sim_mat, axis=0)
    return float((np.mean(best_a) + np.mean(best_b)) / 2.0)

def _final_diversity_diagnose(selected_atoms, comp: KernelSOAPComparator, logger, n_pairs=120):
    """
    Perform an ultralightweight diversity diagnosis on the selected representative set:
    - randomly sample a number of pairs (120 by default) and compute their Kernel SOAP similarities;
    - report min/mean/max/p5/p50/p95 together with the number of sampled pairs.
    """
    n = len(selected_atoms)
    if n < 2 or n_pairs <= 0:
        return

    # 允许的 pair 数上限
    max_pairs = n * (n - 1) // 2
    n_pairs = int(min(max_pairs, max(1, n_pairs)))

    rng = np.random.default_rng(2026)
    sims = []

    # 抽样策略：随机抽样 pair index（避免 O(n^2)）
    for _ in range(n_pairs):
        i = int(rng.integers(0, n))
        j = int(rng.integers(0, n - 1))
        if j >= i:
            j += 1
        s = _kernel_soap_similarity(comp, selected_atoms[i], selected_atoms[j])
        if np.isfinite(s):
            sims.append(s)

    if not sims:
        logger.info("[Final Diversity] Diagnostic skipped (no valid similarity computed).")
        return

    sims = np.array(sims, dtype=float)
    sims.sort()

    def _pct(x):
        return float(np.percentile(sims, x))

    logger.info(
        "[Final Diversity] KernelSOAP similarity (sampled) "
        f"| n_selected={n} | n_pairs={len(sims)} "
        f"| min={sims[0]:.6f} mean={float(np.mean(sims)):.6f} max={sims[-1]:.6f} "
        f"| p5={_pct(5):.6f} p50={_pct(50):.6f} p95={_pct(95):.6f}"
    )

def final_strict_relax_topk(db, pool, args, logger, out_dir: Path):
    """
    Retrieve all relaxed candidates from the database and sort them by search-stage energy,
    first perform Final Top-K basin selection (Kernel SOAP deduplication for representative selection),
    then perform strict re-relaxation of the representative structures using final_fmax/final_steps and write the final output files.

    输出：
      ga_outputs/final/final_selected_search.xyz  （Final basin 代表，search 阶段结构）
      ga_outputs/final/final_relax_log.csv
      ga_outputs/final/final_topk_relaxed.xyz
    """
    all_relaxed = _get_all_relaxed_candidates(db)
    if not all_relaxed:
        logger.warning("Final strict relax skipped: no relaxed candidates found in DB.")
        return

    # Sort by the raw_score recorded during the search stage (lower energy is better)
    scored = []
    for a in all_relaxed:
        E_search = _safe_get_raw_E(a)
        if np.isfinite(E_search):
            scored.append((float(E_search), a))
    if not scored:
        logger.warning("Final strict relax skipped: cannot read energies from candidates.")
        return

    scored.sort(key=lambda x: x[0])

    # ==========================================================
    # Final Top-K basin selection: scan from low to high energy + SOAP-based deduplication
    # ==========================================================
    final_dir = out_dir / "final"
    final_dir.mkdir(exist_ok=True, parents=True)

    diversity_thr = args.final_diversity_sim if args.final_diversity_sim is not None else args.loose_sim
    max_scan = int(args.final_max_scan)
    topk_target = max(1, int(args.final_topk))

    # Comparator for basin selection: parameters follow the strict/loose SOAP settings to remain consistent with comp_* defined above
    comp_div = KernelSOAPComparator(
        r_cut=10.0, n_max=10, l_max=8, sigma=0.2, similarity_threshold=float(diversity_thr)
    )

    selected = []              # list[(E_search, atoms)]
    selected_scan_rank = []    # list[int] 1-based rank in scored (for tracing)
    scanned = 0

    for idx, (E_search, a) in enumerate(scored, start=1):
        scanned += 1
        if scanned > max_scan:
            break

        # Check for duplication against the selected representative set
        is_dup = False
        best_sim = -1.0
        for (E0, a0) in selected:
            is_sim, sim_val = comp_div.looks_like(a, a0)
            if sim_val > best_sim:
                best_sim = sim_val
            if is_sim:
                is_dup = True
                break

        if is_dup:
            continue

        selected.append((E_search, a))
        selected_scan_rank.append(idx)

        if len(selected) >= topk_target:
            break

    if not selected:
        logger.warning("Final strict relax skipped: basin selection returned empty.")
        return

    logger.info(
        f"=== Final Top-K basin selection: selected={len(selected)}/{topk_target}, "
        f"scanned={min(scanned, max_scan)}/{max_scan}, diversity_sim={float(diversity_thr):.6f} ==="
    )

    # Write the selected representatives (search-stage structures) to facilitate manual inspection of diversity
    selected_search_xyz = final_dir / "final_selected_search.xyz"
    try:
        write(selected_search_xyz, [a.copy() for _, a in selected], format="extxyz")
        logger.info(f"Final selected (search) structures saved: {selected_search_xyz.resolve()}")
    except Exception as e:
        logger.warning(f"Writing final_selected_search.xyz failed: {e}")

    # Ultralightweight diversity diagnostics (sampled pairs)
    try:
        _final_diversity_diagnose([a for _, a in selected], comp_div, logger, n_pairs=int(args.final_diag_pairs))
    except Exception as e:
        logger.warning(f"[Final Diversity] Diagnostic failed: {e}")

    # ==========================================================
    # Final Strict Relax：对代表结构做严格重松弛
    # ==========================================================
    logger.info(f"=== Final Strict Relax: Basin-Top-{len(selected)} representatives ===")

    final_csv = final_dir / "final_relax_log.csv"
    final_xyz = final_dir / "final_topk_relaxed.xyz"

    # 写表头
    with open(final_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "rank", "confid", "gen", "op", "desc",
            "E_search(eV)", "E_final(eV)", "delta(eV)",
            "final_steps", "final_fmax", "opt_steps", "opt_time(s)",
            "diversity_sim_used", "scan_rank_in_db"
        ])

    # Assemble tasks: by default, no constraints are retained during strict re-relaxation (even if present in the structure object, they are removed at the end of the worker routine)
    tasks = []
    metas = []
    for rank, ((E_search, a), scan_rank) in enumerate(zip(selected, selected_scan_rank), start=1):
        aa = a.copy()
        # Clear the SOAP cache to avoid unnecessary serialization
        if 'soap_local_fp' in aa.info:
            del aa.info['soap_local_fp']

        meta = {
            "rank": rank,
            "E_search": float(E_search),
            "confid": aa.info.get("confid", ""),
            "gen": aa.info.get("key_value_pairs", {}).get("gen", ""),
            "op": aa.info.get("key_value_pairs", {}).get("op", ""),
            "desc": aa.info.get("desc", ""),
            "scan_rank": int(scan_rank)
        }
        metas.append(meta)
        tasks.append((aa, args.final_fmax, args.final_steps, args.final_optimizer, meta, True))

    results = list(pool.map(relax_worker_task, tasks))

    # Collect and write outputs
    final_atoms = []
    best_final = None

    for atoms_rel, E_final, opt_steps, opt_time, meta in results:
        if atoms_rel is None or isinstance(E_final, str):
            # Failures may also be logged here if desired
            continue

        delta = float(E_final) - float(meta["E_search"])
        final_atoms.append(atoms_rel)

        with open(final_csv, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                meta["rank"], meta.get("confid", ""), meta.get("gen", ""), meta.get("op", ""), meta.get("desc", ""),
                f"{meta['E_search']:.6f}", f"{float(E_final):.6f}", f"{delta:.6f}",
                args.final_steps, args.final_fmax, opt_steps, f"{opt_time:.2f}",
                f"{float(diversity_thr):.6f}", meta.get("scan_rank", "")
            ])

        if best_final is None or float(E_final) < best_final[0]:
            best_final = (float(E_final), atoms_rel)

        # Optional: write back to the database (disabled by default)
        if args.final_update_db:
            try:
                set_raw_score(atoms_rel, -float(E_final))
                db.add_relaxed_step(atoms_rel)
            except Exception:
                pass

    if final_atoms:
        write(final_xyz, final_atoms, format="extxyz")
        logger.info(f"Final strict structures saved: {final_xyz.resolve()}")
        logger.info(f"Final strict log saved: {final_csv.resolve()}")
        if best_final is not None:
            logger.info(f"Final best (strict) E = {best_final[0]:.6f} eV")
    else:
        logger.warning("Final strict relax produced no valid structures.")

def main():
    args = parse_args()
    logger = setup_logging()

    validator = StructureValidator(min_distance=0.85, max_energy=-50.0)

    logger.info("=== GA Optimization Started ===")
    logger.info(f"Constraint Probability: {args.constrain_prob * 100:.1f}% (Fix C atoms in XY -> Z-only)")
    logger.info(f"Search relax: steps={args.search_steps}, fmax={args.search_fmax}")
    if args.skip_final:
        logger.info("Final  relax: SKIPPED (--skip_final enabled)")
    else:
        logger.info(f"Final  relax: steps={args.final_steps}, fmax={args.final_fmax}, topk={args.final_topk}")

    logger.info(f"Early stop: {args.early_stop} | mode={args.early_stop_mode} | min_gens={args.min_gens}")

    # Comparators
    comp_strict = KernelSOAPComparator(
        r_cut=10.0, n_max=10, l_max=8, sigma=0.2, similarity_threshold=args.strict_sim
    )
    comp_loose = KernelSOAPComparator(
        r_cut=10.0, n_max=10, l_max=8, sigma=0.2, similarity_threshold=args.loose_sim
    )
    ENERGY_DIFF_THRESHOLD = float(args.energy_diff_threshold)

    # --- Step 0: Preparation stage ---
    db = DataConnection(args.db_file)
    pop_size = args.pop_size if args.pop_size else db.get_param("population_size")

    if os.path.exists(args.li_sites):
        li_sites = np.loadtxt(args.li_sites)
    else:
        logger.error(f"Li sites file not found: {args.li_sites}")
        return

    oclist = [
        (1, LiVacancySwapOperator(li_sites)),
        (2, InplaneShiftOperator(layer="top", max_disp=1.46)),
        (2, OutplaneStretchOperator(max_disp=0.5)),
    ]
    operation_selector = OperationSelector(*zip(*oclist))
    pop = Population(data_connection=db, population_size=pop_size, comparator=None, logfile=None)

    out_dir = Path("ga_outputs")
    xyz_dir = out_dir / "gens_xyz"
    pool_dir = out_dir / "pool_xyz"
    for d in [out_dir, xyz_dir, pool_dir]:
        d.mkdir(exist_ok=True, parents=True)

    csv_path = out_dir / "offspring_log.csv"
    if not csv_path.exists():
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow([
                "gen", "confid", "raw_E(eV)", "neg_E", "op", "parents", "desc",
                "opt_steps", "opt_time", "is_intermediate"
            ])

    # --- Step 1: Initial population relaxation (serial)---
    # Note: the initial population is also recommended to use the Search-stage parameters (consistent with those used inside GA), to avoid sample collapse caused by overly strict convergence of the initial structures.
    num_unrelaxed = db.get_number_of_unrelaxed_candidates()
    if num_unrelaxed > 0:
        logger.info(f"Relaxing initial candidates... (Total: {num_unrelaxed})")
        logger.info("Loading Calculator for Init Population (Main Process)...")

        debug_calc = get_calculator(args.calc, model_path=args.model_path, device='cuda')

        cnt = 0
        while db.get_number_of_unrelaxed_candidates() > 0:
            a = db.get_an_unrelaxed_candidate()
            a.calc = debug_calc
            
            # Record the start time
            t0 = time.time()

            if args.optimizer == "CG":
                dyn = SciPyFminCG(a, logfile=None)
            else:
                dyn = LBFGS(a, logfile=None)

            try:
                dyn.run(fmax=args.search_fmax, steps=args.search_steps)
                E = a.get_potential_energy()

                # --- [修改点 B] Retrieve the number of steps and elapsed time, and store them in info ---
                opt_steps = dyn.get_number_of_steps()
                opt_time = time.time() - t0
                a.info['init_opt_steps'] = opt_steps
                a.info['init_opt_time'] = opt_time

                cnt += 1
                logger.info(f"Init Relax #{cnt} | E={E:.8f} eV | Steps: {opt_steps}")

                set_raw_score(a, -E)
                db.add_relaxed_step(a)

            except Exception as e:
                logger.error(f"Init Relax Error: {e}")

            a.calc = None
            # 清理约束与缓存（init 通常无约束，但这里保险）
            a.set_constraint(None)
            if 'soap_local_fp' in a.info:
                del a.info['soap_local_fp']

        del debug_calc
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Initial relaxation done. Calculator unloaded.")

    # --- [修改点 C] Write the generation-0 log ---
    # This block is intentionally placed here, i.e., immediately after initial relaxation is completed
    pop.update()
    logger.info("Writing Gen 0 info to log...")
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        for atoms in pop.pop:
            origin_gen = atoms.info.get("gen", 0)
            # If gen is 0 or absent, the structure is treated as an initial structure
            if origin_gen == 0:
                confid = atoms.info.get("confid", "init")
                raw_score = atoms.info.get("key_value_pairs", {}).get("raw_score", 0.0)
                E = -1.0 * raw_score
                
                # Retrieve the variables saved above; if unavailable (e.g., skipped due to an error), use 0 as a fallback
                o_steps = atoms.info.get('init_opt_steps', 0)
                o_time = atoms.info.get('init_opt_time', 0.0)
                
                w.writerow([0, confid, f"{E:.8f}", f"{-E:.8f}", "init", "[]", "random_init", o_steps, f"{o_time:.2f}", False])

    # =========================================================================
    # Step 2: Initialize the worker pool (shared by both Search relaxation and Final strict re-relaxation)
    # =========================================================================
    ctx = mp.get_context("spawn")

    logger.info(f"Initializing Worker Pool ({args.relax_jobs} workers)...")
    logger.info("Loading MLIP models on workers...")

    # ---- State variables for early stopping ----
    best_E_so_far = None
    no_improve_E_gens = 0
    no_novelty_gens = 0

    with ctx.Pool(
        processes=args.relax_jobs,
        initializer=relax_worker_init,
        initargs=(args.calc, args.model_path, 'cuda:0')
    ) as pool:

        logger.info("Worker Pool Ready.")

        for gen in range(args.num_gen):
            gen_start_time = time.time()
            logger.info(f"--- Gen {gen+1}/{args.num_gen} ---")

            candidates = []
            candidates_meta = []

            attempts = 0
            generated = 0
            max_attempts = 200

            # -------------------------------
            # Candidate generation: strict duplicate detection (comp_strict)
            # -------------------------------
            while generated < pop_size and attempts < max_attempts:
                attempts += 1

                op = operation_selector.get_operator()
                parents = pop.get_two_candidates() if "Crossover" in op.__class__.__name__ else [pop.get_one_candidate()]
                offspring, desc = op.get_new_individual(parents)

                if offspring is None:
                    continue

                offspring.rattle(stdev=0.02)
                if 'soap_local_fp' in offspring.info:
                    del offspring.info['soap_local_fp']

                # Distance check
                is_valid, _ = validator.is_valid(offspring, energy=None)
                if not is_valid:
                    continue

                # Strict duplicate detection against the population pool
                is_duplicate = False
                max_similarity = -1.0

                for p in pop.pop:
                    is_sim, sim_val = comp_strict.looks_like(offspring, p)
                    if sim_val > max_similarity:
                        max_similarity = sim_val
                    if is_sim:
                        is_duplicate = True
                        break

                if is_duplicate:
                    logger.info(f"Att {attempts:02d} | Dup in Pool | Sim={max_similarity:.6f} | REJECT")
                    continue

                # Strict duplicate detection within the current batch
                is_batch_duplicate = False
                for cand in candidates:
                    is_sim, _ = comp_strict.looks_like(offspring, cand)
                    if is_sim:
                        is_batch_duplicate = True
                        break
                if is_batch_duplicate:
                    logger.info(f"Att {attempts:02d} | Dup in Batch | REJECT")
                    continue

                # -----------------------------------------
                # Randomized constraint logic (applied probabilistically): Z-only constraints on C atoms only
                # -----------------------------------------
                constraint_applied = False
                constraint_tag = ""

                if np.random.random() < args.constrain_prob:
                    carbon_indices = [atom.index for atom in offspring if atom.symbol == 'C']
                    if carbon_indices:
                        constraints = FixedLine(indices=carbon_indices, direction=[0, 0, 1])
                        offspring.set_constraint(constraints)
                        constraint_applied = True
                        constraint_tag = " [Cons:Z-Only]"

                log_msg = f"Att {attempts:02d} | Sim={max_similarity:.6f} | ACCEPT"
                if constraint_applied:
                    log_msg += " | Constrained"
                logger.info(log_msg)

                # Clear the SOAP cache
                if 'soap_local_fp' in offspring.info:
                    del offspring.info['soap_local_fp']

                candidates.append(offspring)
                candidates_meta.append({
                    "op": op.__class__.__name__,
                    "parents": offspring.info.get("data", {}).get("parents"),
                    "desc": desc + constraint_tag
                })
                generated += 1

            logger.info(f"Generated {generated} candidates. Starting Parallel Relaxation...")
            if generated == 0:
                logger.warning("No candidates generated in this generation. Continue.")
                continue

            # -------------------------------
            # Search relaxation (parallel): use search_steps/search_fmax
            # -------------------------------
            relax_tasks = [
                (at, args.search_fmax, args.search_steps, args.optimizer, candidates_meta[i], True)
                for i, at in enumerate(candidates)
            ]
            relaxed_results = list(pool.map(relax_worker_task, relax_tasks))

            # Valid energies in the current generation
            gen_energies = [r[1] for r in relaxed_results if (r[0] is not None and not isinstance(r[1], str))]

            # -------------------------------
            # Database insertion check (comp_loose + ΔE-based intermediate classification)
            # -------------------------------
            accepted_count = 0
            for i, res_tuple in enumerate(relaxed_results):
                atoms_rel, E, steps, t_opt, _ = res_tuple
                meta = candidates_meta[i]
                if atoms_rel is None or isinstance(E, str):
                    continue

                is_new_basin = True
                is_intermediate = False

                for existing in pop.pop:
                    looks_sim, _sim_val = comp_loose.looks_like(atoms_rel, existing)
                    if looks_sim:
                        # Note: existing.get_potential_energy() depends on an attached calculator and is therefore not always reliable
                        # Here, the energy of an existing structure is preferentially obtained from raw_score
                        E_exist = _safe_get_raw_E(existing)
                        if np.isfinite(E_exist) and abs(float(E) - float(E_exist)) > ENERGY_DIFF_THRESHOLD:
                            is_intermediate = True
                            is_new_basin = True
                            break
                        else:
                            is_new_basin = False
                            break

                if is_new_basin:
                    set_raw_score(atoms_rel, -float(E))
                    atoms_rel.info.setdefault("key_value_pairs", {})
                    atoms_rel.info["key_value_pairs"].update({"gen": gen+1, "op": meta["op"]})
                    atoms_rel.info["desc"] = meta.get("desc", "")

                    db.add_relaxed_step(atoms_rel)

                    with open(csv_path, "a", newline="") as f:
                        csv.writer(f).writerow([
                            gen+1,
                            atoms_rel.info.get("confid"),
                            f"{float(E):.8f}",
                            f"{-float(E):.8f}",
                            meta["op"],
                            meta["parents"],
                            meta["desc"],
                            steps,
                            f"{t_opt:.2f}",
                            is_intermediate
                        ])
                    accepted_count += 1

            pop.update()
            logger.info(f"Accepted: {accepted_count}/{len(candidates)}")

            # Write the population pool and structures of the current generation
            write(pool_dir / f"pool_{gen+1:03d}.xyz", pop.pop, format="extxyz")
            valid_relaxed = [r[0] for r in relaxed_results if r[0] is not None]
            if valid_relaxed:
                write(xyz_dir / f"gen_{gen+1:03d}.xyz", valid_relaxed, format="extxyz")

            # Best energy in the current generation
            gen_best_E = None
            if gen_energies:
                gen_best_E = float(np.min(gen_energies))
                logger.info(f"Best E (gen): {gen_best_E:.6f} eV")

            # -------------------------------
            # Early Stop：Convergence / early-stopping criteria
            # -------------------------------
            if args.early_stop and (gen + 1) >= args.min_gens:
                # 1) Energy-stagnation criterion
                improved = False
                if gen_best_E is not None:
                    if best_E_so_far is None:
                        best_E_so_far = gen_best_E
                        improved = True
                    else:
                        if (best_E_so_far - gen_best_E) > args.min_delta_E:
                            best_E_so_far = gen_best_E
                            improved = True

                if improved:
                    no_improve_E_gens = 0
                else:
                    no_improve_E_gens += 1

                # 2) Novelty-stagnation criterion
                accept_ratio = accepted_count / float(pop_size) if pop_size else 0.0
                if accept_ratio >= args.min_accept_ratio:
                    no_novelty_gens = 0
                else:
                    no_novelty_gens += 1

                logger.info(
                    f"[EarlyStop Monitor] best_E={best_E_so_far if best_E_so_far is not None else np.nan:.6f} "
                    f"| no_improve={no_improve_E_gens}/{args.patience_E} "
                    f"| accept_ratio={accept_ratio:.3f} "
                    f"| no_novelty={no_novelty_gens}/{args.patience_novelty}"
                )

                stalled_E = (no_improve_E_gens >= args.patience_E)
                stalled_N = (no_novelty_gens >= args.patience_novelty)

                if args.early_stop_mode == "AND":
                    should_stop = stalled_E and stalled_N
                else:
                    should_stop = stalled_E or stalled_N

                if should_stop:
                    logger.info(
                        f"=== Early Stop Triggered at Gen {gen+1} "
                        f"(stalled_E={stalled_E}, stalled_N={stalled_N}, mode={args.early_stop_mode}) ==="
                    )
                    break

            duration = time.time() - gen_start_time
            logger.info(f"--- Gen {gen+1} Finished in {duration:.2f} s ---")

        # =========================================================================
        # Step 3: Final Strict Relax Top-K（Reuse the worker pool）
        # =========================================================================
        if args.skip_final:
            logger.info("=== Skip Final Strict Relax Top-K (per --skip_final) ===")
        else:
            try:
                final_strict_relax_topk(db, pool, args, logger, out_dir)
            except Exception as e:
                logger.error(f"Final strict relax failed: {e}")


    logger.info("=== GA Finished ===")

    try:
        logger.info("Generating visualization plots...")
        figs_dir = out_dir / "figs"
        plot_ga_results(csv_path=str(csv_path), output_dir=str(figs_dir))
        logger.info(f"Plots saved to: {figs_dir.resolve()}")
    except Exception as e:
        logger.error(f"Plotting failed: {e}")

if __name__ == "__main__":
    main()
