#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calculator Module (calculator.py)

This module provides a unified interface to retrieve MACE, DFTB+, or ORB calculators.

The core design principle is "Dynamic Importing" to resolve issues where different 
calculators depend on distinct and potentially conflicting Python virtual environments. 
Each retrieval function attempts to import the required libraries internally. 
This way, as long as you do not call a specific calculator's function, you do not 
need to install its corresponding libraries, thus avoiding ImportErrors.

Usage:
In your main script, import the get_calculator function and call it by name:
from calculator import get_calculator
calc = get_calculator('mace', device='cpu')
# or
calc = get_calculator('dftb', slako_dir='/path/to/skf/')
"""

import os

def get_mace_calculator(model_path=None, device='cuda', **kwargs):
    """
    Get a MACE calculator instance.
    Imports mace dynamically inside the function to avoid environment conflicts.
    """
    try:
        from mace.calculators import MACECalculator
    except ImportError:
        print("Error: MACE calculator requires the 'mace-torch' package. Please run in the correct virtual environment.")
        raise

    if model_path is None:
        # Default model path
        model_path = "/home/user/Desktop/LHR/MLP-GA/MACE-GA/models/MACE_v6_finetune_orbga_plus_energy.model"

    print(f"Loading MACE calculator, model path: {model_path}")
    
    mace_kwargs = {
        'model_paths': model_path,
        'device': device,
        'default_dtype': 'float64'
    }
    mace_kwargs.update(kwargs)
    
    return MACECalculator(**mace_kwargs)


def get_dftb_calculator(slako_dir=None, **kwargs):
    """
    Get a DFTB+ calculator instance.
    Imports Dftb dynamically inside the function to avoid environment conflicts.
    """
    try:
        from ase.calculators.dftb import Dftb
    except ImportError:
        print("Error: DFTB+ calculator requires the ASE built-in dftbplus module. Please run in the correct virtual environment.")
        raise
    
    if slako_dir is None:
        # Default Slater-Koster files path
        slako_dir = "/home/user/Desktop/LHR/MACE-GA/skf_pram/Annie2021/slako/"
        
    print("Loading DFTB+ calculator (SCC-DFTB with UFF dispersion)")

    # Default settings updated based on provided parameters
    dftb_kwargs = {
        'label': 'calc/dftb',
        'kpts': (3, 3, 1),
        'Hamiltonian_SCC': "Yes",
        'Hamiltonian_MaxSCCIterations': 800,
        'Hamiltonian_SCCTolerance': 1e-5,
        'Hamiltonian_Filling': """Fermi {
            Temperature [Kelvin] = 600
        }""",
        'Hamiltonian_Mixer': """Broyden {
            MixingParameter = 0.5
        }""",
        'Hamiltonian_Dispersion': """LennardJones {
            Parameters = UFFParameters {}
        }""",
    }
    
    if slako_dir is not None:
        dftb_kwargs['slako_dir'] = slako_dir
        
    dftb_kwargs.update(kwargs)
    
    return Dftb(**dftb_kwargs)


def get_orb_calculator(device='cuda', **kwargs):
    """
    Get an ORB v3 calculator instance.
    [Debug Version] includes strict GPU checks and status printing.
    """
    # 1. Explicitly print incoming parameters for debugging
    print(f"DEBUG: get_orb_calculator called with device='{device}'")

    # 2. Core Check: Can PyTorch see the GPU?
    import torch
    if device == 'cuda':
        if not torch.cuda.is_available():
            print(" CRITICAL: device='cuda' but torch.cuda.is_available() is False!")
            print("   -> Potential Reason 1: CUDA_VISIBLE_DEVICES env var lost in sub-process")
            print("   -> Potential Reason 2: PyTorch version mismatch")
            raise RuntimeError("GPU requested but not available. Aborting.")
        
        # Print visible device information
        device_count = torch.cuda.device_count()
        print(f"DEBUG: torch sees {device_count} GPUs. Current device index: {torch.cuda.current_device()}")

    try:
        from orb_models.forcefield import pretrained
        from orb_models.forcefield.calculator import ORBCalculator
    except ImportError:
        print("Error: ORB calculator requires the 'orb-models' package. Please run in the correct virtual environment.")
        raise

    print(f"Loading ORB v3 direct inf omat calculator (Target: {device})")

    # Load model
    orbff = pretrained.orb_v3_direct_inf_omat(
        device=device,
        precision="float64",
    )
    
    # 3. Final Confirmation: Are model parameters actually on the GPU?
    param_device = next(orbff.parameters()).device
    print(f"DEBUG: Model loaded. First parameter is on: {param_device}")
    
    if device == 'cuda' and str(param_device) == 'cpu':
        raise RuntimeError("Model loaded onto CPU despite requesting CUDA! (Silent Fallback detected)")

    return ORBCalculator(orbff, device=device)


def get_calculator(name='mace', **kwargs):
    """
    Retrieve the corresponding calculator based on the name.
    This acts as a convenient dispatch function.

    :param name: Calculator name ('mace', 'dftb', 'orb')
    :param kwargs: Extra arguments passed to specific calculator constructors
    :return: An ASE calculator instance
    """
    name = name.lower()
    if name == 'mace':
        return get_mace_calculator(**kwargs)
    elif name == 'dftb':
        return get_dftb_calculator(**kwargs)
    elif name == 'orb':
        return get_orb_calculator(**kwargs)
    else:
        raise ValueError(f"Unknown calculator name '{name}'. Please choose from 'mace', 'dftb', or 'orb'.")

if __name__ == '__main__':
    # Usage demonstration
    # Ensure you are in the correct virtual environment before running
    
    # --- Example Test in MACE environment ---
    # calc_mace = get_calculator('mace', device='cpu')
    # print("Successfully retrieved MACE calculator:", calc_mace)
    
    # --- Example Test in DFTB+ environment ---
    # calc_dftb = get_calculator('dftb', slako_dir='/path/to/your/skf/files/')
    # print("Successfully retrieved DFTB+ calculator:", calc_dftb)

    # --- Example Test in ORB environment ---
    # calc_orb = get_calculator('orb', device='cpu')
    # print("Successfully retrieved ORB calculator:", calc_orb)
    pass