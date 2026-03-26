# Reproduction Guide for DESSP

This guide describes the procedures to reproduce the results for the Data-Efficient Stable Structure Prediction (DESSP) framework. The workflow is divided into two main stages: MLIP Training (Distillation and Calibration) and Batch Global Structure Search.

## 1. Model Training

The training process involves a two-stage strategy: distilling knowledge from a universal MLIP (uMLIP) and calibrating the result with DFT-labeled data. We provide a shell script to automate this process.

### Training Execution

To start the training for the Li-intercalated BLG system, execute the provided bash script:

```
# Ensure you are in the project root directory
bash MACE_DESSP.sh
```

This script handles:

- Data preprocessing and pseudo-label generation using the uMLIP.
- The distillation stage to capture the overall potential energy surface (PES).
- The calibration stage against high-fidelity DFT labels to recover structural stability accuracy.

Configuration parameters such as learning rate, batch size, and model architecture are managed within the `./configs/MACE_DESSP.yaml` file referenced by the script.

## 2. Global Structure Search

Once the MLIP model is trained or if you are using the provided surrogate models, you can perform global structure searches across various supercells and concentrations.

### Batch Execution

For large-scale production runs and to reproduce the thermodynamic convex hull data, use the `run_campaign.py` script. This script acts as a high-level scheduler that manages multiple GA search tasks across available hardware.

```
# Execute batch structure search campaigns
python run_campaign.py
```

### Script Functionality

- **`run_campaign.py`**: Batch schedules multiple GA instances. It automatically assigns tasks to available GPUs and manages the workspace for each concentration/supercell configuration.
- **Internal Scheduling**: It leverages `run_ga.py` for individual search trajectories while maintaining hardware efficiency (binding tasks to specific GPU IDs).

## 3. Core File Usage Summary

| **File**            | **Purpose**                                          | **Execution Command**         |
| ------------------- | ---------------------------------------------------- | ----------------------------- |
| `MACE_DESSP.sh`     | Main training pipeline (Distillation + Calibration)  | `bash MACE_DESSP.sh`          |
| `run_campaign.py`   | Batch structural search and production runs          | `python run_campaign.py`      |
| `run_ga.py`         | Individual Genetic Algorithm search engine           | (Called by `run_campaign.py`) |

By following these steps, you can reproduce the structural predictions and thermodynamic analysis reported in our work.
