# Guide for DESSP

This guide describes the procedures to reproduce the results for the Data-Efficient Stable Structure Prediction (DESSP) framework. The workflow is divided into two main stages: MLIP Training (including Distillation and Calibration) and Batch Global Structure Search.

## 1. Model Training

The training process involves a multi-stage strategy depending on the desired model performance and comparison needs. We provide several shell scripts to automate these training workflows.

### Training Execution

To start the training, execute the corresponding bash script based on the model type:

- **DESSP Main Pipeline (Distillation + Calibration):**

  ```
  bash MACE_DESSP.sh
  ```

  This is the core implementation that performs uMLIP-to-task-specific distillation followed by high-fidelity DFT calibration.

- **Distilled Model Training:**

  ```
  bash MACE_ORB_Distilled.sh
  ```

  This script specifically handles the distillation stage, transferring the potential energy surface (PES) landscape knowledge from the universal MLIP (uMLIP) to the task-specific model.

- **Baseline Model Training (DFT-GO):**

  ```
  bash MACE_VASPGO.sh
  ```

  This serves as the baseline for performance comparison, training the MLIP using datasets obtained through standard local structural relaxations (DFT-GO).

Configuration parameters such as learning rate, batch size, and model architecture are managed within the corresponding `.yaml` files in the `./configs/` directory.

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

| **File**                | **Purpose**                                          | **Execution Command**         |
| ----------------------- | ---------------------------------------------------- | ----------------------------- |
| `MACE_DESSP.sh`         | Main DESSP pipeline (Distillation + Calibration)     | `bash MACE_DESSP.sh`          |
| `MACE_ORB_Distilled.sh` | Distilled model training (uMLIP knowledge transfer)  | `bash MACE_ORB_Distilled.sh`  |
| `MACE_VASPGO.sh`        | Baseline model training (DFT-GO dataset)             | `bash MACE_VASPGO.sh`         |
| `run_campaign.py`       | Batch structural search and production runs          | `python run_campaign.py`      |
| `run_ga.py`             | Individual Genetic Algorithm search engine           | (Called by `run_campaign.py`) |

By following these steps, you can reproduce the structural predictions and thermodynamic analysis reported in our work.
