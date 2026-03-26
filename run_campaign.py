#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_campaign_v2_mace.py

Compatible with: GA Script V8 (Constrained) + MACE
Purpose: multi-task scheduling (one GPU bound per task) + intra-task multiprocess parallel relaxation
Update: add runtime timer and a visual progress bar
"""

import os
import glob
import subprocess
import time
import sys
import shutil
from pathlib import Path

# ===========================
#  Core configuration
# ===========================

# Notes:
# 1. The paths below may be specified as either absolute or relative paths.
# 2. If a relative path is used, it is resolved relative to the directory containing
#    the current scheduler script run_campaign_v2_mace.py, rather than relative to
#    the case_dir after subsequent subprocess directory switching.
SCRIPT_DIR = Path(__file__).resolve().parent


def resolve_path(path_str, base_dir=SCRIPT_DIR):
    """
    Resolve all user-specified paths in the core configuration section to absolute paths.

    Parameters
    ----------
    path_str : str
        User-configured path, which may be either absolute or relative.
    base_dir : str or Path
        Base directory used when path_str is a relative path.
        By default, the directory containing the current scheduler script, SCRIPT_DIR, is used.

    Returns
    -------
    str
        The resolved absolute path as a string.
    """
    if path_str is None:
        return None

    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = Path(base_dir) / path
    return str(path.resolve())


# === Base directories ===
STRUCTURES_DIR = resolve_path("./structures")
WORK_DIR = resolve_path("BFGS-fmax0.01_200")

# === 你的 GA V8 脚本路径 ===
GA_SCRIPT = resolve_path("./ga4LiC/run_ga.py")

# 初始化脚本
INIT_SCRIPT = resolve_path("./ga4LiC/initialize.py")

# Li-sites.txt is assumed to reside in the same directory as the GA script
LI_SITES_FILE = resolve_path("Li-sites.txt", base_dir=Path(GA_SCRIPT).resolve().parent)

# === Python interpreter ===
# An absolute path is still recommended for the Python interpreter; if a relative path is used, it will also be resolved relative to SCRIPT_DIR.
PYTHON_EXEC = resolve_path("/home/user/anaconda3/envs/mace/bin/python")

# === MACE 模型路径 ===
MACE_MODEL_PATH = resolve_path("./mace-train/models/MACE_DESSP.model")

# === Parallel scheduling configuration ===
MAX_CAMPAIGN_TASKS = 4
NUM_GPUS = 4
GA_INTERNAL_JOBS = 12
REQUIRED_VRAM_MB = 12000

# ===========================
# Color codes
# ===========================
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

# ===========================
# Utility functions
# ===========================

def is_pid_running(pid):
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True

def get_task_status(case_dir):

    pid_file = os.path.join(case_dir, "ga_run.pid")

    log_files = [
        os.path.join(case_dir, "ga_stdout.log"),
        os.path.join(case_dir, "ga_outputs", "run.log"),
    ]

    is_finished = False
    for log_file in log_files:
        if os.path.exists(log_file):
            try:
                with open(log_file, 'r') as f:
                    lines = f.readlines()[-80:]
                    for line in lines:
                        if "GA Finished" in line or "Best structure found" in line:
                            is_finished = True
                            break
            except:
                pass
        if is_finished:
            break

    if is_finished:
        return 'completed', None

    if os.path.exists(pid_file):
        try:
            with open(pid_file, 'r') as f:
                content = f.read().strip()
                if not content:
                    return 'failed', None
                pid = int(content)

            if is_pid_running(pid):
                return 'running', pid
            else:
                return 'failed', None
        except:
            return 'failed', None

    return 'new', None

def get_gpu_status():
    """Return the list of free GPU memory values (MB)."""
    try:
        result = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.used,memory.total,memory.free', '--format=csv,nounits,noheader'],
            encoding='utf-8'
        )
        lines = result.strip().split('\n')
        free_list = []
        for line in lines:
            if not line.strip():
                continue
            u, t, f = line.split(',')
            free_list.append(int(f))
        return free_list
    except:
        return [99999] * NUM_GPUS

def format_time(seconds):
    """Convert seconds to HH:MM:SS format."""
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

def print_status(total, running, completed, start_time):

    # 1. Compute elapsed time
    elapsed = time.time() - start_time
    time_str = format_time(elapsed)
    
    # 2. Compute progress bar state
    bar_len = 20
    if total > 0:
        percent = completed / total
        filled_len = int(bar_len * percent)
    else:
        percent = 0
        filled_len = 0
    
    # Render the progress bar using Unicode block characters
    bar = '█' * filled_len + '░' * (bar_len - filled_len)
    
    sys.stdout.write("\r\033[K") # Clear the current terminal line
    msg = (
        f"{Colors.BOLD}[{time_str}]{Colors.ENDC} "  # [00:01:30]
        f"{Colors.BLUE}|{bar}|{Colors.ENDC} "      # |████░░░░|
        f"{percent*100:5.1f}% "                      # 45.0%
        f"Done:{Colors.GREEN}{completed}{Colors.ENDC} | "
        f"Run:{Colors.BLUE}{len(running)}{Colors.ENDC} | "
        f"Wait:{Colors.YELLOW}{total - completed - len(running)}{Colors.ENDC}"
    )
    sys.stdout.write(msg)
    sys.stdout.flush()

# ===========================
# Main program
# ===========================

def main():
    # Path validation
    if not os.path.exists(PYTHON_EXEC):
        print(f"{Colors.RED}Error: Python executable not found: {PYTHON_EXEC}{Colors.ENDC}")
        return
    if not os.path.exists(STRUCTURES_DIR):
        print(f"{Colors.RED}Error: Structures dir not found: {STRUCTURES_DIR}{Colors.ENDC}")
        return
    if not os.path.exists(GA_SCRIPT):
        print(f"{Colors.RED}Error: GA Script not found: {GA_SCRIPT}{Colors.ENDC}")
        return
    if not os.path.exists(INIT_SCRIPT):
        print(f"{Colors.RED}Error: Init Script not found: {INIT_SCRIPT}{Colors.ENDC}")
        return
    if not os.path.exists(MACE_MODEL_PATH):
        print(f"{Colors.RED}Error: MACE model not found: {MACE_MODEL_PATH}{Colors.ENDC}")
        return

    base_files = sorted(glob.glob(os.path.join(STRUCTURES_DIR, "*.vasp")))
    if not base_files:
        print(f"{Colors.RED}No .vasp files found in {STRUCTURES_DIR}{Colors.ENDC}")
        return

    print(f"{Colors.HEADER}=== GA Scheduler (MACE + V8 Constrained) ==={Colors.ENDC}")
    print(f"Script Dir: {SCRIPT_DIR}")
    print(f"Base Structures: {STRUCTURES_DIR}")
    print(f"Work Dir: {WORK_DIR}")
    print(f"GA Script: {GA_SCRIPT}")
    print(f"Init Script: {INIT_SCRIPT}")
    print(f"Python Exec: {PYTHON_EXEC}")
    print(f"Li-sites File: {LI_SITES_FILE}")
    print(f"MACE Model: {MACE_MODEL_PATH}")
    print(f"Internal relax_jobs per GA: {GA_INTERNAL_JOBS}")

    # 1. Build the task queue
    queue = []
    task_counter = 0

    for base_f in base_files:
        struct_name = os.path.basename(base_f).split('.')[0]

        for n_li in [1, 2]:
            task_counter += 1
            case_name = f"{struct_name}_Li{n_li}"
            case_dir = os.path.join(WORK_DIR, case_name)
            db_file = os.path.join(case_dir, "ga_test.db")

            init_cmd = [
                PYTHON_EXEC, INIT_SCRIPT,
                "--structure", base_f,
                "--db", "ga_test.db",
                "--n_li", str(n_li),
                "--pop_size", "20"
            ]

            ga_cmd = [
                PYTHON_EXEC, GA_SCRIPT,
                "--db_file", "ga_test.db",
                "--li_sites", "Li-sites.txt",
                "--num_gen", "30",
                "--calc", "mace",
                "--model_path", MACE_MODEL_PATH,
                "--search_steps", "200",
                "--search_fmax", "0.01",
                "--strict_sim", "0.9998",
                "--loose_sim", "0.9995",
                "--energy_diff_threshold", "0.0001",
                "--skip_final",
                "--early_stop",
                "--min_gens", "8",
                "--min_delta_E", "0.001",
                "--patience_E", "10",
                "--min_accept_ratio", "0.01",
                "--patience_novelty", "6",
                "--early_stop_mode", "AND",
                "--optimizer", "BFGS",
                "--pop_size", "10",
                "--constrain_prob", "0.0",
                "--relax_jobs", str(GA_INTERNAL_JOBS)
            ]

            queue.append({
                "name": case_name,
                "dir": case_dir,
                "init_cmd": init_cmd,
                "ga_cmd": ga_cmd,
                "db_path": db_file,
                "pid_file": os.path.join(case_dir, "ga_run.pid")
            })

    # 2. Scan existing tasks and resume if possible
    running_tasks = []
    completed_count = 0
    total_tasks = len(queue) # Store the total number of tasks for the progress bar

    for task in queue[:]:
        status, pid = get_task_status(task['dir'])
        if status == 'completed':
            completed_count += 1
            queue.remove(task)
        elif status == 'running':
            print(f"Resuming {task['name']} (PID {pid})")
            running_tasks.append({'meta': task, 'pid': pid, 'gpu': -1})
            queue.remove(task)

    print(f"Tasks: {len(queue)} pending, {len(running_tasks)} running, {completed_count} done.")
    time.sleep(1)

    gpu_cursor = 0
    
    # === Record the scheduler start time ===
    start_time = time.time()

    # 3. Scheduling loop
    try:
        while queue or running_tasks:
            # A. Inspect currently running tasks
            active = []
            for rt in running_tasks:
                case_dir = rt['meta']['dir']
                pid = rt['pid']

                status, _ = get_task_status(case_dir)

                if status == 'completed':
                    sys.stdout.write("\n")
                    print(f"{Colors.GREEN} {rt['meta']['name']} Finished (detected via log).{Colors.ENDC}")
                    completed_count += 1

                    if is_pid_running(pid):
                        try:
                            os.kill(pid, 9)
                        except:
                            pass

                elif is_pid_running(pid):
                    active.append(rt)
                else:
                    sys.stdout.write("\n")
                    print(f"{Colors.RED} {rt['meta']['name']} Failed (Process died). Retrying...{Colors.ENDC}")
                    queue.append(rt['meta'])

            running_tasks = active

            # B. Submit new tasks
            while len(running_tasks) < MAX_CAMPAIGN_TASKS and queue:
                free_mems = get_gpu_status()
                target_gpu = -1

                for _ in range(NUM_GPUS):
                    idx = gpu_cursor % NUM_GPUS
                    gpu_cursor += 1
                    if free_mems[idx] > REQUIRED_VRAM_MB:
                        target_gpu = idx
                        break

                if target_gpu == -1:
                    break

                task = queue.pop(0)
                os.makedirs(task['dir'], exist_ok=True)

                sys.stdout.write("\n")
                print(f" Launching {task['name']} on GPU {target_gpu} (free_mem={free_mems[target_gpu]} MB)")

                if os.path.exists(LI_SITES_FILE):
                    shutil.copy(LI_SITES_FILE, os.path.join(task['dir'], "Li-sites.txt"))
                
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = str(target_gpu)
                env["OMP_NUM_THREADS"] = "1"
                env["MKL_NUM_THREADS"] = "1"
                env["TF_CPP_MIN_LOG_LEVEL"] = "3"
                env["PYTHONWARNINGS"] = "ignore"

                if not os.path.exists(task['db_path']) or os.path.getsize(task['db_path']) == 0:
                    init_stdout_path = os.path.join(task['dir'], "init_stdout.log")
                    init_stderr_path = os.path.join(task['dir'], "init_stderr.log")

                    with open(init_stdout_path, "w") as init_out, open(init_stderr_path, "w") as init_err:
                        try:
                            subprocess.run(
                                task['init_cmd'],
                                cwd=task['dir'],
                                env=env,
                                check=True,
                                stdout=init_out,
                                stderr=init_err
                            )
                        except subprocess.CalledProcessError as e:
                            print(f"{Colors.RED}Init failed for {task['name']} (returncode={e.returncode}){Colors.ENDC}")
                            print(f"{Colors.YELLOW}See logs: {init_stdout_path} and {init_stderr_path}{Colors.ENDC}")
                            continue
                        except FileNotFoundError as e:
                            print(f"{Colors.RED}Init failed for {task['name']}: {e}{Colors.ENDC}")
                            print(f"{Colors.YELLOW}Check whether the configured path is correct: {task['init_cmd'][0]} / {task['init_cmd'][1]}{Colors.ENDC}")
                            continue

                out_log = open(os.path.join(task['dir'], "ga_stdout.log"), "w")
                err_log = open(os.path.join(task['dir'], "ga_stderr.log"), "w")

                p = subprocess.Popen(
                    task['ga_cmd'],
                    cwd=task['dir'],
                    env=env,
                    stdout=out_log,
                    stderr=err_log,
                    start_new_session=True
                )

                with open(task['pid_file'], "w") as f:
                    f.write(str(p.pid))

                running_tasks.append({'meta': task, 'pid': p.pid, 'gpu': target_gpu})

            # === Update the status line (passing start_time) ===
            print_status(total_tasks, running_tasks, completed_count, start_time)
            time.sleep(10)

    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Scheduler paused. Running tasks continue in background.{Colors.ENDC}")

if __name__ == "__main__":
    main()