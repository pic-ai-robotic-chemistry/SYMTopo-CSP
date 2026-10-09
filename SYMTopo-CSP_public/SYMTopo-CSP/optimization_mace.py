"""Coordinate parallel MACE relaxation workers across available GPUs."""

import argparse
import os
import subprocess
import sys
import time


class CrystalOptimizer:
    """Base interface for crystal-optimization backends."""

    registry = {}

    def __init__(self, results_dir: str, **kwargs: object) -> None:
        self.target_dir = results_dir

    def __init_subclass__(
        cls, plugin_name: str | None = None, **kwargs: object
    ) -> None:
        super().__init_subclass__(**kwargs)

        if plugin_name is None:
            plugin_name = cls.__name__

        CrystalOptimizer.registry[plugin_name] = cls

    def start_optimize(self) -> None:
        raise NotImplementedError("Subclasses must implement this method")

    def wait_for_completion(self) -> None:
        raise NotImplementedError("Subclasses must implement this method")


class MultiScriptOptimizer(CrystalOptimizer):
    """Launch one MACE worker process per selected GPU."""

    def __init__(self, results_dir: str, **kwargs: object) -> None:
        super().__init__(results_dir, **kwargs)
        self.processes = []
        self.gpu_list = kwargs.get("gpu_list", [0])
        self.gpu_cnt = len(self.gpu_list)
        self.jobs_per_gpu = kwargs.get("jobs_per_gpu", 10)

    def start_optimize(self) -> None:
        if len(self.gpu_list) == 0:
            self.run_script_single()
        for i in self.gpu_list:
            process = self.run_script_single(i)
        self.processes.append(process)

    def run_script_single(self, gpu_id: int | str = "") -> subprocess.Popen:
        project_path = os.path.dirname(os.path.abspath(__file__))
        path_to_script = os.path.join(project_path, "opt_scripts", "mace_opt_slurm.py")
        command = [
            sys.executable,
            path_to_script,
            "--path",
            self.target_dir,
            "--n_jobs",
            str(self.jobs_per_gpu),
        ]
        if gpu_id == "":
            command.append("--use_cpu_only")
        my_env = os.environ.copy()
        my_env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        process = subprocess.Popen(command, env=my_env)
        return process

    def wait_for_completion(self) -> None:
        lock_folder = os.path.join(self.target_dir, "lock")
        pid_file = os.path.join(lock_folder, "pid.txt")
        start_time = time.time()
        near_end_time = time.time()
        max_wait = 600
        start_protect = 120
        threshold = self.gpu_cnt * self.jobs_per_gpu * 0.5
        while True:
            time.sleep(10)
            if os.path.exists(os.path.join(self.target_dir, "stop")):
                break
            if time.time() - start_time < start_protect:
                continue
            try:
                lock_count = len(os.listdir(lock_folder))
                if lock_count == 0 or (lock_count == 1 and os.path.exists(pid_file)):
                    break
            except FileNotFoundError:
                break
            if lock_count > threshold:
                near_end_time = time.time()
            if time.time() - near_end_time > max_wait:
                break

        if os.path.exists(pid_file):
            with open(pid_file, "r") as f:
                pid_list = [int(line.strip()) for line in f.readlines()]
            for pid in pid_list:
                try:
                    os.kill(pid, 9)
                except Exception:
                    continue
        time.sleep(5)
        for process in self.processes:
            if process.poll() is None:
                process.terminate()


if __name__ == "__main__":
    default_workers = os.cpu_count() or 1
    default_gpu_list = [
        i for i in os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")
    ]
    default_jobs_per_gpu = min(10, max(1, default_workers // len(default_gpu_list)))
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path", type=str, default="./example", help="Path to the target directory"
    )
    parser.add_argument(
        "--gpu_list",
        nargs="+",
        default=default_gpu_list,
        help="List of GPU IDs to use",
    )
    parser.add_argument(
        "--jobs_per_gpu",
        type=int,
        default=default_jobs_per_gpu,
        help="Number of jobs to run per GPU",
    )
    args = parser.parse_args()

    print(f"Starting optimization at {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    optimizer = MultiScriptOptimizer(
        results_dir=args.path,
        gpu_list=args.gpu_list,
        jobs_per_gpu=args.jobs_per_gpu,
    )
    optimizer.start_optimize()
    optimizer.wait_for_completion()
    print(f"Optimization completed at {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
