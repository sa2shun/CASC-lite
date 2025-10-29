#!/usr/bin/env python
"""Run fixed-n CASC-lite jobs in parallel shards, batching generate calls for large n."""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, List

DEFAULT_CMD = [sys.executable, "-m", "casc_lite.cli.run_once"]


def detect_gpus() -> List[str]:
    cmd = ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"]
    try:
        output = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True)
        gpus = [line.strip() for line in output.splitlines() if line.strip()]
        return gpus
    except (subprocess.CalledProcessError, FileNotFoundError):
        return []


def chunk(iterable: List[str], n_chunks: int) -> List[List[str]]:
    if n_chunks <= 1 or len(iterable) == 0:
        return [iterable]
    chunk_size = math.ceil(len(iterable) / n_chunks)
    return [iterable[i : i + chunk_size] for i in range(0, len(iterable), chunk_size)]


def run_worker(args: List[str], env: dict[str, str]) -> subprocess.CompletedProcess:
    return subprocess.run(args, env=env, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)


def main() -> None:
    parser = argparse.ArgumentParser(description="Parallel fixed-n runner with dataset sharding")
    parser.add_argument("--config", default="src/casc_lite/config/default.yaml")
    parser.add_argument("--data", default="data/gsm8k_full.jsonl")
    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--n_fixed", type=int, default=50)
    parser.add_argument("--K", type=int, default=None)
    parser.add_argument("--num_shards", type=int, default=None, help="Number of dataset shards (defaults to GPU count)")
    parser.add_argument("--gpus", default="auto", help="Comma-separated GPU indices or 'auto' to detect")
    parser.add_argument("--extra_args", default="", help="Additional CLI arguments appended to run_once")
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    with data_path.open("r", encoding="utf-8") as fp:
        lines = [line for line in fp.readlines() if line.strip()]

    if not lines:
        raise RuntimeError("Dataset contains no entries")

    if args.gpus.lower() == "auto":
        gpu_indices = detect_gpus()
    else:
        gpu_indices = [idx.strip() for idx in args.gpus.split(",") if idx.strip()]

    if not gpu_indices:
        print("[WARN] No GPUs detected. Falling back to CPU execution.")
        gpu_indices = [""]

    num_shards = args.num_shards or len(gpu_indices)
    shards = chunk(lines, num_shards)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    temp_root = Path(tempfile.mkdtemp(prefix="casc_shards_", dir=str(output_dir)))

    shard_files: List[Path] = []
    for i, shard_lines in enumerate(shards):
        shard_file = temp_root / f"shard_{i:03d}.jsonl"
        with shard_file.open("w", encoding="utf-8") as fp:
            fp.writelines(shard_lines)
        shard_files.append(shard_file)

    print(f"==> Prepared {len(shard_files)} shard files in {temp_root}")

    futures = []
    with ThreadPoolExecutor(max_workers=len(gpu_indices)) as executor:
        for idx, shard_file in enumerate(shard_files):
            gpu = gpu_indices[idx % len(gpu_indices)]
            env = os.environ.copy()
            repo_root = Path(__file__).resolve().parent.parent
            additional_path = str(repo_root / "src")
            existing_pp = env.get("PYTHONPATH", "")
            if existing_pp:
                env["PYTHONPATH"] = os.pathsep.join([additional_path, existing_pp])
            else:
                env["PYTHONPATH"] = additional_path
            if gpu:
                env["CUDA_VISIBLE_DEVICES"] = gpu
            else:
                env.pop("CUDA_VISIBLE_DEVICES", None)

            cmd = DEFAULT_CMD + [
                "--config",
                str(args.config),
                "--data",
                str(shard_file),
                "--mode",
                "fixed",
                "--n_fixed",
                str(args.n_fixed),
                "--output_dir",
                str(output_dir),
            ]
            if args.K is not None:
                cmd += ["--K", str(args.K)]
            if args.extra_args:
                cmd += args.extra_args.split()

            futures.append(executor.submit(run_worker, cmd, env))

        for fut in as_completed(futures):
            try:
                result = fut.result()
                if result.stdout:
                    print(result.stdout)
            except subprocess.CalledProcessError as exc:
                print(exc.stdout, file=sys.stderr)
                raise

    shutil.rmtree(temp_root)
    print("==> All shards complete.")


if __name__ == "__main__":
    main()
