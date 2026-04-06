import os
import sys
import argparse
import math
import json
import time
from multiprocessing import Process
import torch

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


def run_script(rank, all_ranks, input_file, output_dir, resolution, tokenizer_path, log_dir):
    num_available_gpus = torch.cuda.device_count()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank % num_available_gpus)

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    command = (f"{sys.executable} -u pre_tokenize_action_state_local.py "
               f"--splits={all_ranks} "
               f"--rank={rank} "
               f"--in_filename {input_file} "
               f"--out_dir {output_dir} "
               f"--target_size {resolution} "
               f"--tokenizer {tokenizer_path} "
               f"> {log_dir}/worker_{rank}.log 2>&1")

    os.system(command)


def monitor_progress(all_ranks, output_dir, total_items):
    num_per_rank = math.ceil(total_items / all_ranks)

    bars = []
    for rank in range(all_ranks):
        start = num_per_rank * rank
        end = min(num_per_rank * (rank + 1), total_items)
        bar = tqdm(
            total=end - start,
            desc=f"Worker {rank:>2}",
            position=rank,
            leave=True,
            bar_format="{l_bar}{bar}| {n}/{total} [{elapsed}<{remaining}]",
        )
        bars.append((rank, bar, start, end))

    finished = [False] * all_ranks
    prev_progress = [None] * all_ranks

    while not all(finished):
        for rank, bar, start, end in bars:
            if finished[rank]:
                continue
            progress_file = os.path.join(output_dir, f"{rank}-of-{all_ranks}-progress.txt")
            try:
                with open(progress_file) as f:
                    val = f.read().strip()
                if val == "finished":
                    if not finished[rank]:
                        bar.n = end - start
                        bar.refresh()
                        finished[rank] = True
                else:
                    current = int(val)
                    if current != prev_progress[rank]:
                        bar.n = current - start + 1
                        bar.refresh()
                        prev_progress[rank] = current
            except (FileNotFoundError, ValueError):
                pass
        time.sleep(0.5)

    for _, bar, _, _ in bars:
        bar.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run parallel data processing script with direct file paths.")
    parser.add_argument('--input_file', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--resolution', type=int, required=True)
    parser.add_argument('--tokenizer_path', type=str, required=True)
    args = parser.parse_args()

    with open(args.input_file) as f:
        total_items = len(json.load(f))

    num_gpus = torch.cuda.device_count()
    all_ranks = num_gpus * 16  # ~1.15GB per worker, 16 workers = ~18GB on a 32GB card

    log_dir = os.path.join(args.output_dir, "logs")
    print(f"Starting {all_ranks} workers ({num_gpus} GPU(s)). Logs in {log_dir}/")

    processes = []
    for i in range(all_ranks):
        p = Process(target=run_script, args=(i, all_ranks, args.input_file, args.output_dir, args.resolution, args.tokenizer_path, log_dir))
        p.start()
        processes.append(p)

    if HAS_TQDM:
        monitor_progress(all_ranks, args.output_dir, total_items)
    else:
        for p in processes:
            p.join()

    for p in processes:
        p.join()

    print("\nAll processes have completed.")
    print(f"Logs in {log_dir}/")
