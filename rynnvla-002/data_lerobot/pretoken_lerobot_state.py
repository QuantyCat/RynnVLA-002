import os
import sys
import argparse  # 导入 argparse 模块
from multiprocessing import Process
import torch

# 1. 更新函数签名，直接接收输入文件和输出目录路径
def run_script(rank, all_ranks, input_file, output_dir, resolution, tokenizer_path):
    """
    为单个进程执行预分词脚本。
    """
    num_available_gpus = torch.cuda.device_count()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank % num_available_gpus)
    print(f"Starting running on rank {rank} for file {input_file}.")

    # 2. 确保输出目录存在
    #    pre_tokenize_action_local.py 脚本可能不会自动创建目录，
    #    在这里加上可以保证进程启动时目录已存在。
    os.makedirs(output_dir, exist_ok=True)

    # 3. 更新 os.system 命令，直接使用传入的完整路径
    command = (f"{sys.executable} -u pre_tokenize_action_state_local.py "
               f"--splits={all_ranks} "
               f"--rank={rank} "
               f"--in_filename {input_file} "
               f"--out_dir {output_dir} "
               f"--target_size {resolution} "
               f"--tokenizer {tokenizer_path}")
    
    print(f"Rank {rank} executing command: {command}")
    os.system(command)


if __name__ == "__main__":
    # 1. 创建 ArgumentParser 对象，并更新描述
    parser = argparse.ArgumentParser(description="Run parallel data processing script with direct file paths.")

    # 2. 修改命令行参数，直接接收文件和目录路径
    parser.add_argument('--input_file', type=str, required=True,
                        help="Full path to the input JSON file (e.g., /path/to/data/train.json).")
    parser.add_argument('--output_dir', type=str, required=True,
                        help="Full path to the output directory (e.g., /path/to/output/tokens/train).")
    parser.add_argument('--resolution', type=int, required=True,
                        help="Target resolution for processing (e.g., 256).")
    parser.add_argument('--tokenizer_path', type=str, required=True, 
                        help="tokenizer path for ../ckpts/models--Alpha-VLLM--Lumina-mGPT-7B-768/snapshots/9624463a82ea5ce814af9b561dcd08a31082c3af")

    # 3. 解析命令行参数
    args = parser.parse_args()

    # 4. 移除 data_type 循环，直接使用命令行参数
    processes = []
    all_ranks = 32  # 总进程数
    for i in range(all_ranks):
        # 5. 将解析到的新参数传递给 run_script 函数
        p = Process(target=run_script, args=(i, all_ranks, args.input_file, args.output_dir, args.resolution, args.tokenizer_path))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print("All processes have completed.")

