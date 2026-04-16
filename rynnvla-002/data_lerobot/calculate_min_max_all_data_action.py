#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
该脚本用于分析指定目录下的机器人 Action 数据。

它会递归地查找包含 'episode' 子目录的文件夹，然后在这些文件夹中
查找所有 'abs_action/*.npy' 文件。接着，它会使用多线程并行加载这些 .npy 文件，
验证其形状是否为 (6,)，最后计算并打印所有有效 Action 数据的统计信息
（最小值、最大值、1%分位数和99%分位数）。

用法:
    python analyze_actions.py /path/to/your/data/dir1 [/path/to/your/data/dir2 ...]

示例:
    # 分析单个根目录
    python analyze_actions.py /mnt/PLNAS/cenjun/all_data/extracted

    # 分析多个根目录
    python analyze_actions.py /mnt/data/set1 /mnt/data/set2

    # 分析一个非常具体的任务目录 (它本身包含 episode 子目录)
    python analyze_actions.py /mnt/PLNAS/cenjun/all_data/extracted/Place_the_block_inside_the_circle./yumingj/data/LeRobot_data_hdf5/hdf5/standard_desktop_data/data_0616/laptop_03/so100_Pick-up-all-the-blocks-one-by-one-with-tweezers-and-put-them-in-the-roll-2
"""

import os
import glob
import numpy as np
from tqdm import tqdm
import sys
import concurrent.futures
import argparse

# --- 可配置参数 ---
# 根据你的CPU核心数和I/O能力调整。对于网络文件系统(NAS/NFS)上的I/O密集型任务，
# 设置比CPU核心数更多的线程通常是有效的。可以从 16 或 32 开始尝试。
MAX_WORKERS = 32

def find_all_action_npy_files_fast(root_directories):
    """
    使用 glob 模式在给定的目录列表中快速查找所有 action .npy 文件。
    目录结构假定为 <root_directory>/<any_name>/abs_action/<chunk>/<step>.npy
    """
    npy_file_paths = []
    if not root_directories:
        return npy_file_paths

    print("开始使用 glob 模式快速扫描文件...")
    # 统计每个 action chunk 中的所有子动作，而不只是第一个 0.npy。
    glob_pattern = os.path.join('*', 'abs_action', '*', '*.npy')

    for root_dir in tqdm(root_directories, desc="扫描目录", unit="dir"):
        if not os.path.isdir(root_dir):
            print(f"警告: 目录 '{root_dir}' 不存在，已跳过。", file=sys.stderr)
            continue
        
        search_path = os.path.join(root_dir, glob_pattern)
        matched_files = glob.glob(search_path, recursive=False) # recursive=False is default, but explicit
        npy_file_paths.extend(matched_files)

    print(f"扫描完成，共找到 {len(npy_file_paths)} 个 .npy 文件。")
    return npy_file_paths

def load_and_validate_action(file_path):
    """
    由单个线程执行的工作函数：加载一个 .npy 文件并验证其形状。
    返回 Action 数据或 None（如果失败）。
    """
    try:
        action_data = np.load(file_path)
        if action_data.shape == (6,):
            return action_data
        else:
            # 将警告和错误打印到 stderr，以免干扰 tqdm 进度条
            print(f"\n警告: 文件 '{file_path}' 的形状为 {action_data.shape}，期望为 (6,)。已跳过。", file=sys.stderr)
            return None
    except Exception as e:
        print(f"\n错误: 无法加载或处理文件 '{file_path}'。错误信息: {e}", file=sys.stderr)
        return None

def analyze_action_data_multithreaded(file_paths):
    """
    使用多线程加载所有 .npy 文件，然后计算统计数据并打印结果。
    """
    if not file_paths:
        print("未找到任何 .npy 文件进行分析。")
        return

    all_actions = []
    print(f"正在使用最多 {MAX_WORKERS} 个线程并行加载和处理 Action 数据...")

    # 使用 ThreadPoolExecutor 来并行处理文件加载
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # executor.map 会将 file_paths 中的每个元素传给 load_and_validate_action 函数
        # 它会立即返回一个迭代器，结果在可用时产生
        # 我们用 tqdm 包裹这个迭代器来显示进度条
        results_iterator = executor.map(load_and_validate_action, file_paths)
        
        # 从迭代器中收集所有非 None 的结果
        all_actions = [result for result in tqdm(results_iterator, total=len(file_paths), desc="加载 .npy 文件", unit="file") if result is not None]

    if not all_actions:
        print("\n没有成功加载任何有效的数据。请检查文件内容和形状是否正确。")
        return

    # 将所有 action 列表转换为一个大的 (N, 6) NumPy 数组
    print(f"\n数据加载完成。正在堆叠数组...")
    stacked_actions = np.array(all_actions)
    
    print(f"成功加载并堆叠了 {stacked_actions.shape[0]} 个 action，形成数组，形状为: {stacked_actions.shape}")
    print("正在计算统计数据...")

    # 计算统计数据
    min_vals = np.min(stacked_actions, axis=0)
    max_vals = np.max(stacked_actions, axis=0)
    q01_vals = np.percentile(stacked_actions, 1, axis=0)
    q99_vals = np.percentile(stacked_actions, 99, axis=0)

    # 打印结果
    print("\n--- Action 数据统计结果 ---")
    print("-" * 85)
    print(f"{'维度':<10} | {'最小值':<20} | {'最大值':<20} | {'1% 分位数 (q01)':<20} | {'99% 分位数 (q99)':<20}")
    print("-" * 85)
    
    for i in range(6):
        print(f"维度 {i:<5} | {min_vals[i]:<20.8f} | {max_vals[i]:<20.8f} | {q01_vals[i]:<20.8f} | {q99_vals[i]:<20.8f}")
    
    print("-" * 85)

def find_episode_directories(root_dir):
    """
    在给定的根目录下查找所有“任务目录”。
    一个目录被认为是“任务目录”的条件是：它的所有直接子目录都以 'episode' 开头。
    这种结构通常表示该目录直接包含了所有 episode 数据。
    """
    path_list = []
    if not os.path.isdir(root_dir):
        print(f"错误: 提供的路径 '{root_dir}' 不是一个有效的目录。", file=sys.stderr)
        return path_list

    for dirpath, dirnames, filenames in os.walk(root_dir):
        # 忽略没有子目录的文件夹
        if not dirnames:
            continue
        
        # 判断当前目录下的所有子目录是否都以 'episode' 开头
        all_episodes = all(dirname.lower().startswith('episode') for dirname in dirnames)
        
        if all_episodes:
            path_list.append(dirpath)
            # 找到后，不再深入这个目录的子目录进行搜索，以避免重复和提高效率
            dirnames.clear()

    return path_list

def run_analysis_on_dirs(input_dirs):
    """
    对给定的输入目录列表执行完整的分析流程。
    """
    print("--- 开始查找任务目录 ---")
    all_episode_dirs = []
    for path in input_dirs:
        print(f"正在 '{path}' 中搜索...")
        found_dirs = find_episode_directories(path)
        if found_dirs:
            print(f"  -> 在 '{path}' 中找到 {len(found_dirs)} 个符合条件的任务目录。")
            all_episode_dirs.extend(found_dirs)
        else:
            print(f"  -> 在 '{path}' 中未找到符合条件的任务目录。")

    if not all_episode_dirs:
        print("\n在所有提供的路径中均未找到符合条件的任务目录。脚本退出。")
        print("提示：脚本会寻找这样一个目录，其所有子目录都以 'episode' 开头。")
        return
        
    print(f"\n总共找到 {len(all_episode_dirs)} 个任务目录用于后续分析。")
    
    # 从找到的任务目录中查找所有的 .npy 文件
    all_files = find_all_action_npy_files_fast(all_episode_dirs)
    
    # 对找到的所有文件进行多线程分析
    analyze_action_data_multithreaded(all_files)

def main():
    """
    主函数，用于解析命令行参数并启动分析流程。
    """
    parser = argparse.ArgumentParser(
        description="分析机器人 action .npy 数据文件并计算统计数据。",
        formatter_class=argparse.RawTextHelpFormatter, # 保持换行
        epilog=__doc__ # 使用文件顶部的文档字符串作为帮助结尾
    )
    parser.add_argument(
        'input_dirs', 
        nargs='+',  # 接受一个或多个参数
        metavar='INPUT_DIR',
        help='一个或多个要搜索的根目录路径。'
    )
    
    args = parser.parse_args()
    
    run_analysis_on_dirs(args.input_dirs)

# 脚本入口
if __name__ == "__main__":
    main()
