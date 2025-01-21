import os
import json
import random
from collections import defaultdict
from itertools import combinations
import re
import csv
import argparse

def parse_filename(filename):
    """
    解析文件名，架构, 编译器优化器, 源文件名, 函数名。
    
    参数:
        filename (str): 文件名，例如 "arm-32_binutils-2.34-O0_addr2line___aeabi_idivmod_asm.json"
    
    返回:
        tuple: (架构, 编译器优化器, 源文件名, 函数名)
    """
    pattern = r"^(.*?)-(.*?)_(.*?)_(.*?)_asm.json$"
    match = re.match(pattern, filename)
    if not match:
        return None
    return match.groups()

def find_cross_arch_pairs(asm_output_folder):
    """
    在 asm_output 文件夹中寻找跨架构的对比样本。
    
    参数:
        asm_output_folder (str): asm_output 文件夹路径。
    
    返回:
        dict: 键为 (编译器优化器, 源文件名, 函数名)，值为不同架构的文件路径列表。
    """
    # 用于存储相同函数的不同架构文件
    function_map = defaultdict(list)

    # 遍历 asm_output 文件夹中的所有文件
    for filename in os.listdir(asm_output_folder):
        #print(filename)
        # 解析文件名
        parsed = parse_filename(filename)
        if not parsed:
            continue

        arch, compiler, source_file, func_name = parsed
        key = (compiler, source_file, func_name)

        # 将文件路径添加到对应的键中
        file_path = os.path.join(asm_output_folder, filename)
        function_map[key].append((arch, file_path))

    # 筛选出跨架构的对比样本
    cross_arch_pairs = {k: v for k, v in function_map.items() if len(v) > 1}
    return cross_arch_pairs

def extract_json_content(file_path):
    """
    从 JSON 文件中提取内容。
    
    参数:
        file_path (str): JSON 文件路径。
    
    返回:
        dict: 包含 fname, arch, disassembly 等字段的字典。
    """
    with open(file_path, "r") as f:
        data = json.load(f)
        # 提取第一个键对应的值（假设 JSON 文件只有一个键）
        content = list(data.values())[0]
        return content

def save_triplets_to_csv(triplets, output_csv, mode='a'):
    """
    将三元组数据保存到 CSV 文件。
    
    参数:
        triplets (list): 三元组列表，格式为 [(anchor, positive, negative), ...]。
        output_csv (str): 输出 CSV 文件路径。
        mode (str): 文件写入模式，'a' 表示追加，'w' 表示覆盖。
    """
    with open(output_csv, mode=mode, newline='') as csv_file:
        fieldnames = [
            "anchor_fname", "anchor_arch", "anchor_disassembly",
            "positive_fname", "positive_arch", "positive_disassembly",
            "negative_fname", "negative_arch", "negative_disassembly"
        ]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if mode == 'w':  # 如果是覆盖模式，写入表头
            writer.writeheader()
        
        for anchor, positive, negative in triplets:
            writer.writerow({
                "anchor_fname": anchor["fname"],
                "anchor_arch": anchor["arch"],
                "anchor_disassembly": "\n".join(anchor["disassembly"]),
                "positive_fname": positive["fname"],
                "positive_arch": positive["arch"],
                "positive_disassembly": "\n".join(positive["disassembly"]),
                "negative_fname": negative["fname"],
                "negative_arch": negative["arch"],
                "negative_disassembly": "\n".join(negative["disassembly"]),
            })

def generate_triplets(function_map, output_csv, max_triplets):
    """
    生成三元组数据 (anchor, positive, negative) 并分批写入 CSV 文件。
    
    参数:
        function_map (dict): 函数映射，键为 (编译器优化器, 源文件名, 函数名)，值为不同架构的文件路径列表。
        output_csv (str): 输出 CSV 文件路径。
        max_triplets (int): 每次写入 CSV 文件的最大三元组数量。
    """
    triplets = []
    
    # 按编译器优化器和函数名分组
    compiler_func_map = defaultdict(list)
    for key, files in function_map.items():
        compiler, source_file, func_name = key
        compiler_func_map[(compiler, func_name)].extend(files)
    
    # 生成三元组
    for (compiler, func_name), files in compiler_func_map.items():
        if len(files) < 2:
            continue  # 至少需要两个样本才能生成正样本对
        
        # 生成正样本对 (anchor, positive)
        for (arch1, file1), (arch2, file2) in combinations(files, 2):
            anchor = extract_json_content(file1)
            positive = extract_json_content(file2)
            
            # 随机选择一个负样本
            negative = None
            while not negative:
                # 随机选择一个不同的函数
                random_compiler, random_func_name = random.choice(list(compiler_func_map.keys()))
                if random_func_name != func_name:  # 确保函数名不同
                    random_files = compiler_func_map[(random_compiler, random_func_name)]
                    if random_files:
                        _, random_file = random.choice(random_files)
                        negative = extract_json_content(random_file)
            
            # 添加三元组
            triplets.append((anchor, positive, negative))
            
            # 如果达到最大数量，写入 CSV 文件并清空列表
            if len(triplets) >= max_triplets:
                save_triplets_to_csv(triplets, output_csv, mode='a' if os.path.exists(output_csv) else 'w')
                print(f"[D] Saved {len(triplets)} triplets to {output_csv}")
                triplets = []  # 清空列表
    
    # 写入剩余的三元组
    if triplets:
        save_triplets_to_csv(triplets, output_csv, mode='a' if os.path.exists(output_csv) else 'w')
        print(f"[D] Saved {len(triplets)} triplets to {output_csv}")

def process_triplet_data(asm_output_folder, output_csv, max_triplets):
    """
    处理跨架构的对比样本并生成三元组数据。
    
    参数:
        asm_output_folder (str): asm_output 文件夹路径。
        output_csv (str): 输出 CSV 文件路径。
        max_triplets (int): 每次写入 CSV 文件的最大三元组数量。
    """
    # 确保输出文件的父目录存在
    output_folder = os.path.dirname(output_csv)
    if output_folder and not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # 寻找跨架构的对比样本
    function_map = find_cross_arch_pairs(asm_output_folder)
    
    # 生成三元组数据并分批写入 CSV 文件
    generate_triplets(function_map, output_csv, max_triplets)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate triplet data for contrastive learning.")
    parser.add_argument('-d', '--asm-folder', required=True, help="Folder containing asm.json files.")
    parser.add_argument('-o', '--output-csv', required=True, help="Output CSV file path.")
    parser.add_argument('-n', '--max-triplets', type=int, required=True, help="Maximum number of triplets per batch.")
    args = parser.parse_args()

    # 配置路径
    asm_output_folder = args.asm_folder  # asm_output 文件夹路径
    output_csv = args.output_csv  # 输出 CSV 文件路径
    max_triplets = args.max_triplets  # 每次写入 CSV 文件的最大三元组数量

    # 处理跨架构的对比样本并生成三元组数据
    process_triplet_data(asm_output_folder, output_csv, max_triplets)