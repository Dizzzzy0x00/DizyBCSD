import os
import json
import re
import csv
from collections import defaultdict
from itertools import combinations
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

def save_to_csv(pair_id, content1, content2, output_csv):
    """
    将对比样本保存到 CSV 文件中。
    
    参数:
        pair_id (int): 对比样本的 ID，例如 1。
        content1 (dict): 第一个架构的内容。
        content2 (dict): 第二个架构的内容。
        output_csv (str): 输出 CSV 文件路径。
    """
    # 构建 CSV 行数据
    row = {
        "pair_id": pair_id,
        "fname1": content1["fname"],
        "arch1": content1["arch"],
        "disassembly1": "\n".join(content1["disassembly"]),  # 将列表转换为字符串
        "preprocess_disassembly1":content1["preprocess_disassembly"],
        "fname2": content2["fname"],
        "arch2": content2["arch"],
        "disassembly2": "\n".join(content2["disassembly"]),  # 将列表转换为字符串
        "preprocess_disassembly2":content2["preprocess_disassembly"],
        "label": 1
    }

    # 写入 CSV 文件
    with open(output_csv, mode='a', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=row.keys())
        if csv_file.tell() == 0:  # 如果文件为空，写入表头
            writer.writeheader()
        writer.writerow(row)

def process_cross_arch_pairs(asm_output_folder, output_csv):
    """
    处理跨架构的对比样本并保存到 CSV 文件。
    
    参数:
        asm_output_folder (str): asm_output 文件夹路径。
        output_csv (str): 输出 CSV 文件路径。
    """
    # 确保输出文件的父目录存在
    output_folder = os.path.dirname(output_csv)
    if output_folder and not os.path.exists(output_folder):  # 如果父目录存在且不为空
        os.makedirs(output_folder)

    # 寻找跨架构的对比样本
    cross_arch_pairs = find_cross_arch_pairs(asm_output_folder)

    # 处理每个对比样本
    pair_id = 1
    for key, files in cross_arch_pairs.items():
        # 如果有多个架构，从中选取两个作为对比样本
        if len(files) >= 2:
            # 使用 combinations 生成所有可能的两个架构的组合
            for (arch1, file1), (arch2, file2) in combinations(files, 2):
                # 提取两个架构的内容
                content1 = extract_json_content(file1)
                content2 = extract_json_content(file2)

                # 保存对比样本到 CSV 文件
                save_to_csv(pair_id, content1, content2, output_csv)
                print(f"[D] Saved pair {pair_id:06d} to CSV for {key} ({arch1} vs {arch2})")

                pair_id += 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate cross-architecture disassembly comparison pairs.")
    parser.add_argument('-d', '--asm-folder', required=True, help="Folder containing asm.json files.")
    parser.add_argument('-o', '--output-csv', required=True, help="Output CSV file path.")
    args = parser.parse_args()

    # 配置路径
    asm_output_folder = args.asm_folder  # asm_output 文件夹路径
    output_csv = args.output_csv  # 输出 CSV 文件路径

    # 处理跨架构的对比样本
    process_cross_arch_pairs(asm_output_folder, output_csv)