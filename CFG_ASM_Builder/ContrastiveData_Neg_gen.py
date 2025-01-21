import os
import json
import csv
from collections import defaultdict
from itertools import combinations, product
import argparse
import re

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
        # 解析文件名
        parsed = parse_filename(filename)
        if not parsed:
            continue

        arch, compiler, source_file, func_name = parsed
        key = (compiler, source_file, func_name)

        # 将文件路径添加到对应的键中
        file_path = os.path.join(asm_output_folder, filename)
        function_map[key].append((arch, file_path))

    return function_map

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

def save_to_csv(pair_id, content1, content2, output_csv, label):
    """
    将对比样本保存到 CSV 文件中。
    
    参数:
        pair_id (int): 对比样本的 ID，例如 1。
        content1 (dict): 第一个架构的内容。
        content2 (dict): 第二个架构的内容。
        output_csv (str): 输出 CSV 文件路径。
        label (int): 样本标签，0 表示负样本，1 表示正样本。
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

def generate_negative_pairs(function_map, output_csv, max_pairs):
    """
    生成负样本并保存到 CSV 文件。
    
    参数:
        function_map (dict): 函数映射，键为 (编译器优化器, 源文件名, 函数名)，值为不同架构的文件路径列表。
        output_csv (str): 输出 CSV 文件路径。
        max_pairs (int): 最大负样本对数，默认为 100000。
    """
    # 按编译器优化器分组
    compiler_map = defaultdict(list)
    for key, files in function_map.items():
        compiler = key[0]  # 编译器优化器
        compiler_map[compiler].extend(files)

    # 生成负样本
    pair_id = 1
    for compiler, files in compiler_map.items():
        # 如果文件数量不足，跳过
        if len(files) < 2:
            continue

        # 生成不同架构或不同函数的负样本
        for (arch1, file1), (arch2, file2) in combinations(files, 2):
            content1 = extract_json_content(file1)
            content2 = extract_json_content(file2)

            # 如果架构不同或函数不同，则保存为负样本
            if arch1 != arch2 or content1["fname"] != content2["fname"]:
                save_to_csv(pair_id, content1, content2, output_csv, label=0)
                print(f"[D] Saved negative pair {pair_id:04d} to CSV ({arch1} vs {arch2})")

                pair_id += 1
                if pair_id > max_pairs:
                    return

def process_cross_arch_pairs(asm_output_folder, output_csv,num):
    """
    处理跨架构的对比样本并保存到 CSV 文件。
    
    参数:
        asm_output_folder (str): asm_output 文件夹路径。
        output_csv (str): 输出 CSV 文件路径。
        pairs_num:需要生成的负样本数据量
    """
    # 确保输出文件的父目录存在
    output_folder = os.path.dirname(output_csv)
    if output_folder and not os.path.exists(output_folder):  # 如果父目录存在且不为空
        os.makedirs(output_folder)

    # 寻找跨架构的对比样本
    function_map = find_cross_arch_pairs(asm_output_folder)

    # 生成负样本
    generate_negative_pairs(function_map, output_csv,num),

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate cross-architecture disassembly comparison pairs.")
    parser.add_argument('-d', '--asm-folder', required=True, help="Folder containing asm.json files.")
    parser.add_argument('-o', '--output-csv', required=True, help="Output CSV file path.")
    parser.add_argument('-n', '--pairs-num',type=int, required=True, help="Output data number")
    args = parser.parse_args()

    # 配置路径
    asm_output_folder = args.asm_folder  # asm_output 文件夹路径
    output_csv = args.output_csv  # 输出 CSV 文件路径
    pairs_num = args.pairs_num

    # 处理跨架构的对比样本
    process_cross_arch_pairs(asm_output_folder, output_csv,pairs_num)