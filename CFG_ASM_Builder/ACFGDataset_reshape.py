import json
import os
import pandas as pd
from tqdm import tqdm  # 用于显示进度条
import argparse

def parse_filename(filename):
    parts = filename.split('_')
    arch_compiler_opt = parts[0]  # arm32-gcc-9-O1
    func_name = parts[-2]  # __do_global_dtors_aux
    arch_compiler, optimization = arch_compiler_opt.split('-', 1)

    return {
        'arch_compiler': arch_compiler,  # arm32
        'compiler': arch_compiler,  # gcc
        'optimization': optimization,   # gcc-9-O1
        'func_name': func_name
    }

# 提取 JSON 文件的关键信息
def extract_json_info(json_path):
    with open(json_path, 'r') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            print(f"Error decoding JSON file: {json_path}")
            return None
    
    content = list(data.values())[0]
    func_info = list(content.values())[1]

    # 检查是否包含必要的字段
    required_fields = ['fname', 'nodes', 'edges', 'basic_blocks']
    if not all(field in func_info for field in required_fields):
        print(f"Missing required fields in {json_path}")
        return None

    info = parse_filename(os.path.basename(json_path))
    info['file_path'] = json_path
    info['func_name'] = func_info['fname']
    info['num_nodes'] = len(func_info['nodes'])
    info['num_edges'] = len(func_info['edges'])

    # 记录节点的详细信息
    # 提取节点信息
    nodes = func_info['nodes']
    info['nodes'] = str(nodes)  # 将节点 ID 列表存储为字符串

    # 提取边信息
    edges = func_info['edges']
    info['edges'] = str(edges)  # 将边列表存储为字符串

    # 提取基本块的特征和汇编指令
    bb_features = []
    bb_disasm = []
    for bb_addr, bb_data in func_info['basic_blocks'].items():
        # 11维度的普通信息
        bb_feature = [
            bb_data['bb_size'],
            bb_data['n_numeric_consts'],
            bb_data['n_string_consts'],
            bb_data['n_instructions'],
            bb_data['n_arith_instrs'],
            bb_data['n_call_instrs'],
            bb_data['n_logic_instrs'],
            bb_data['n_transfer_instrs'],
            bb_data['n_redirect_instrs'],
            len(bb_data['bb_numerics']),
            len(bb_data['bb_strings'])
        ]
        bb_features.append(bb_feature)
        bb_disasm.append(bb_data['bb_disasm'])

    # 将列表转换为字符串存储
    info['bb_features'] = str(bb_features)
    info['bb_disasm'] = str(bb_disasm)

    return info



# 提取所有 JSON 文件信息并保存到 CSV
def reshape_data(json_dir, output_csv, batch_size=100):
    # 获取所有 JSON 文件
    json_files = [os.path.join(json_dir, f) for f in os.listdir(json_dir) if f.endswith('.json')]
    
    # 如果输出文件已存在，读取已处理的文件列表
    processed_files = set()
    if os.path.exists(output_csv):
        df_existing = pd.read_csv(output_csv)
        processed_files = set(df_existing['file_path'].tolist())
    
    # 过滤掉已处理的文件
    json_files = [f for f in json_files if f not in processed_files]
    
    # 分批次处理文件
    for i in tqdm(range(0, len(json_files), batch_size), desc="Processing batches"):
        batch_files = json_files[i:i + batch_size]
        data = []
        
        for json_file in batch_files:
            try:
                info = extract_json_info(json_file)
                if info:
                    data.append(info)
                # 删除已处理的文件
                os.remove(json_file)
            except Exception as e:
                print(f"Error processing {json_file}: {e}")
        
        # 将当前批次的数据追加到 CSV 文件
        df = pd.DataFrame(data)
        if os.path.exists(output_csv):
            df.to_csv(output_csv, mode='a', header=False, index=False)
        else:
            df.to_csv(output_csv, index=False)
        
        print(f"Processed {len(data)} records in this batch.")

    print(f"All records saved to {output_csv}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate triplet data for contrastive learning.")
    parser.add_argument('-d', '--acfg-folder', required=True, help="Folder containing asm.json files.")
    parser.add_argument('-o', '--output-csv', required=True, help="Output CSV file path.")
    parser.add_argument('-b', '--batch-size', type=int, default=100, help="Number of JSON files to process per batch.")
    args = parser.parse_args()

    # 配置路径
    acfg_output_folder = args.acfg_folder  # asm_output 文件夹路径
    output_csv = args.output_csv  # 输出 CSV 文件路径
    batch_size = args.batch_size  # 每批次处理的文件数量

    # 处理跨架构的对比样本并生成三元组数据
    reshape_data(acfg_output_folder, output_csv, batch_size)

#python .\ACFGDataset_reshape.py -d .\DataSet\clamav\output\acfg_output\ -o .\DataSet\clamav\acfg_data\output.csv  -b 200             