import os
import subprocess
import argparse

def generate_acfg(idb_folder):
    """
    遍历给定文件夹中的所有二进制文件，生成 IDB 文件并分析反汇编代码。

    参数:
        idb_folder (str): 输出文件夹路径。
        idb_utils_script (str): idb_utils.py 脚本路径。
        run_acfg_builder_script (str): run_asm_builder.py 脚本路径。
        #ida_path (str): IDA Pro 可执行文件路径。
    """
    # 确保输出文件夹存在
    acfg_output_dir = os.path.join(idb_folder, f"acfg_output")
    if not os.path.exists(acfg_output_dir):
        os.makedirs(acfg_output_dir)

    # 遍历二进制文件夹中的所有文件
    for filename in os.listdir(idb_folder):
        idb_path = os.path.join(idb_folder, filename)

        # 检查是否为文件（排除文件夹）
        if not os.path.isfile(idb_path):
            continue
        if not filename.endswith('.i64'):
            continue

        cmd_generate_acfg = [
            "python", "run_cfg_buider.py",
            "-t", idb_path,
            "-o", acfg_output_dir,
            "-f", "Full-file-analysis",
            #"-i", ida_path
        ]
        subprocess.run(cmd_generate_acfg, check=True)

        print(f"[D] Completed processing {filename}.\n")

if __name__ == '__main__':
    # 使用 argparse 解析命令行参数
    parser = argparse.ArgumentParser(description="Generate IDB files and disassembly for binaries in a folder.")
    #parser.add_argument('-d', '--binary-folder', required=True, help="Folder containing binary files.")
    parser.add_argument('-d', '--idb-folder', required=True, help="Output folder for IDB and ASM files.")
    #parser.add_argument('-i', '--ida-path', default=r'F:\PWN\IDA Pro 7.6\idat64.exe', help='The path of IDA. Defaults to "F:\\PWN\\IDA Pro 7.6\\idat64.exe" if not provided.')
    args = parser.parse_args()

    # 调用主函数
    generate_acfg(
        #binary_folder=args.binary_folder,
        idb_folder=args.idb_folder,
        #ida_path=args.ida_path
    )


    #python .\ACFGDataset_gen.py -d .\testBinData\output 
    #python .\ContrastiveData_gen.py -d .\testBinData\output\asm_output\ -o .\testBinData\cross_arch_pairs\output.csv