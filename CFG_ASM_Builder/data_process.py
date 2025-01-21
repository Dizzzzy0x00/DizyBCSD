import os
import subprocess
import argparse

def generate_idb_and_asm(binary_folder, output_folder):
    """
    遍历给定文件夹中的所有二进制文件，生成 IDB 文件并分析反汇编代码。

    参数:
        binary_folder (str): 包含二进制文件的文件夹路径。
        output_folder (str): 输出文件夹路径。
        idb_utils_script (str): idb_utils.py 脚本路径。
        run_asm_builder_script (str): run_asm_builder.py 脚本路径。
        #ida_path (str): IDA Pro 可执行文件路径。
    """
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历二进制文件夹中的所有文件
    for filename in os.listdir(binary_folder):
        binary_path = os.path.join(binary_folder, filename)

        # 检查是否为文件（排除文件夹）
        if not os.path.isfile(binary_path):
            continue

        # 生成 IDB 文件
        idb_output_path = os.path.join(output_folder, f"{filename}.i64")
        print(f"[D] Generating IDB for {filename}...")
        cmd_generate_idb = [
            "python", "idb_utils.py",
            "-i", binary_path,
            "-o", idb_output_path,
            #"-i", ida_path
        ]
        subprocess.run(cmd_generate_idb, check=True)

        # 分析整个文件的反汇编代码
        asm_output_dir = os.path.join(output_folder, f"asm_output")
        print(f"[D] Generating ASM for {filename}...")
        cmd_generate_asm = [
            "python", "run_asm_buider.py",
            "-t", idb_output_path,
            "-o", asm_output_dir,
            "-f", "Full-file-analysis",
            #"-i", ida_path
        ]
        subprocess.run(cmd_generate_asm, check=True)

        print(f"[D] Completed processing {filename}.\n")

if __name__ == '__main__':
    # 使用 argparse 解析命令行参数
    parser = argparse.ArgumentParser(description="Generate IDB files and disassembly for binaries in a folder.")
    parser.add_argument('-d', '--binary-folder', required=True, help="Folder containing binary files.")
    parser.add_argument('-o', '--output-folder', required=True, help="Output folder for IDB and ASM files.")
    #parser.add_argument('-i', '--ida-path', default=r'F:\PWN\IDA Pro 7.6\idat64.exe', help='The path of IDA. Defaults to "F:\\PWN\\IDA Pro 7.6\\idat64.exe" if not provided.')
    args = parser.parse_args()

    # 调用主函数
    generate_idb_and_asm(
        binary_folder=args.binary_folder,
        output_folder=args.output_folder,
        #ida_path=args.ida_path
    )


    #python .\data_process.py -d .\testBinData\ -o .\testBinData\output
    #python .\ContrastiveData_gen.py -d .\testBinData\output\asm_output\ -o .\testBinData\cross_arch_pairs\output.csv