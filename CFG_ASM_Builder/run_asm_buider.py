
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import click
import json
import subprocess
import time

import click

from os import getenv
from os.path import abspath
from os.path import dirname
from os.path import isfile
from os.path import join

LOG_PATH = ".\log\ASM_builder_log.txt"
IDA_PLUGIN = join(dirname(abspath(__file__)), 'IDA_ASM_builder.py')


# 创建命令行界面 (CLI)
@click.command()
@click.option('-t', '--target-file', required=True, help='The file to analyze.')
@click.option('-o', '--output-dir', required=True, help='Directory to store output.')
@click.option('-i', '--ida-path', default=r'F:\PWN\IDA Pro 7.6\idat64.exe', show_default=True, help='The path of IDA. Defaults to "F:\\PWN\\IDA Pro 7.6\\idat64.exe" if not provided.')
@click.option('-f', '--fname', required=True, help='Specify a function name to generate disassembly only for this function.')
def main(target_file, output_dir, ida_path, fname):
    try:
        # 检查 IDA 路径是否有效
        if not isfile(ida_path):
            print("[!] Error: IDA_PATH:{} not valid".format(ida_path))
            print("Use '-i' to set IDA path")
            return

        # 检查目标文件是否存在
        if not isfile(target_file):
            print("[!] Error: {} does not exist".format(target_file))
            return

        # 记录开始时间
        start_time = time.time()
        success_cnt, error_cnt = 0, 0

        # 构建 IDA 命令行参数
        cmd = [
            ida_path,
            '-A',  # 自动模式
            '-L{}'.format(LOG_PATH),  # 日志文件
            '-S{}'.format(IDA_PLUGIN),  # 插件脚本
            '-Oasm_builder:{};{};{}'.format(target_file, output_dir, fname),  # 插件选项
            target_file
        ]

        print("[D] cmd: {}".format(cmd))

        # 启动 IDA 进程
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = proc.communicate()

        # 检查进程返回值
        if proc.returncode == 0:
            print("[D] {}: success".format(target_file))
            success_cnt += 1
        else:
            print("[!] Error in {} (returncode={})".format(
                target_file, proc.returncode))
            error_cnt += 1

        # 记录结束时间
        end_time = time.time()
        print("[D] Elapsed time: {}".format(end_time - start_time))
        with open(LOG_PATH, "a+") as f_out:
            f_out.write("elapsed_time: {}\n".format(end_time - start_time))

        # 输出统计信息
        print("\n# IDBs correctly processed: {}".format(success_cnt))
        print("# IDBs error: {}".format(error_cnt))
    except Exception as e:
        print("[!] Something wrong!\n{}, see details in log file {}".format(e, LOG_PATH))

        
if __name__ == '__main__':
    print("hello!")
    main()
