
import click
import subprocess

from os import getenv
from os import makedirs
from os import mkdir
from os import walk
from os.path import abspath
from os.path import dirname
from os.path import isdir
from os.path import isfile
from os.path import join
from os.path import relpath
from os.path import samefile
LOG_PATH = ".\log\idb_utils_log.txt"

def export_idb(input_path, output_path,IDA_PATH):
    """Launch IDA Pro and export the IDB. Inner function."""
    try:
        print("Export IDB for {}".format(input_path))
        ida_output = str(subprocess.check_output([
            IDA_PATH,
            "-L{}".format(LOG_PATH),  # name of the log file. "Append mode"
            "-a-",  # enables auto analysis
            "-B",  # batch mode. IDA will generate .IDB and .ASM files
            "-o{}".format(output_path),
            input_path
        ]))

        if not isfile(output_path):
            print("[!] Error: file {} not found".format(output_path))
            print(ida_output)
            return False

        return True

    except Exception as e:
        print("[!] Exception in export_idb\n{}".format(e))


#创建命令行界面 (CLI) 
@click.command()
@click.option('-i', '--input-file', required=True, help='the path of input binary file')
@click.option('-o', '--output-file', required=True, help='the path of output idb.')
@click.option('-ida', '--ida-path', default=r'F:\PWN\IDA Pro 7.6\idat64.exe', show_default=True, help='The path of IDA. Defaults to "F:\\PWN\\IDA Pro 7.6\\idat64.exe" if not provided.')
def main(input_file, output_file, ida_path):
    if not isfile(ida_path):
        print("[!] Error: IDA_PATH:{} not valid".format(ida_path))
        print("Use '-ida .\idapath'")
        return
    if not isfile(input_file):
        print("[!] Error: INPUT_PATH:{} not valid".format(input_file))
        print("Use '-i .\input.exe'")
        return
    export_idb(input_file, output_file, ida_path)

if __name__ == "__main__":
    main()