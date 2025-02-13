创建python env环境，安装依赖：

```bash

python -m venv env
pipenv shell
pip install -r requirements.txt
#for IDA3:
pip3 install networkx==2.5
#for IDA2:
pip2 install capstone==3.0.4

```

安装后运行testcase：

```bash
pipenv shell
python run_cfg_buider.py -t .\testcase\hello.exe -o .\testcase\output
```

使用示例：

```bash
#生成IDB文件
> python idb_utils.py -i .\testBinData\arm-32_binutils-2.34-O0_addr2line -o .\testBinData\arm-32_binutils-2.34-O0_addr2line.i64

#分析IDB文件：
> python .\run_cfg_buider.py -t .\testBinData\arm-32_binutils-2.34-O0_addr2line.i64 -o .\testBinData\output\ -f main  
hello!

#生成反汇编代码：
#特定函数：
> python .\run_asm_buider.py -t .\testBinData\arm-32_binutils-2.34-O0_addr2line.i64 -o .\testBinData\output2\ -f main

#整个文件：
> python .\run_asm_buider.py -t .\testBinData\x86-32_binutils-2.34-O0_addr2line.i64 -o .\testBinData\output\ -f Full-file-analysis
```

对指定文件夹内的所有文件进行反汇编分析，并生成函数基本asm文件对比样本对：

```bash
python .\data_process.py -d .\testBinData\ -o .\testBinData\output
python .\ContrastiveData_gen.py -d .\testBinData\output\asm_output\ -o .\testBinData\cross_arch_pairs\output.csv
```

对指定文件夹中的idb进行处理，生成acfg文件：

```
#生成json，-d指定idb所在文件夹，输出目录在idb的子目录acfg_output
python .\ACFGDataset_gen.py -d .\testBinData\output\

#reshape 将json提取成csv，-b是batchsize
python .\ACFGDataset_reshape.py -d .\testBinData\output\acfg_output\ -o .\testBinData\acfg_data\output.csv  -b 100 
```
