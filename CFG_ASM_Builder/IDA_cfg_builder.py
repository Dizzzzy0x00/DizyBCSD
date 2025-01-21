import base64
import idaapi
import idc
import idautils
import json
import ntpath
import os
import time

from capstone import *
from collections import namedtuple

from bb_analyse_utils import *

BasicBlock = namedtuple('BasicBlock', ['va', 'size', 'succs'])
#va (Virtual Address),size,succs (Successors)

Function = namedtuple('Function', ['fname', 'addr'])
#fname(function name),addr(function address)

def get_bitness():
    #获取程序的位信息
    info = idaapi.get_inf_structure()
    if info.is_64bit():
        return 64
    elif info.is_32bit():
        return 32

def convert_procname_to_str(arch, bitness):
    #格式化json输出
    if arch == 'mipsb':
        return "mips_{}".format(bitness)
    if arch == "arm":
        return "arm_{}".format(bitness)
    if "pc" in arch:
        return "x86_{}".format(bitness)
    raise RuntimeError(
        "[!] Arch not supported ({}, {})".format(
            arch, bitness))

def get_func_bb(func_addr):
    #给定函数地址，返回这个函数所有的基本块BasicBlock
    bb_list = list() #初始化
    
    func = idaapi.get_func(func_addr)
    if func is None:
        return bb_list
    for bb in idaapi.FlowChart(func,flags=idaapi.FC_PREDS):
        print("正在获取函数基本块，起始地址{}\{}，块大小：{}".format(bb.start_ea,bb.end_ea,bb.end_ea - bb.start_ea))
        bb_list.append(
            BasicBlock(
                va=bb.start_ea,
                size=bb.end_ea - bb.start_ea,
                succs=[x.start_ea for x in bb.succs()]))
    return bb_list

def get_capstone(procname,bitness):

    #https://github.com/williballenthin/python-idb

    md = None
    arch = ""

    # WARNING: mipsl mode not supported here
    if procname == 'mipsb':
        arch = "MIPS"
        if bitness == 32:
            md = Cs(CS_ARCH_MIPS, CS_MODE_MIPS32 | CS_MODE_BIG_ENDIAN)
        if bitness == 64:
            md = Cs(CS_ARCH_MIPS, CS_MODE_MIPS64 | CS_MODE_BIG_ENDIAN)

    if procname == "arm":
        arch = "ARM"
        if bitness == 32:
            # WARNING: THUMB mode not supported here
            md = Cs(CS_ARCH_ARM, CS_MODE_ARM)
        if bitness == 64:
            md = Cs(CS_ARCH_ARM64, CS_MODE_ARM)

    if "pc" in procname:
        arch = "x86"
        if bitness == 32:
            md = Cs(CS_ARCH_X86, CS_MODE_32)
        if bitness == 64:
            md = Cs(CS_ARCH_X86, CS_MODE_64)

    if md is None:
        raise RuntimeError(
            "Capstone initialization failure ({}, {})".format(
                procname, bitness))

    # Set detail to True to get the operand detailed info
    md.detail = True
    return md, arch


def capstone_disassembly(md, ea, size):
    try:
        bb_mnems, bb_disasm, bb_norm ,bb_numerics= list(), list(), list(), list()
        # 遍历每一条指令
        for inst in md.disasm(idc.get_bytes(ea, size), ea):
            bb_mnems.append(inst.mnemonic) #指令
            bb_disasm.append("{} {}".format(
                inst.mnemonic,
                inst.op_str))

            # 对指令进行规范化
            norm_inst = inst.mnemonic
            #遍历指令的操作数
            for op in inst.operands:
                #寄存器
                if (op.type == 1):
                    norm_inst = norm_inst + " " + inst.reg_name(op.reg)
                #立即数
                elif (op.type == 2):
                    imm = int(op.imm)
                    bb_numerics.append(imm)
                    if (-int(5000) <= imm <= int(5000)):
                        norm_inst += " " + str(hex(op.imm))
                    else:
                        norm_inst += " " + str('HIMM')
                #内存地址
                elif (op.type == 3):
                    # If the base register is zero, convert to "MEM"
                    if (op.mem.base == 0):
                        norm_inst += " " + str("[MEM]")
                    else:
                        # Scale not available, e.g. for ARM
                        if not hasattr(op.mem, 'scale'):
                            norm_inst += " " + "[{}+{}]".format(
                                str(inst.reg_name(op.mem.base)),
                                str(op.mem.disp))
                        else:
                            norm_inst += " " + "[{}*{}+{}]".format(
                                str(inst.reg_name(op.mem.base)),
                                str(op.mem.scale),
                                str(op.mem.disp))

                if (len(inst.operands) > 1):
                    norm_inst += ","

            norm_inst = norm_inst.replace("*1+", "+")
            norm_inst = norm_inst.replace("+-", "-")

            if "," in norm_inst:
                norm_inst = norm_inst[:-1]
            norm_inst = norm_inst.replace(" ", "_").lower()
            bb_norm.append(str(norm_inst))

        return bb_mnems, bb_disasm, bb_norm ,bb_numerics
    
    except Exception as e:
        print("[!] Capstone exception", e)
        return list(), list(), list(), list()


def get_bb_feature(md,bb,arch):
    features_dict = dict()
    # 处理空块：
    if bb.size == 0:
        features_dict = dict(
            bb_size=0,
            #bb_mnems=list(),
            bb_disasm=list(),
            #bb_norm=list(),
            # BB list-type features
            bb_numerics=list(),
            bb_strings=list(),
            # BB numerical-type features
            n_numeric_consts=0,
            n_string_consts=0,
            n_instructions=0,
            n_arith_instrs=0,
            n_call_instrs=0,
            n_logic_instrs=0,
            n_transfer_instrs=0,
            n_redirect_instrs=0
        )
        return features_dict

    # Get the BB bytes, disassembly, mnemonics and other features
    bb_mnems, bb_disasm, bb_norm ,bb_numerics= \
        capstone_disassembly(md,bb.va,bb.size)

    # Get static strings from the BB
    bb_strings = get_bb_strings(bb)
    
    counts = get_instruction_counts(bb_mnems, arch)

    features_dict = dict(
        bb_size=bb.size,
        #bb_mnems=bb_mnems,
        bb_disasm=bb_disasm,
        #bb_norm=bb_norm,
        # BB list-type features
        bb_numerics=bb_numerics,
        bb_strings=bb_strings,
        # BB numerical-type features
        n_numeric_consts=len(bb_numerics),
        n_string_consts=len(bb_strings),
        n_instructions=len(bb_mnems),
        n_arith_instrs=counts['arithmetic'],
        n_call_instrs=counts['call'],
        n_logic_instrs=counts['logic'],
        n_transfer_instrs=counts['transfer'],
        n_redirect_instrs=counts['redirect'],
    )
    return features_dict


#函数调用链分析
def get_related_functions(entry_func):
    """递归获取与入口函数相关的所有函数"""
    related_funcs = set()
    worklist = [entry_func]
    while worklist:
        func = worklist.pop()
        if func in related_funcs:
            continue
        related_funcs.add(func)
        callees = idaapi.get_func_callees(func)
        worklist.extend(callees)
    return related_funcs


#对系统库函数，可能会造成非常大的计算开销
#外部库函数通常在 .plt 或 .idata 段中，根据段名过滤
def is_library_function(func_addr):
    segment_name = idc.get_segm_name(func_addr)
    return segment_name in [".plt", ".idata"]

# 用户代码段通常在 .text
def is_user_defined_function(addr):
    segment_name = idc.get_segm_name(addr)
    return segment_name in [".text"]  # 用户代码段通常在 .text


def build_cfg(target_file, output_dir):
    # 创建输出目录
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    output_dict = dict()
    output_dict[target_file] = dict()

    # 获取基本信息
    procname = idaapi.get_inf_structure().procName.lower()
    bitness = get_bitness()

    output_dict[target_file]['arch'] = convert_procname_to_str(procname, bitness)
    md, arch = get_capstone(procname, bitness)

    # 初始化函数列表
    output_dict[target_file]['lib_func_list'] = list()

    # 遍历所有函数
    for func in idautils.Functions():
        func_name = idc.get_func_name(func)  # 获取函数名
        func_addr = hex(func)  # 获取函数地址并转为16进制字符串

        print(f"Found function '{func_name}' at address {func_addr}")

        if is_library_function(func):
            # 将函数名和地址存入字典
            output_dict[target_file]['lib_func_list'].append(
                {'fname': func_name, 'addr': func_addr})
        # 记录 main 函数的地址
        if func_name == "main":
            main_addr = hex(func)

        if is_user_defined_function(func):
            nodes_set, edges_set = set(), set()
            bbs_dict = dict()
            for bb in get_func_bb(func):
                nodes_set.add(hex(bb.va))
                # 遍历后继 bb 节点，添加边信息
                for next_ea in bb.succs:
                    edges_set.add((hex(bb.va), hex(next_ea)))
                bbs_dict[bb.va] = get_bb_feature(md, bb, arch)
            func_dict = {
                'fname': func_name,
                'nodes': list(nodes_set),
                'edges': list(edges_set),
                'basic_blocks': bbs_dict,
                # 'call_chain': get_related_functions(func)
            }
            output_dict[target_file][hex(func)] = func_dict

    output_dict[target_file]['main_addr'] = main_addr

    # 保存输出
    base_name, ext = os.path.splitext(os.path.basename(target_file))
    out_filename = f"{base_name}_cfg.json"
    with open(os.path.join(output_dir, out_filename), "w") as f_out:
        json.dump(output_dict, f_out)


def build_cfg_func(target_file, output_dir, fname=None):
    """生成指定函数的 CFG 和反汇编代码。"""
    # 创建输出目录
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    output_dict = dict()
    output_dict[target_file] = dict()

    # 获取基本信息
    procname = idaapi.get_inf_structure().procName.lower()
    bitness = get_bitness()

    output_dict[target_file]['arch'] = convert_procname_to_str(procname, bitness)
    md, arch = get_capstone(procname, bitness)

    # 遍历所有函数
    for func in idautils.Functions():
        func_name = idc.get_func_name(func)  # 获取函数名
        func_addr = hex(func)  # 获取函数地址并转为16进制字符串

        # 如果指定了 fname，则只处理匹配的函数
        if fname and func_name != fname:
            continue

        print(f"Found function '{func_name}' at address {func_addr}")

        if is_user_defined_function(func):
            nodes_set, edges_set = set(), set()
            bbs_dict = dict()
            for bb in get_func_bb(func):
                nodes_set.add(hex(bb.va))
                # 遍历后继 bb 节点，添加边信息
                for next_ea in bb.succs:
                    edges_set.add((hex(bb.va), hex(next_ea)))
                bbs_dict[bb.va] = get_bb_feature(md, bb, arch)
            func_dict = {
                'fname': func_name,
                'nodes': list(nodes_set),
                'edges': list(edges_set),
                'basic_blocks': bbs_dict,
            }
            output_dict[target_file][hex(func)] = func_dict

    # 保存输出
    base_name, ext = os.path.splitext(os.path.basename(target_file))
    out_filename = f"{base_name}_{fname}_acfg.json"
    with open(os.path.join(output_dir, out_filename), "w") as f_out:
        json.dump(output_dict, f_out)

def get_function_size(func):
    """获取函数的大小（字节数）。"""
    start_ea = idc.get_func_attr(func, idc.FUNCATTR_START)
    end_ea = idc.get_func_attr(func, idc.FUNCATTR_END)
    return end_ea - start_ea

def build_cfg_for_all_functions(target_file, output_dir):
    """生成所有符合条件的函数的acfg。"""
    # 创建输出目录
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    # 遍历所有函数
    for func in idautils.Functions():
        size = get_function_size(func)
        func_name = idc.get_func_name(func)
        # 筛选函数大小在合理范围内（例如 10 到 1000 字节）
        if 20 <= size <= 512:
            print(f"Processing function: {func_name} (size: {size} bytes)")
            build_cfg_func(target_file, output_dir, func_name)

if __name__ == '__main__':
    if not idaapi.get_plugin_options("cfg_builder"):
        print("[!] -Ocfg_builder option is missing")
        idc.Exit(1)
    plugin_options = idaapi.get_plugin_options("cfg_builder").split(";")
    target_file = plugin_options[0]
    output_dir = plugin_options[1]
    fname = None  # 默认值为 None
    if len(plugin_options) > 2:  # 如果指定了 fname
        fname = plugin_options[2]

    print("[D] Starting to generate the cfg.json of targetfile {}".format(target_file))

    # 根据 fname 是否指定选择调用哪个函数
    if fname:
        if(fname=='Full-file-analysis'):
            build_cfg_for_all_functions(target_file, output_dir)
            print("[D] Starting to generate the acfg.json of full func in targetfile {}".format(target_file))
        else:
            # 如果指定了 fname，调用 build_cfg_simple
            build_cfg_func(target_file, output_dir, fname)
    else:
        # 如果未指定 fname，调用 build_cfg
        build_cfg(target_file, output_dir)

    idc.qexit(0)
