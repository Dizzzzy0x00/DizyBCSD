import idautils

from architecture import ARCH_MNEM

def get_bb_strings(bb):
    #获取bb中的字符串
    d_from = []
    strings = []
    for h in idautils.Heads(bb.va, bb.va + bb.size):
        for xf in idautils.DataRefsFrom(h):
            d_from.append(xf)
    string_list=list(idautils.Strings())
    for k in string_list:
        if k.ea in d_from:
            strings.append(str(k))
    return strings

def get_n_transfer_instrs(mnem_list, arch):
    #获取基本块中的传输指令数量
    return len([m for m in mnem_list if m in ARCH_MNEM[arch]['transfer']])

def get_n_redirect_instrs(mnem_list, arch):
    #获取条件、无条件和调用指令的数量
    temp_instrs = ARCH_MNEM[arch]['conditional'] | \
        ARCH_MNEM[arch]['unconditional'] | \
        ARCH_MNEM[arch]['call']

    return len([m for m in mnem_list if m in temp_instrs])

def get_n_call_instrs(mnem_list, arch):
    #获取基本块中的call指令数量
    return len([m for m in mnem_list if m in ARCH_MNEM[arch]['call']])

def get_n_arith_instrs(mnem_list, arch):
    #获取基本块中的算术运算指令数量
    return len([m for m in mnem_list if m in ARCH_MNEM[arch]['arithmetic']])


def get_n_logic_instrs(mnem_list, arch):
    #获取基本块中的逻辑运算指令数量
    return len([m for m in mnem_list if m in ARCH_MNEM[arch]['logic']])

def get_instruction_counts(mnem_list, arch):
    """
    一次遍历获取不同类型的指令计数。
    :param mnem_list: 基本块中的指令助记符列表。
    :param arch: 架构类型，用于从 ARCH_MNEM 获取指令集。
    :return: 字典，包含各类型指令的计数。
    """
    # 获取架构对应的指令分类
    arch_mnems = ARCH_MNEM[arch]
    transfer_set = arch_mnems['transfer']
    conditional_set = arch_mnems['conditional']
    unconditional_set = arch_mnems['unconditional']
    call_set = arch_mnems['call']
    arithmetic_set = arch_mnems['arithmetic']
    logic_set = arch_mnems['logic']

    # 初始化计数
    counts = {
        'transfer': 0,
        'redirect': 0,
        'call': 0,
        'arithmetic': 0,
        'logic': 0
    }

    # 一次遍历优化
    for m in mnem_list:
        if m in transfer_set:
            counts['transfer'] += 1
        if m in conditional_set or m in unconditional_set or m in call_set:
            counts['redirect'] += 1
        if m in call_set:
            counts['call'] += 1
        if m in arithmetic_set:
            counts['arithmetic'] += 1
        if m in logic_set:
            counts['logic'] += 1

    return counts
