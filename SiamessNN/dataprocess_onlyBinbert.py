import os
import re
import pandas as pd
import torch
import numpy as np
from torch_geometric.data import Data
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset
import random
import torch.nn as nn
from torch_geometric.loader import DataLoader as GeometricDataLoader

# 设置 'spawn' 启动方法
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)
import itertools
import random
from tqdm import tqdm  # 导入 tqdm

def parse_file_path(file_path):
    """
    解析文件路径，提取架构、编译器优化器和函数文件。
    """
    filename = file_path.rpartition('\\')[-1]  # 获取文件名部分
    parts = filename.split('_')  # 按下划线分割
    
    arch_compiler = parts[0]  # 获取 arch 和 compiler-opt 部分（例如 arm32-gcc-7-O2）
    func_file = '_'.join(parts[1:]).replace('.json', '')  # 获取 func_file，去掉文件扩展名
    
    # 分割 arch 和 compiler_opt
    arch = arch_compiler.split('-')[0]  # 获取架构部分，例如 arm32
    compiler_opt = '-'.join(arch_compiler.split('-')[1:])  # 获取编译器优化器部分，例如 gcc-7-O2

    return {
        'arch': arch,
        'compiler_opt': compiler_opt,
        'func_file': func_file
    }

# 数据预处理逻辑
from transformers import AutoTokenizer, AutoModel

# CodeBERT 预处理函数
def preprocess_disassembly(disassembly, max_length=512):
    """
    预处理汇编指令，确保格式统一。
    """
    preprocessed_disassembly = []
    for ins in disassembly:
        ins = re.sub(r'".*?"', '<str>', ins)  # 处理字符串
        ins = re.sub(r'\b0x[0-9a-fA-F]+\b', '[addr]', ins)  # 处理地址
        ins = re.sub(r'\b\d+\b', '[num]', ins)  # 处理数字
        preprocessed_disassembly.append(ins[:max_length])  # 统一长度
    return preprocessed_disassembly

# CodeBERT 模型类
class CodeBERTEmbedding(nn.Module):
    def __init__(self, base_model_name="microsoft/codebert-base", embedding_dim=128):
        super(CodeBERTEmbedding, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        self.base_model = AutoModel.from_pretrained(base_model_name)
        self.fc = nn.Linear(self.base_model.config.hidden_size, embedding_dim)

    def forward(self, disassembly, device):
        """
        计算汇编指令嵌入，返回 (num_nodes, 128) 矩阵
        """
        tokens = self.tokenizer(disassembly, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        with torch.no_grad():
            embeddings = self.base_model(**tokens).last_hidden_state.mean(dim=1)
        return self.fc(embeddings)
        
# 加载模型
class EmbeddingModel(nn.Module):
    def __init__(self, base_model_name="microsoft/codebert-base", embedding_dim=128):
        super(EmbeddingModel, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        self.base_model = AutoModel.from_pretrained(base_model_name)
        self.fc = nn.Linear(self.base_model.config.hidden_size, embedding_dim)

    def forward(self, encodings):
        outputs = self.base_model(**encodings).last_hidden_state.mean(dim=1)
        embeddings = self.fc(outputs)
        return embeddings



def load_trained_model(model_path, base_model_name="microsoft/codebert-base", device="cuda"):
    """
    加载训练好的模型。
    """
    model = EmbeddingModel(base_model_name)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

# 构造图数据的方法
# 仅仅使用128维Binbert嵌入信息，不使用11维统计数据作为节点特征
def process_function(row, binbert, device):
    try:
        bb_disasm = eval(row['bb_disasm'])
        if len(bb_disasm) == 0:
            return None  # 跳过空数据

        disasm_features = []
        for instructions in bb_disasm:
            processed_instructions = "\n".join(preprocess_disassembly(instructions))
            encodings = binbert.tokenizer(processed_instructions, padding=True, truncation=True, return_tensors="pt").to(device)
            with torch.no_grad():
                embedding = binbert(encodings).squeeze().cpu().numpy()  # 确保输出为 (128,)
            disasm_features.append(embedding)

        disasm_features = np.array(disasm_features, dtype=np.float32)
        if disasm_features.ndim == 1:
            disasm_features = disasm_features.reshape(1, -1)
        assert disasm_features.shape[1] == 128, f"特征维度错误: {disasm_features.shape}"
        
        node_features = disasm_features  # (num_nodes, 128)
        edges = eval(row['edges'])

        # 构建节点索引映射
        unique_nodes = set()
        for src, dst in edges:
            unique_nodes.add(src)
            unique_nodes.add(dst)
        addr_to_idx = {addr: idx for idx, addr in enumerate(sorted(unique_nodes))}
    
        # 转换边索引
        edge_index = torch.tensor([[addr_to_idx[e[0]], addr_to_idx[e[1]]] for e in edges], dtype=torch.long).t().contiguous()
        num_nodes = len(addr_to_idx)
        self_loops = torch.arange(num_nodes).view(1, -1).repeat(2, 1)
        edge_index = torch.cat([edge_index, self_loops], dim=1)
    
        return Data(x=torch.tensor(node_features, dtype=torch.float), edge_index=edge_index)
        
    except Exception as e:
        print(f"处理数据失败: {e}")
        return None
   

# 构造三元组的方法
def construct_triplets(df, target_triplets=80000):
    df_parsed = pd.DataFrame([parse_file_path(fp) for fp in df['file_path']])
    df = pd.concat([df, df_parsed], axis=1)

    grouped = df.groupby(['compiler_opt', 'func_file'])

    print("Generating triplets...")
    triplets = []

    for _, group in tqdm(grouped, desc="Processing groups", total=len(grouped)):
        if len(group) > 1:
            pos_pairs = list(itertools.combinations(group.index, 2))
            for idx1, idx2 in pos_pairs:
                # 获取当前正样本对的值
                compiler_opt_1, func_file_1 = df.loc[idx1, ['compiler_opt', 'func_file']]
                
                # 随机选择负样本直到找到符合要求的
                while True:
                    neg_idx = random.choice(df.index)
                    neg_compiler_opt, neg_func_file = df.loc[neg_idx, ['compiler_opt', 'func_file']]
                    
                    # 检查负样本是否符合要求
                    #if (neg_compiler_opt != compiler_opt_1) or (neg_func_file != func_file_1):
                    if ( neg_func_file != func_file_1 ):
                        break  # 找到符合要求的负样本，跳出循环

                #if(len(triplets)>target_triplets+10):
                    #break
                # 生成三元组
                #else:
                
                triplets.append((idx1, idx2, neg_idx))

    # 随机抽取所需数量的三元组
    triplets = random.sample(triplets, min(len(triplets), target_triplets))

    print(f"生成的三元组数量: {len(triplets)}")
    return triplets


# 数据集封装
class GraphTripletDataset(Dataset):
    def __init__(self, df, triplets, binbert, device):
        self.df = df
        self.triplets = triplets
        self.binbert = binbert
        self.device = device

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        triplet = self.triplets[idx]
        idx1, idx2, idx3 = triplet

        row1 = self.df.iloc[idx1]
        row2 = self.df.iloc[idx2]
        row3 = self.df.iloc[idx3]

        data1 = process_function(row1, self.binbert, self.device)
        data2 = process_function(row2, self.binbert, self.device)
        data3 = process_function(row3, self.binbert, self.device)

        return data1, data2, data3
        
from sklearn.model_selection import train_test_split

# 数据加载器
def prepare_triplet_datasets(csv_file, model_path, device, test_size=0.2, batch_size=16):
    binbert = load_trained_model(model_path, device=device)
    df = pd.read_csv(csv_file)
    print("Finish data reading.")

    #triplets = construct_triplets(df,target_triplets=800)
    triplets = construct_triplets(df)
    train_triplets, val_triplets = train_test_split(triplets, test_size=test_size, random_state=42)

    train_dataset = GraphTripletDataset(df, train_triplets, binbert, device)
    train_loader = GeometricDataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    
    val_dataset = GraphTripletDataset(df, val_triplets, binbert, device)
    val_loader = GeometricDataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)


    return train_loader, val_loader

if __name__ == "__main__":
    csv_file = "./dataset/acfg.csv"
    model_path = "./BinbertModels/triplet_model_epoch_2.pth"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_loader, val_loader = prepare_triplet_datasets(csv_file, model_path, device, 0.1, 32)

    for batch in train_loader:
        data1, data2, data3 = batch
        print(f"Data1: {data1}")
        print(f"Data2: {data2}")
        print(f"Data3: {data3}")
        break
