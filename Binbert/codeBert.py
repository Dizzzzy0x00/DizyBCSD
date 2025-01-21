import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import os
import re

from torch.cuda.amp import GradScaler, autocast
#from torch.amp import GradScaler, autocast
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'


from tqdm import tqdm
import logging

# 设置日志记录
logging.basicConfig(
    filename="training_log.log", 
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def preprocess_instruction(ins):
    """
    预处理汇编指令，规范化字符串、常量和函数调用。
    
    返回:
        str: 预处理后的指令，例如 "mov eax, [rax+<const>]"
    """
    # 将字符串字面量替换为 <str>
    ins = re.sub(r'".*?"', '<str>', ins)

    # 处理十六进制常量
    def replace_hex_const(match):
        hex_const = match.group(0)
        if len(hex_const) >= 7:  # 0x + 5 digits or more
            return '[addr]'
        else:
            return hex_const  # 保留较短的常量，后续可以编码为 one-hot 向量

    ins = re.sub(r'\b0x[0-9a-fA-F]+\b', replace_hex_const, ins)

    # 处理十进制常量
    def replace_dec_const(match):
        dec_const = match.group(0)
        if len(dec_const) >= 5:  # 5 digits or more
            return '[addr]'
        else:
            return dec_const  # 保留较短的常量，后续可以编码为 one-hot 向量

    ins = re.sub(r'\b\d+\b', replace_dec_const, ins)

    # 处理函数调用
    # 假设函数调用格式为 "call <function_name>"
    #if ins.startswith("call"):
        #parts = ins.split()
        #if len(parts) >= 2:
            #func_name = parts[1]
            #if func_name not in external_functions:
                # 将内部函数调用替换为 <function>
                #ins = "call <function>"

    return ins

def preprocess_disassembly(disassembly, max_length=512):
    """
    预处理反汇编代码列表，确保每条代码的长度一致。
    
    参数:
        disassembly (list): 反汇编代码列表。
        external_functions (set): 外部函数名称集合。
        max_length (int): 每条代码的最大长度，默认为 512。
    
    返回:
        list: 预处理后的反汇编代码列表。
    """
    preprocessed_disassembly = []
    for ins in disassembly:
        preprocessed_ins = preprocess_instruction(ins)
        preprocessed_disassembly.append(preprocessed_ins)
    
    # 截断或填充到固定长度
    preprocessed_disassembly = [ins[:max_length] for ins in preprocessed_disassembly]
    return preprocessed_disassembly


# 定义 Triplet Loss
class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        positive_distance = torch.norm(anchor - positive, p=2, dim=1)
        negative_distance = torch.norm(anchor - negative, p=2, dim=1)
        loss = torch.mean(torch.clamp(positive_distance - negative_distance + self.margin, min=0.0))
        return loss

def custom_collate_fn(batch):
    """
    自定义 collate_fn，仅处理 disassembly 字段。
    
    参数:
        batch (list): 批次数据，格式为 [(anchor_disassembly, positive_disassembly, negative_disassembly), ...]
    
    返回:
        dict: 包含编码后的锚点、正样本和负样本。
    """
    anchors, positives, negatives = [], [], []
    for item in batch:
        anchors.append(item[0])  # anchor_disassembly
        positives.append(item[1])  # positive_disassembly
        negatives.append(item[2])  # negative_disassembly

    # 使用 CodeBERT 的 tokenizer 对汇编指令进行编码
    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    anchor_encodings = tokenizer(anchors, padding=True, truncation=True, return_tensors="pt", max_length=512)
    positive_encodings = tokenizer(positives, padding=True, truncation=True, return_tensors="pt", max_length=512)
    negative_encodings = tokenizer(negatives, padding=True, truncation=True, return_tensors="pt", max_length=512)

    return {
        "anchor": anchor_encodings,
        "positive": positive_encodings,
        "negative": negative_encodings,
    }

class TripletDataset(Dataset):
    def __init__(self, csv_files):
        """
        :param csv_files: 包含三元组数据的 CSV 文件列表
        """
        self.csv_files = csv_files
        self.triplets = self._load_triplets()

    def _load_triplets(self):
        triplets = []
        for csv_file in self.csv_files:
            print(f"Loading data from {csv_file}...")
            for chunk in pd.read_csv(csv_file, chunksize=10000):  # 增大 chunksize
                # 批量预处理
                anchor_list = chunk["anchor_disassembly"].apply(lambda x: "\n".join(preprocess_disassembly(x.split("\n")))).tolist()
                positive_list = chunk["positive_disassembly"].apply(lambda x: "\n".join(preprocess_disassembly(x.split("\n")))).tolist()
                negative_list = chunk["negative_disassembly"].apply(lambda x: "\n".join(preprocess_disassembly(x.split("\n")))).tolist()
                # 批量添加到 triplets
                triplets.extend(zip(anchor_list, positive_list, negative_list))
        print(f"Total triplets loaded: {len(triplets)}")
        return triplets

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        anchor, positive, negative = self.triplets[idx]
        return anchor, positive, negative  # 直接返回 disassembly 字段

class EmbeddingModel(nn.Module):
    def __init__(self, base_model_name, embedding_dim=128):
        super(EmbeddingModel, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        self.base_model = AutoModel.from_pretrained(base_model_name)
        self.fc = nn.Linear(self.base_model.config.hidden_size, embedding_dim)

    def forward(self, encodings):
        """
        输入汇编代码的编码，输出128维嵌入向量。
        """
        outputs = self.base_model(**encodings).last_hidden_state.mean(dim=1)  # 平均池化
        embeddings = self.fc(outputs)
        return embeddings

def train_model(train_dataset, test_dataset, base_model_name, epochs=2, batch_size=32, learning_rate=2e-5, save_dir="/root/autodl-fs/BinBert/models", resume_from_checkpoint=None):
    """
    微调训练流程，支持训练集和测试集，并添加进度条和日志记录。
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print("create save_dir")

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=custom_collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=custom_collate_fn)
    
    model = EmbeddingModel(base_model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model = model.to(device)
    triplet_loss = TripletLoss(margin=1.0)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scaler = GradScaler()

    # 如果提供了检查点路径，则加载模型和优化器状态
    start_epoch = 0
    start_batch = 0
    if resume_from_checkpoint and os.path.exists(resume_from_checkpoint):
        checkpoint = torch.load(resume_from_checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        start_batch = checkpoint['batch'] + 1  # 从下一个 batch 开始
        print(f"Resuming training from epoch {start_epoch}, batch {start_batch}")

    checkpoint_interval = 1000  # 每 1000 个 batch 保存一次检查点

    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_loss = 0
        # 添加训练进度条
        train_progress = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs} [Train]")  
        for i, batch in enumerate(train_dataloader):
            if epoch == start_epoch and i < start_batch:
                continue  # 跳过已经训练过的 batch

            optimizer.zero_grad()
            with autocast():
                anchor_embeddings = model(batch["anchor"].to(device))
                positive_embeddings = model(batch["positive"].to(device))
                negative_embeddings = model(batch["negative"].to(device))
                loss = triplet_loss(anchor_embeddings, positive_embeddings, negative_embeddings)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()
            
            train_progress.set_postfix(loss=epoch_loss / (i + 1))
            
            # 定期保存检查点
            if (i + 1) % checkpoint_interval == 0:
                checkpoint_path = os.path.join(save_dir, "latest_checkpoint.pth")  
                torch.save({
                    'epoch': epoch,
                    'batch': i,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss.item(),
                }, checkpoint_path)
                logging.info(f"Checkpoint of {epoch + 1} and batch {i + 1} saved at {checkpoint_path}")
                logging.info(f"Epoch {epoch+1}/{epochs} and batch {i + 1} - Train Loss: {epoch_loss / len(train_dataloader):.4f}")

        logging.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {epoch_loss / len(train_dataloader):.4f}")

        # 测试过程
        model.eval()
        test_loss = 0
        test_progress = tqdm(test_dataloader, desc=f"Epoch {epoch+1}/{epochs} [Test]")  # 测试进度条

        with torch.no_grad():
            for batch in test_progress:
                anchor_embeddings = model(batch["anchor"].to(device))
                positive_embeddings = model(batch["positive"].to(device))
                negative_embeddings = model(batch["negative"].to(device))
                loss = triplet_loss(anchor_embeddings, positive_embeddings, negative_embeddings)
                test_loss += loss.item()
                test_progress.set_postfix(loss=test_loss / (test_progress.n + 1))

        logging.info(f"Epoch {epoch+1}/{epochs} - Test Loss: {test_loss / len(test_dataloader):.4f}")

        # 保存模型
        model_path = os.path.join(save_dir, f"triplet_model_epoch_{epoch + 1}.pth")
        torch.save(model.state_dict(), model_path)
        logging.info(f"Model saved at {model_path}")

    return model

# 主函数
if __name__ == "__main__":
    base_dir = "/root/autodl-fs/BinBert"
    train_csv_files = [
        os.path.join(base_dir, "DataSet", "train.csv"),
    ]
    test_csv_file = [os.path.join(base_dir, "DataSet", "test.csv")]  # 测试集文件

    # 加载数据集
    train_dataset = TripletDataset(train_csv_files)
    test_dataset = TripletDataset(test_csv_file)

    print(f"Total triplets in dataset: {len(train_dataset)}")

    # 使用 CodeBERT 作为预训练模型
    base_model_name = "microsoft/codebert-base"

    # 训练模型
    checkpoint_path = "/root/autodl-fs/BinBert/models/latest_checkpoint.pth"
    model = train_model(
        train_dataset,
        test_dataset,
        base_model_name,
        epochs=2,
        batch_size=32,
        learning_rate=2e-5,
        save_dir="/root/autodl-fs/BinBert/models",
        resume_from_checkpoint=checkpoint_path  # 从检查点恢复训练
    )

    # 保存模型
    torch.save(model.state_dict(), "triplet_model_codebert.pth")