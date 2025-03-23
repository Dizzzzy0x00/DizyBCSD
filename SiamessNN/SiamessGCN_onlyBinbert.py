import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from dataprocess_onlyBinbert import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.loader import DataLoader
import torch.nn.init as init

class GraphEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim):
        super(GraphEncoder, self).__init__()
        # 定义多个GCN层
        self.conv1 = GCNConv(input_dim, hidden_dim1)  # (128, 256)
        self.conv2 = GCNConv(hidden_dim1, hidden_dim2)  # (256, 512)
        self.conv3 = GCNConv(hidden_dim2, hidden_dim2)  # (512, 512)
        self.dropout = nn.Dropout(0.3)  # Dropout
        self.fc = nn.Linear(hidden_dim2, output_dim)  # (512, 256) 

        # 初始化网络参数
        self.init_weights()

    def init_weights(self):
        # 初始化GCN层的权重
        init.kaiming_normal_(self.conv1.lin.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.conv2.lin.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.conv3.lin.weight, mode='fan_out', nonlinearity='relu')

        # 初始化全连接层的权重
        init.kaiming_normal_(self.fc.weight, mode='fan_out', nonlinearity='relu')
        if self.fc.bias is not None:
            init.constant_(self.fc.bias, 0)

    def forward(self, x, edge_index, batch):
        # 通过GCN层提取特征
        x = F.relu(self.conv1(x, edge_index))  # 第一层
        x = F.relu(self.conv2(x, edge_index))  # 第二层
        x = F.relu(self.conv3(x, edge_index))  # 第三层

        # 添加Dropout
        x = self.dropout(x)
        
        # 使用全局平均池化
        x = global_mean_pool(x, batch)  # 对图的节点特征进行池化，返回图级特征

        # 通过全连接层输出
        x = self.fc(x)
        
        return x






# GNN孪生网络
class SiameseGNN(nn.Module):
    def __init__(self, encoder):
        super(SiameseGNN, self).__init__()
        self.encoder = encoder

    def forward(self, data1, data2, data3):
        # 分别计算三元组中每个图的嵌入
        emb1 = self.encoder(data1.x, data1.edge_index, data1.batch)
        emb2 = self.encoder(data2.x, data2.edge_index, data2.batch)
        emb3 = self.encoder(data3.x, data3.edge_index, data3.batch)
        return emb1, emb2, emb3


class TripletLoss(nn.Module):
    def __init__(self, margin=20.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, emb1, emb2, emb3):
        # 计算正样本和负样本的欧氏距离
        pos_distance = torch.norm(emb1 - emb2, p=2, dim=1)
        neg_distance = torch.norm(emb1 - emb3, p=2, dim=1)
        # 计算三元组损失
        loss = torch.clamp(pos_distance - neg_distance + self.margin, min=0)
        return loss.mean()


class PairwiseLoss(nn.Module):
    def __init__(self, margin=0.1):
        super(PairwiseLoss, self).__init__()
        self.margin = margin

    def forward(self, emb1, emb2, labels):
        # 计算余弦相似度
        similarity = F.cosine_similarity(emb1, emb2)
        
        # 正样本 (label = 1)
        positive_loss = labels * (1 - similarity)
        
        # 负样本 (label = 0)
        negative_loss = (1 - labels) * F.relu(self.margin - similarity)
        
        # 计算最终损失
        loss = positive_loss + negative_loss
        return loss.mean()




import logging

# 日志配置
def setup_logger(log_file="TripletGNN_training.log"):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="a"),  # 追加日志
            logging.StreamHandler()  # 终端输出
        ]
    )

# 三元组训练函数
def train(model, loader, optimizer, loss_fn, device, epoch, save_dir="./model_onlyBinbert/TripletGNN", 
          checkpoint_path="./model_onlyBinbert/latest_checkpoint.pth", log_file="./training_onlyBinbert.log"):
    setup_logger(log_file)
    logger = logging.getLogger()

    model.train()
    total_loss = 0
    progress_bar = tqdm(loader, desc=f"Epoch {epoch+1} [Training]")

    # 用于存储前十个批次的正负样本相似度信息
    batch_count = 0

    for batch_idx, (data1, data2, data3) in enumerate(progress_bar):
        data1, data2, data3 = data1.to(device), data2.to(device), data3.to(device)

        optimizer.zero_grad()
        emb1, emb2, emb3 = model(data1, data2, data3)
        loss = loss_fn(emb1, emb2, emb3)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # 输出日志信息
        if batch_count < 10:
            pos_distance = torch.norm(emb1 - emb2, p=2, dim=1).detach().cpu().numpy()
            neg_distance = torch.norm(emb1 - emb3, p=2, dim=1).detach().cpu().numpy()
            logger.info(f"Batch {batch_idx}: Positive Distance - Mean: {pos_distance.mean():.4f}, Min: {pos_distance.min():.4f}, Max: {pos_distance.max():.4f}")
            logger.info(f"Batch {batch_idx}: Negative Distance - Mean: {neg_distance.mean():.4f}, Min: {neg_distance.min():.4f}, Max: {neg_distance.max():.4f}")
            batch_count += 1

        progress_bar.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(loader)
    logger.info(f"Epoch {epoch+1}: Train Loss: {avg_loss:.4f}")

    # 保存模型和检查点
    os.makedirs(save_dir, exist_ok=True)
    epoch_model_path = os.path.join(save_dir, f"TripletGNN_epoch_{epoch+1}.pth")
    torch.save(model.state_dict(), epoch_model_path)
    logger.info(f"Model saved at {epoch_model_path}")

    checkpoint = {
        "epoch": epoch + 1,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "loss": avg_loss
    }
    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Checkpoint saved at {checkpoint_path}")

    return avg_loss

# 评估函数
def evaluate(model, loader, loss_fn, device, epoch):
    model.eval()
    total_loss = 0
    progress_bar = tqdm(loader, desc=f"Epoch {epoch+1} [Validation]")  # 添加进度条

    with torch.no_grad():
        for data1, data2, data3 in progress_bar:
            data1, data2, data3 = data1.to(device), data2.to(device), data3.to(device)

            emb1, emb2, emb3 = model(data1, data2, data3)
            loss = loss_fn(emb1, emb2, emb3)
            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())  # 实时显示 loss

    avg_loss = total_loss / len(loader)
    print(f"Epoch {epoch+1}: Val Loss: {avg_loss:.4f}")
    return avg_loss




from torch.utils.data import Dataset

class GraphPairDataset(Dataset):
    def __init__(self, data_pairs, labels):
        """
        :param data_pairs: [(data1, data2), ...]
        :param labels: [1, 0, ...] (正样本或负样本)
        """
        self.data_pairs = data_pairs
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data_pairs[idx][0], self.data_pairs[idx][1], self.labels[idx]
        
def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, GCNConv):
            init.kaiming_normal_(m.lin.weight, mode='fan_out', nonlinearity='relu')
            if m.lin.bias is not None:
                init.constant_(m.lin.bias, 0)


if __name__ == "__main__":
    # 参数
    input_dim = 128  # 节点特征维度
    hidden_dim1 = 256
    hidden_dim2 = 512
    output_dim = 256
    margin = 20
    batch_size = 32
    epochs = 10
    learning_rate = 0.001
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_path = "./model_onlyBinbert/SiameseGNN_onlyBinbert"  # 模型保存路径
    save_interval = 1  # 每 1 轮保存一次模型
    checkpoint_path="./model_onlyBinbert/latest_checkpoint.pth"
    log_file="./training_onlyBinbert.log"
    
    # 初始化模型 & 优化器
    encoder = GraphEncoder(input_dim, hidden_dim1,hidden_dim2, output_dim)
    model = SiameseGNN(encoder).to(device)

    #initialize_weights(model)
    #initialize_weights(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = TripletLoss(margin)

    # 读取数据集
    csv_file = "./dataset/acfg.csv"
    model_path = "./BinbertModels/triplet_model_epoch_2.pth"
    print("start dataprocess...")
    #train_loader, val_loader = prepare_datasets       (csv_file, model_path, device, test_size=0.2, batch_size=batch_size)
    train_loader, val_loader = prepare_triplet_datasets(csv_file, model_path, device, test_size=0.2, batch_size=batch_size)
    print("finish dataprocess...")
    # 训练 & 评估
    for epoch in range(epochs):
        train_loss = train(model, train_loader, optimizer, loss_fn, device, epoch, save_path, checkpoint_path, log_file)
        val_loss = evaluate(model, val_loader, loss_fn, device, epoch)

