import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from dataprocess import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.loader import DataLoader
import torch.nn.init as init

from torch_geometric.nn import GINConv
from torch_geometric.nn import global_mean_pool

class GraphEncoderGIN(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim):
        super(GraphEncoderGIN, self).__init__()
        # 定义GIN的MLP
        self.mlp1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, hidden_dim1)
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Linear(hidden_dim2, hidden_dim2)
        )
        self.mlp3 = nn.Sequential(
            nn.Linear(hidden_dim2, hidden_dim2),
            nn.ReLU(),
            nn.Linear(hidden_dim2, hidden_dim2)
        )
        self.gin1 = GINConv(self.mlp1)
        self.gin2 = GINConv(self.mlp2)
        self.gin3 = GINConv(self.mlp3)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim2, output_dim)
        
        # 初始化网络参数
        self.init_weights()

    def init_weights(self):
        for layer in self.mlp1:
            if isinstance(layer, nn.Linear):
                init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                if layer.bias is not None:
                    init.constant_(layer.bias, 0)
        for layer in self.mlp2:
            if isinstance(layer, nn.Linear):
                init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                if layer.bias is not None:
                    init.constant_(layer.bias, 0)
        for layer in self.mlp3:
            if isinstance(layer, nn.Linear):
                init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                if layer.bias is not None:
                    init.constant_(layer.bias, 0)
        init.kaiming_normal_(self.fc.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x, edge_index, batch):
        # 通过GIN层传递
        x = F.relu(self.gin1(x, edge_index))  # 第一层GIN
        x = F.relu(self.gin2(x, edge_index))  # 第二层GIN
        x = F.relu(self.gin3(x, edge_index))  # 第三层GIN
        
        # 使用Dropout
        x = self.dropout(x)
        # 全局平均池化
        x = global_mean_pool(x, batch)
        # 通过全连接层输出
        x = self.fc(x)
        return x




# GNN孪生网络
class SiameseGIN(nn.Module):
    def __init__(self, encoder):
        super(SiameseGIN, self).__init__()
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
def setup_logger(log_file="TripletGIN_training.log"):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="a"),  # 追加日志
            logging.StreamHandler()  # 终端输出
        ]
    )

# 三元组训练函数
def train(model, loader, optimizer, loss_fn, device, epoch, save_dir="./model/TripletGIN", 
          checkpoint_path="./model/latest_checkpoint.pth", log_file="./training.log"):
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
def evaluate(model, loader, loss_fn, device, epoch, log_file):
    setup_logger(log_file)
    logger = logging.getLogger()
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    progress_bar = tqdm(loader, desc=f"Epoch {epoch+1} [Validation]")  # 添加进度条

    with torch.no_grad():
        for data1, data2, data3 in progress_bar:
            data1, data2, data3 = data1.to(device), data2.to(device), data3.to(device)

            emb1, emb2, emb3 = model(data1, data2, data3)
            loss = loss_fn(emb1, emb2, emb3)
            total_loss += loss.item()

            # 计算准确率
            pos_distance = torch.norm(emb1 - emb2, p=2, dim=1)
            neg_distance = torch.norm(emb1 - emb3, p=2, dim=1)
            correct += (pos_distance < neg_distance).sum().item()  # 或者使用 (pos_distance < neg_distance + margin)
            total += data1.size(0)

            progress_bar.set_postfix(loss=loss.item())  # 实时显示 loss

    avg_loss = total_loss / len(loader)
    accuracy = correct / total
    print(f"Epoch {epoch+1}: Val Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
    logger.info(f"Epoch {epoch+1}: Val Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
    return avg_loss, accuracy



def save_dataloader_to_file(dataloader, file_path):
    """
    将 dataloader 中的数据保存到文件
    :param dataloader: 需要保存的 DataLoader
    :param file_path: 保存的文件路径
    """
    data_list = []
    for data1, data2, data3 in dataloader:
        # 将数据加载到 CPU，并转换为 numpy 数组或者 Tensor
        data1 = data1.cpu()
        data2 = data2.cpu()
        data3 = data3.cpu()
        
        # 将每个批次的数据保存到列表中
        data_list.append((data1, data2, data3))
    
    # 保存整个数据集
    torch.save(data_list, file_path)
    print(f"Data saved to {file_path}")


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
    input_dim = 139  # 节点特征维度
    hidden_dim1 = 256
    hidden_dim2 = 512
    output_dim = 256
    margin = 20
    batch_size = 48
    epochs = 10
    learning_rate = 0.001
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_path = "./model/SiameseGIN"  # 模型保存路径
    save_interval = 1  # 每 1 轮保存一次模型
    checkpoint_path="./model/latest_checkpoint_GIN.pth"
    log_file="./training.log"
    
    # 初始化模型 & 优化器
    encoder = GraphEncoderGIN(input_dim, hidden_dim1,hidden_dim2, output_dim)
    model = SiameseGIN(encoder).to(device)

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
    
    val_loader_file = "./saved_val_loader_GIN.pth"
    save_dataloader_to_file(val_loader, val_loader_file)
    train_loader_file = "./saved_train_loader_GIN.pth"
    save_dataloader_to_file(train_loader, save_dataloader_to_file)

    print("finish dataprocess...")
    # 训练 & 评估
    for epoch in range(epochs):
        train_loss = train(model, train_loader, optimizer, loss_fn, device, epoch, save_path, checkpoint_path, log_file)
        val_loss = evaluate(model, val_loader, loss_fn, device, epoch, log_file)

