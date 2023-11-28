import platform

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


class TransformerTSClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, d_model=512, nhead=8, num_layers=3):
        super().__init__()

        self.embedding = nn.Sequential(
            nn.Linear(1, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, d_model),
        )

        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead), num_layers
        )

        self.decoder = nn.Sequential(nn.Linear(d_model, num_classes), nn.Softmax(-1))

    def forward(self, x):
        x = x.unsqueeze(-1)
        x = self.embedding(x)
        x = self.transformer_encoder(x)
        x = self.decoder(x[:, -1, :])  # 取最后一个时间步的输出作为预测结果
        return x


class MyDataset(Dataset):
    def __init__(self, csv_path, txt_path):
        self.csv_data = pd.read_csv(csv_path)
        self.txt_data = open(txt_path, "r").readlines()

        # 获取样本数量、特征数量和类别数量
        self.num_samples = len(self.csv_data.columns)
        self.num_features = len(self.csv_data)
        self.num_classes = len(self.txt_data[0].strip().split(","))

        # 将数据和标签划分为训练集和测试集
        self.samples = []
        self.labels = []
        for i in range(self.num_samples):
            sample = []
            for j in range(self.num_features):
                sample.append(self.csv_data.iloc[j, i])
            self.samples.append(sample)
            self.labels.append(list(map(int, self.txt_data[i].strip().split("\t"))))

        train_size = int(0.8 * self.num_samples)  # 训练集大小
        test_size = self.num_samples - train_size  # 测试集大小
        # self.train_set = {'data': self.samples[:train_size], 'target': self.labels[:train_size]}
        # self.test_set = {'data': self.samples[train_size:], 'target': self.labels[train_size:]}
        self.train_set = [
            (torch.tensor(sample), torch.tensor(label))
            for sample, label in zip(
                self.samples[:train_size], self.labels[:train_size]
            )
        ]
        self.test_set = [
            (torch.tensor(sample), torch.tensor(label))
            for sample, label in zip(
                self.samples[train_size:], self.labels[train_size:]
            )
        ]

    def __getitem__(self, index):
        return (
            torch.tensor(self.train_set["data"][index]),
            torch.tensor(self.train_set["target"][index]),
            torch.tensor(self.test_set["data"][index]),
            torch.tensor(self.test_set["target"][index]),
        )

    def __len__(self):
        return len(self.train_set["data"]), len(self.test_set["data"])


if __name__ == "__main__":
    if platform.system() == "Darwin":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    dataset = MyDataset(
        "./x50_240.csv",
        "./label.txt",
    )

    train_dataset = dataset.train_set
    test_dataset = dataset.test_set

    train_loader = DataLoader(train_dataset, batch_size=6, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=6, shuffle=False)

    # 初始化模型
    input_dim = 50  # 输入特征维度
    num_classes = 3  # 分类类别数
    model = TransformerTSClassifier(input_dim, num_classes).to(device=device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    # 训练模型
    num_epochs = 100
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        model.train()
        for inputs, labels in train_loader:
            inputs, labels = torch.tensor(
                inputs, dtype=torch.float32, device=device
            ), torch.tensor(labels, dtype=torch.float32, device=device)
            optimizer.zero_grad()

            outputs = model(inputs)
            print(outputs)
            print(labels)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            predictions = torch.argmax(outputs, 1)
            target = torch.argmax(labels, 1)
            total += labels.size(0)
            correct += (predictions == target).sum().item()

            loss.backward()
            optimizer.step()

        train_loss = running_loss / len(train_loader.dataset)
        train_acc = 100 * correct / total

        # 在测试集上评估模型
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.float(), labels.long()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item()

                _, predictions = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predictions == labels).sum().item()

        test_loss = test_loss / len(test_loader.dataset)
        test_acc = 100 * correct / total

        # 输出当前训练轮次的损失和准确率
        print(
            f"Epoch {epoch + 1:5d} | Train Loss: {train_loss:.6f} | Train Acc: {train_acc:.2f}% | Test Loss: {test_loss:.6f} | Test Acc: {test_acc:.2f}%"
        )
