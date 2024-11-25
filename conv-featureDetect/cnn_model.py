import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split
import myData

# 检查是否有可用的 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# 定义简化的CNN模型（两层卷积层）
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        # 第一层卷积层，输入形状为(640, 480, 3)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)  # 输出通道数为32
        # 第二层卷积层
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # 输出通道数为64
        self.pool = nn.MaxPool2d(2, 2)  # 最大池化层，窗口大小为2x2
        # 展平特征图
        self.flatten = nn.Flatten()
        # 计算展平后的特征图大小，假设输入是 (640, 480) 尺寸的图像
        self.fc1_input_dim = 64 * 7 * 7  # (640/2/2, 480/2/2)
        # self.fc1_input_dim = 64 * 160 * 120  # (640/2/2, 480/2/2)
        # 全连接层
        self.fc1 = nn.Linear(self.fc1_input_dim, 512)  # 隐藏层
        self.fc2 = nn.Linear(512, 8)  # 输出8个数值，用于赛道门四个角的坐标

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))  # 第一层卷积和池化
        x = self.pool(torch.relu(self.conv2(x)))  # 第二层卷积和池化
        x = self.flatten(x)  # 展平操作
        x = torch.relu(self.fc1(x))  # 全连接层1
        x = self.fc2(x)  # 输出层
        return x


# 初始化模型，并将模型移动到GPU
model = CNNModel().to(device)

# 打印模型结构
print(model)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Hyper parameters
num_epochs = 8
num_classes = 10
batch_size = 6
learning_rate = 0.001
transform = transforms.Compose([
    transforms.Resize((28, 28)),  # 调整图像大小
    transforms.ToTensor(),  # 转换为张量
])
# dataset
train_dataset = myData.ImageDataset(csv_file='./dataSet/img_labels.csv',
                                    img_dir='./dataSet/images',
                                    transform=transform)

# 按比例划分训练集和测试集，例如 80% 训练，20% 测试
train_size = int(0.8 * len(train_dataset))
test_size = len(train_dataset) - train_size
train_dataset, test_dataset = random_split(train_dataset, [train_size, test_size])

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)


for epoch in range(num_epochs):
    running_loss = 0.0
    model.train()  # 切换到训练模式
    for images, labels in train_loader:
        # 将输入数据和标签移动到GPU
        images, labels = images.to(device), labels.to(device)
        # 前向传播
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        # 反向传播并更新权重
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader)}")

# 评估模型
model.eval()  # 切换到评估模式
test_loss = 0.0
with torch.no_grad():  # 在评估期间不需要计算梯度
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        test_loss += loss.item()

# Save the model checkpoint
torch.save(model.state_dict(), 'model.ckpt')
print(f"Test Loss: {test_loss / len(test_loader)}")
