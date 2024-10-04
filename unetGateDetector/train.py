
from sympy import false
from torch.utils.tensorboard import SummaryWriter
from model.unet_model import UNet
from utils.dataset import MyDataLoader
from torch import optim
import torch.nn as nn
import torch
import os

def train_net(net, device, data_path, classesnum):
    """ 传入参数
    :param net:网络模型
    :param device: GPU还是CPU
    :param data_path: 训练集目录
    :param classesnum: 分类数量
    :return:

    """

    # ----------------------------#
    # 定义
    epochs = 300
    batch_size = 5
    lr = 0.003
    betas = (0.9, 0.999)
    weight_decay = 0
    # 加载训练集
    dataset = MyDataLoader(data_path, num_classes=classesnum)
    train_loader = torch.utils.data.DataLoader(dataset=dataset,
                                               batch_size=batch_size,
                                               shuffle=false)

    # 定义Adam算法
    optimizer = optim.Adam(net.parameters(), lr=lr, betas=betas,
                           weight_decay=weight_decay)
    # 定义Loss算法
    criterion = nn.CrossEntropyLoss()
    # best_loss统计，初始化为正无穷
    best_loss = float('inf')

    # 建立可视化
    visual_path = './runs'
    os.makedirs(visual_path, exist_ok=True)
    writer = SummaryWriter()
    # 训练epochs次
    for epoch in range(epochs):
        # 训练模式
        net.train()
        # 按照batch_size开始训练
        for image, label in train_loader:
            optimizer.zero_grad()
            # 将数据拷贝到device中
            image = image.to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=torch.float32)  # 确保标签为long类型

            # 使用网络参数，输出预测结果
            pred = net(image)

            # 计算loss
            loss = criterion(pred, label)
            print('Loss/train', loss.item())

            # 保存loss值最小的网络参数
            if loss < best_loss:
                best_loss = loss
                torch.save(net.state_dict(), 'best_model.pth')

            # 更新参数
            loss.backward()
            optimizer.step()
            writer.add_scalar('Loss/train', loss.item(), epoch)


if __name__ == "__main__":
    # 选择设备，有cuda用cuda，没有就用cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 加载网络，图片单通道n_channel，分类为n_classes
    net = UNet(n_channels=1, n_classes=5)
    # 将网络拷贝到deivce中
    net.to(device=device)
    # 指定训练集地址，开始训练
    data_path = "dataSet/"
    train_net(net, device, data_path, classesnum=5)
