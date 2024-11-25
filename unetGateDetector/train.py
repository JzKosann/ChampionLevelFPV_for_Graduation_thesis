"""
train.py
:Author:  JinZ
:Create:  2024/10/6 10:59
:github profile:    https://github.com/JzKosann
:gitlab profile:    https://gitlab.com/JzKosann
Copyright (c) 2019, JinZ Group All Rights Reserved.

File content:
    traning code for unet in GateDetector function for graduation thesis
"""

from torch.utils.data import random_split
from torch.utils.tensorboard import SummaryWriter
from model.unet_model import UNet
from utils.dataset import MyDataLoader
from torch import optim
import torch.nn as nn
import torch
import os
import torch.nn.functional as F
from tqdm import tqdm
from colorama import Fore, Style
import emoji
from utils.anima import anima, train_complete_printf


# 输出为tensorRT可推理的模型
from torch.autograd import Variable
import torch.onnx as torch_onnx
import onnxruntime
import torch.nn.init as init
import onnx

def dice_loss(pred, target):
    """
    Dice Loss function to compute similarity between the predicted and ground truth segmentation.
    :param pred: predicted tensor from the model (N, C, H, W)
    :param target: ground truth tensor (N, C, H, W) with class labels
    :param smooth: a small constant to avoid division by zero
    :return: dice loss value
    """
    smooth = 1e-6  # 平滑项（避免除以零
    pred = F.softmax(pred, dim=1)  # 转换成类别的概率分布
    pred_flat = pred.view(-1).float()
    target_flat = target.view(-1).float()
    intersection = (pred_flat * target_flat).sum()
    dice_score = (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
    return 1 - dice_score


def combined_loss(pred, target, alpha=0.5):
    """
    Combined loss function: weighted sum of CrossEntropyLoss and Dice Loss.
    :param pred: predicted tensor from the model (N, C, H, W)
    :param target: ground truth tensor (N, H, W) with class labels
    :param alpha: weight factor to balance CrossEntropy and Dice Loss
    :return: combined loss value
    """
    target_indexed = torch.argmax(target, dim=1)
    ce_loss = nn.CrossEntropyLoss()(pred, target_indexed)
    d_loss = dice_loss(pred, target)
    return alpha * ce_loss + (1 - alpha) * d_loss


def train_net(net, device, data_path, classesnum, checkpoint_path=None):
    """ 传入参数
    :param net:网络模型
    :param device: GPU还是CPU
    :param data_path: 训练集目录
    :param classesnum: 分类数量
    :return:

    """
    # ----------------------------#
    # 定义
    epochs = 500
    batch_size = 5
    lr = 0.003
    betas = (0.9, 0.999)
    weight_decay = 0
    accumulate_steps = 1
    _to_save_pth = False
    # 加载训练集和验证集
    dataset = MyDataLoader(data_path, num_classes=classesnum)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False)
    # 定义Adam算法 以及自学习率
    optimizer = optim.Adam(net.parameters(), lr=lr, betas=betas,
                           weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',
                                                     factor=0.9, patience=50, verbose=True)
    # best_val_loss统计，初始化为正无穷 最佳验证集损失
    best_val_loss = float('inf')
    best_train_loss = float('inf')
    best_loss_mux = float('inf')
    # 建立可视化
    visual_path = './runs'
    os.makedirs(visual_path, exist_ok=True)
    writer = SummaryWriter()

    start_epoch = 0
    # 如果提供了 checkpoint_path，则加载已有模型
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1  # 从下一个 epoch 继续
        best_train_loss = checkpoint['train_loss']
        best_val_loss = checkpoint['val_loss']
        print(f"Loaded checkpoint from {checkpoint_path}, starting from epoch {start_epoch}")

    # 训练epochs次
    for epoch in range(epochs):
        # 训练模式
        net.train()
        train_loss = 0.0
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader),
                            desc=f"Epoch {epoch + 1}/{epochs} 🚀")
        # 按照batch_size开始训练
        for batch_idx, (image, label) in progress_bar:
            optimizer.zero_grad()
            # 将数据拷贝到device中
            image = image.to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=torch.float32)  # 确保标签为long类型
            # 使用网络参数，输出预测结果
            pred = net(image)
            # 计算loss
            loss = combined_loss(pred, label, 0.5)
            loss.backward()
            if (batch_idx + 1) % accumulate_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            train_loss += loss.item()
            progress_bar.set_postfix({
                f"{Fore.GREEN}live_Loss": f"{loss.item():.4f}{Style.RESET_ALL}",
                f"{Fore.YELLOW}Best Train Loss": f"{best_train_loss:.4f}{Style.RESET_ALL}",
                f"{Fore.RED}Best Val Loss": f"{best_val_loss:.4f}{Style.RESET_ALL}",
                f"{Fore.BLUE}LR": f"{optimizer.param_groups[0]['lr']:.10f}{Style.RESET_ALL}",
                "Step": f"{batch_idx + 1}/{len(train_loader)}",
                f"{Fore.LIGHTCYAN_EX}Status": f"{emoji.emojize('🚀' if loss.item() < best_train_loss else '⌛')}{Style.RESET_ALL}"
            })
            # 保存loss值最小的网络参数
        avg_train_loss = train_loss / len(train_loader)
        if avg_train_loss < best_train_loss:
            best_train_loss = avg_train_loss
        # 评估模式（不计算梯度）
        net.eval()
        val_loss = 0.0
        with torch.no_grad():
            for image, label in val_loader:
                image = image.to(device=device, dtype=torch.float32)
                label = label.to(device=device, dtype=torch.float32)

                pred = net(image)
                loss = combined_loss(pred, label, 0.5)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss

        best_loss_mux = best_val_loss * best_train_loss

        if (avg_val_loss * avg_train_loss) < best_loss_mux:
            torch.save({
                'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': best_train_loss,
                'val_loss': best_val_loss,
            }, 'best_model.pth')
            _to_save_pth = False

        # 每个 epoch 结束后记录到 TensorBoard
        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        writer.add_scalar('Loss/val', avg_val_loss, epoch)
        writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], epoch)
        scheduler.step(avg_val_loss)
    writer.close()
    ending_anima = True
    if ending_anima:
        anima()
        train_complete_printf()

def train():
    # 选择设备，有cuda用cuda，没有就用cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 加载网络，图片单通道n_channel，分类为n_classes
    net = UNet(n_channels=1, n_classes=5)
    # 将网络拷贝到deivce中
    net.to(device=device)
    # 指定训练集地址，开始训练
    data_path = "dataSet/"
    train_net(net, device, data_path, classesnum=5, checkpoint_path='./best_model.pth')
    # train_net(net, device, data_path, classesnum=5)

def export_to_onnx(checkpoint_path, onnx_save_path):
    """
    将训练好的模型导出为 ONNX 格式。
    :param checkpoint_path: 训练模型权重的路径（如 'best_model.pth'）
    :param onnx_save_path: 导出的 ONNX 模型保存路径
    """
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 初始化模型
    net = UNet(n_channels=1, n_classes=5)  # 单通道输入，5类分类
    net.to(device)

    # 加载训练好的权重
    checkpoint = torch.load(checkpoint_path, map_location=device)
    net.load_state_dict(checkpoint['model_state_dict'])
    net.eval()  # 设置为评估模式

    # 准备假输入（输入形状：Batch=1, Channel=1, H=512, W=512）
    dummy_input = torch.randn(2, 1, 480, 640, device=device)

    # 定义动态维度
    dynamic_axes = {
        'input': {0: 'batch'},  # 输入的动态批次维度
        'output': {0: 'batch'}  # 输出的动态批次维度
    }

    # 导出 ONNX 模型
    torch.onnx.export(
        net, dummy_input, onnx_save_path,
        input_names=['input_img'],  # 输入名
        output_names=['output_img'],  # 输出名
        dynamic_axes=dynamic_axes,  # 动态批次支持
        opset_version=11  # 使用 ONNX opset 版本 11
    )

    print(f"模型已成功导出为 ONNX 格式，保存路径：{onnx_save_path}")

if __name__ == "__main__":
    # train()
    export_to_onnx('./best_model.pth', './UNet_model.onnx')
    # model = onnx.load('UNet_model')
    # print(onnx.helper.printable_graph(model.graph))
