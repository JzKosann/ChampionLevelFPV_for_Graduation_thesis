import numpy as np
import torch
import cv2
import os
import glob

from sympy.stats.sampling.sample_numpy import numpy
from torch.utils.data import Dataset
import random
from PIL import Image


class MyDataLoader(Dataset):
    def __init__(self, data_path, num_classes):
        # 初始化函数，读取所有data_path下的图片
        self.data_path = data_path
        self.imgs_path = glob.glob(os.path.join(data_path, 'img_train/imgs/*.png'))
        # self.label_path = glob.glob(os.path.join(data_path, 'masks/train/*.jpg'))
        self.num_classes = num_classes

    def augment(self, image, flipCode):
        # 使用cv2.flip进行数据增强，filpCode为1水平翻转，0垂直翻转，-1水平+垂直翻转
        flip = cv2.flip(image, flipCode)
        return flip

    def __getitem__(self, index):
        # 根据index读取图片
        image_path = self.imgs_path[index]
        # 根据image_path生成label_path
        label_path = image_path.replace('imgs', 'labels')
        # 读取训练图片和标签图片
        image_png = Image.open(image_path)
        label_png = Image.open(label_path)

        image_matrix = np.array(image_png)
        """
        这里将label转换为ONE-HOT格式 通道数为类别数
        """
        label_matrix = np.array(label_png)
        num_classes = self.num_classes
        label_matrix[label_matrix >= num_classes] = num_classes
        seg_labels = np.eye(num_classes, dtype=np.uint8)[label_matrix.reshape(-1)]
        label_png = seg_labels.reshape(label_png.height, label_png.width, num_classes)

        image_png = image_matrix.reshape(image_png.height, image_png.width, 1)

        label_png = torch.tensor(label_png).float().permute(2, 0, 1)
        image_png = torch.tensor(image_png).float().permute(2, 0, 1)

        return image_png, label_png

    def __len__(self):
        # 返回训练集大小
        return len(self.imgs_path)


if __name__ == "__main__":
    num_classes = 5  # 类别数量
    pic = cv2.imread(
        "E:/project/SCU-TuHaiyan-2024-Drone_RL/template/unetGateDetector/dataSet/img_train/labels/img0_20240924_161703_class_2.png")
    my_dataset = MyDataLoader("../dataSet/", num_classes)
    print("数据个数：", len(my_dataset))

    train_loader = torch.utils.data.DataLoader(dataset=my_dataset, batch_size=2, shuffle=True)
    for image, label in train_loader:
        print(image.shape, label.shape)
