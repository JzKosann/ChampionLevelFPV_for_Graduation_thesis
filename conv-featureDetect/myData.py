import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np


# 定义函数解析特征点
def parse_point(point_str):
    # 假设格式为 (x, y)
    point_str = point_str.strip('()')
    x, y = point_str.split(', ')
    return int(x), int(y)


# 自定义 PyTorch 数据集类
class ImageDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        # 加载 CSV 数据
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 获取图像路径
        img_name = self.data.iloc[idx]['img_name']
        img_path = f"{self.img_dir}/{img_name}"
        image = Image.open(img_path).convert("RGB")

        # 解析特征点
        lu_point = parse_point(self.data.iloc[idx]['lu'])
        ro_point = parse_point(self.data.iloc[idx]['ro'])
        ld_point = parse_point(self.data.iloc[idx]['ld'])
        rd_point = parse_point(self.data.iloc[idx]['rd'])

        # 将特征点转换为张量
        labels = torch.tensor([*lu_point, *ro_point, *ld_point, *rd_point], dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, labels

if __name__ == '__main__':
    csv_file = './dataSet/img_labels.csv'  # 你的CSV文件
    img_dir = './dataSet/images/'  # 图像文件夹路径

    # 定义你想要的图像变换，适用于训练时的数据增强
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.Resize((28, 28)),  # 调整图像大小
        transforms.ToTensor(),  # 转换为张量
    ])

    # 创建数据集
    dataset = ImageDataset(csv_file=csv_file, img_dir=img_dir, transform=transform)

    # 创建 DataLoader
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    a = np.asarray(Image.open(r"./dataSet/images/1.png"))
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    b = transform(a)
    print(b.data)
    # 检查数据
    for images, labels in dataloader:
        print(f"Images batch shape: {images.shape}")
        print(f"Labels batch shape: {labels.shape}")
        break
