import torch
import numpy as np
from utils.utils import show_config
from .unet_model import UNet
import torch.nn as nn
import torchvision.transforms as tf
from PIL import Image
import torch.nn.functional as F


class Unet(object):
    _defaults = {
        "model_path": "./best_model.pth",
        "num_classes": 5,
        "input_shape": [480, 640],
        "cuda": True,
    }

    def __init__(self, **kwargs):
        self.__dict__.update(Unet._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)

        # {'_background_': 0, 'gate_below': 1, 'gate_left': 2, 'gate_right': 3, 'gate_up': 4}
        self.colors = [(0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128)]
        self.generate()
        show_config(**self._defaults)

    def generate(self):
        self.net = UNet(n_channels=1, n_classes=self.num_classes)
        # choose device = 'cuda'
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # 加载模型权重
        checkpoint = torch.load(self.model_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            # 如果 checkpoint 包含训练状态，提取模型参数
            self.net.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            # 否则直接加载整个 state_dict
            self.net.load_state_dict(checkpoint, strict=False)

        # self.net.load_state_dict(torch.load(self.model_path, map_location=self.device, weights_only=True))
        self.net = self.net.eval()
        print('{} model, and classes loaded.'.format(self.model_path))
        if self.cuda:
            self.net = nn.DataParallel(self.net)
            self.net = self.net.cuda()

    # --------------------------#
    # 输入img为灰度图像：
    # 传入image
    # --------------------------#
    def detect(self, img1, img2):
        # 输入图像为shape为（height*width）图像
        # image_matrix = np.array(img1)
        # image_matrix = image_matrix.reshape(image.height, image.width, 1)
        # img_tensor = (tf.ToTensor()(image_matrix).unsqueeze(0)
        #               .to(device=self.device, dtype=torch.float32))
        img1_tensor = tf.ToTensor()(img1).unsqueeze(0)
        img2_tensor = tf.ToTensor()(img2).unsqueeze(0)
        img_tensor = torch.cat((img1_tensor, img2_tensor), 0).to('cuda')
        pred = self.net(img_tensor).data.cpu()
        pred = F.softmax(pred, dim=1)
        # pred = torch.argmax(pred, dim=1).squeeze().numpy()
        pred = torch.argmax(pred, dim=1)
        preds = []
        for i in range(2):
            pred_img = pred[i].squeeze().numpy().astype(np.uint8)
            preds.append(pred_img)
        # pred = Image.fromarray(pred.astype('uint8'))
        # pred = self.draw_color(pred)
        # pred.show(title="pred")
        # 用OPENCV来展示视频帧
        # cv2.imshow("pred", pred)
        return pred

    # --------------------------#
    # 绘制标签颜色
    # 传入image --> 转化为RGB格式 --> 检测标签值 --> 索引颜色并赋值 --> 生成图像
    # @
    # --------------------------#
    def draw_color(self, image):
        # 用 numpy 矩阵存储输入图像
        img_array = np.array(image)
        chosen_labels = np.unique(image)
        chosen_colors = [self.colors[label] for label in chosen_labels]
        # 新建一个RGB （height,width,channel=3）
        color_mask = np.zeros((image.height, image.width, 3), dtype=np.uint8)
        for i, label in enumerate(chosen_labels):
            if label != 0:
                idx = np.nonzero(img_array == label)
                color_mask[idx[0], idx[1]] = chosen_colors[i]
        return color_mask

    # --------------------------#
    # 获取最大区域，计算质点
    # 传入img -> 颜色分离 -> 获取图像最大轮廓 -> 获取图像轮廓中点（四个）
    # @return : corners
    # --------------------------#
