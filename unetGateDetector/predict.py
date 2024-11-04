import glob

import cv2
import numpy as np
# import cv2
from PIL import Image
# from lxml.saxparser import prefix

from model import unet
from tqdm import tqdm

if __name__ == "__main__":
    net = unet.Unet()
    # # 读取所有图片路径
    # tests_path = glob.glob('dataSet/img_train/test/*.png')
    #
    # # 遍历素有图片
    # for test_path in tqdm(tests_path, desc="🧐image_detecting...", total=len(tests_path)):
    #     # 保存结果地址
    #     save_res_path = test_path.split('.')[0] + '_res.png'
    #     # 读取图片
    #     img = Image.open(test_path)
    #     pred = net.detect(img)
    #     pred = Image.fromarray(pred.astype('uint8'))
    #     pred.save(save_res_path)
    path1 = "./dataSet/img_train/test/img0_20240924_161725.png"
    path2 = "./dataSet/img_train/test/img0_20240924_161728.png"
    img1 = cv2.imread(path1, 0)
    img2 = cv2.imread(path2, 0)
    preds = net.detect(img1,img2)
    cv2.imshow("1",preds[0])
    cv2.imshow("2",preds[1])
    cv2.imwrite("test1.png",preds[0])
    cv2.imwrite("test2.png",preds[1])
    cv2.waitKey(0)
