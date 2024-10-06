import argparse
import base64
import json
import os
import os.path as osp
from pkgutil import get_loader

import cv2
import imgviz
import PIL.Image
import numpy as np

from labelme.logger import logger
from labelme import utils
from pygments.formatters import img
# from scipy.constants import point
# from scipy.special import points


def to_one_hot(lbl, num_classes):
    one_hot = np.zeros((lbl.shape[0], lbl.shape[1], num_classes), dtype=np.uint8)
    for c in range(num_classes):
        one_hot[..., c] = (lbl == c).astype(np.uint8)
    return one_hot


def main():
    logger.warning(
        "This script is aimed to demonstrate how to convert the "
        "JSON file to a single image dataset."
    )
    logger.warning(
        "It won't handle multiple JSON files to generate a "
        "real-use dataset."
    )

    # json_file是标注完之后生成的json文件的目录。out_dir是输出目录，即数据处理完之后文件保存的路径
    # json_file = r"D:\learning_Code\python\pytorch\unet-pytorch-main\myself_datas\01_cat_dog\before"
    json_file = "../img_train/labelimg"
    # out_dir1 = r"D:\learning_Code\python\pytorch\unet-pytorch-main\myself_datas\01_cat_dog\SegmentationClass"
    out_dir1 = "../img_train/labels"

    # 如果输出的路径不存在，则自动创建这个路径
    if not osp.exists(out_dir1):
        os.mkdir(out_dir1)
        # 将类别名称转换成数值，以便于计算
    label_name_to_value = {"_background_": 0}
    for file_name in os.listdir(json_file):
        # 遍历json_file里面所有的文件，并判断这个文件是不是以.json结尾
        if file_name.endswith(".json"):
            path = os.path.join(json_file, file_name)
            if os.path.isfile(path):
                data = json.load(open(path))

                # 获取json里面的图片数据，也就是二进制数据
                imageData = data.get("imageData")
                # 如果通过data.get获取到的数据为空，就重新读取图片数据
                if not imageData:
                    imagePath = os.path.join(json_file, data["imagePath"])
                    with open(imagePath, "rb") as f:
                        imageData = f.read()
                        imageData = base64.b64encode(imageData).decode("utf-8")
                #  将二进制数据转变成numpy格式的数据
                img = utils.img_b64_to_arr(imageData)

                for shape in sorted(data["shapes"], key=lambda x: x["label"]):
                    label_name = shape["label"]
                    if label_name in label_name_to_value:
                        label_value = label_name_to_value[label_name]
                    else:
                        label_value = len(label_name_to_value)
                        label_name_to_value[label_name] = label_value
                lbl, _ = utils.shapes_to_label(img.shape, data["shapes"], label_name_to_value)

                label_names = [None] * (max(label_name_to_value.values()) + 1)
                for name, value in label_name_to_value.items():
                    label_names[value] = name

                lbl_viz = imgviz.label2rgb(
                    label=lbl, image=imgviz.asgray(img), label_names=label_names, loc="rb"
                )
                utils.lblsave(osp.join(out_dir1, "%s.png" % file_name.split(".")[0]), lbl)

                logger.info("Saved to: {}".format(out_dir1))

    # 将字典 label_name_to_value 写入val.txt文件
    js = json.dumps(label_name_to_value)
    file = open('val.txt', 'w')
    file.write(js)
    file.close()
    print("label:", label_name_to_value)

pic = cv2.imread("./test/img0_20240924_163714_856_res.png")
def mouse_handler(event, x, y, flags, param):
    global pic
    imgCopy = pic.copy()
    if event == cv2.EVENT_MOUSEMOVE:
        font = cv2.FONT_HERSHEY_PLAIN
        message = '{}'.format(pic[y, x])
        cv2.putText(imgCopy, message, (50, 50), cv2.FONT_HERSHEY_PLAIN, 2, (125, 125, 125))
        cv2.imshow("win", imgCopy)
    # elif event == cv2.EVENT_MOUSEMOVE:
    #     cv2.imshow("win", pic)


if __name__ == "__main__":

    # cv2.namedWindow("win", cv2.WINDOW_AUTOSIZE)
    # cv2.setMouseCallback("win", mouse_handler)
    #
    # cv2.imshow("win", pic)
    # cv2.waitKey()
    main()
    file = open('val.txt', 'r')
    js = file.read()
    dic = json.loads(js)
    file.close()
    print(dic)

