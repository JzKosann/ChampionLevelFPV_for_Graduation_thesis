import glob
# import cv2
from PIL import Image
from model import unet
from tqdm import tqdm

if __name__ == "__main__":
    net = unet.Unet()
    # 读取所有图片路径
    tests_path = glob.glob('dataSet/img_train/test/*.png')
    # 遍历素有图片
    for test_path in tqdm(tests_path, desc="🧐image_detecting...", total=len(tests_path)):
        # 保存结果地址
        save_res_path = test_path.split('.')[0] + '_res.png'
        # 读取图片
        img = Image.open(test_path)
        pred = net.detect(img)
        pred = Image.fromarray(pred.astype('uint8'))
        pred.save(save_res_path)
