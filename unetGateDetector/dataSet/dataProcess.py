import os
import shutil

import cv2
import numpy as np
from PIL import Image
import time
from tqdm import tqdm

"""
数据清洗：
    1、统计数据信息
    2、去除已损坏图片
    3、去除模糊图像 https://www.cnblogs.com/greentomlee/p/9379471.html
    4、去除相似图像

"""


class CDataProcess:
    def __init__(self):
        self.dir = None
        self.size = 0
        self.number = 0
        self.bad_number = 0

    # 获取数据集存储大小、图片数量、破损图片数量
    def get_data_info(self):
        print("正在获取图片数据信息...")

        # 获取所有文件
        img_files = []
        for root, dirs, files in os.walk(self.dir):
            img_files.extend(os.path.join(root, file) for file in files if os.path.isfile(os.path.join(root, file)))

        # 使用 tqdm 显示进度条
        for file_path in tqdm(img_files, desc="Getting images info", total=len(img_files)):
            try:
                file_size = os.path.getsize(file_path)
                self.size += file_size
                self.number += 1

                img = Image.open(file_path)
                img.load()
            except (OSError, Exception) as e:
                self.bad_number += 1
                print(f"损坏的图片: {file_path} - {e}")

        print(f"数据集存储大小: {self.size / 1024 / 1024:.2f} MB")
        print(f"图片数量: {self.number}")
        print(f"损坏图片数量: {self.bad_number}")
        return self.size / 1024 / 1024, self.number, self.bad_number

    def move_with_retry(self, src, dst, retries=5, delay=1):
        for _ in range(retries):
            try:
                shutil.move(src, dst)
                return True
            except Exception as e:
                print(f"移动失败: {e}，正在重试...")
                time.sleep(delay)
        return False

    def filter_bad(self):
        filter_dir = "./dataSet/img_train/bad_img"
        os.makedirs(filter_dir, exist_ok=True)
        filter_count = 0
        # file_count = 0
        img_files = []
        for root, dirs, files in os.walk(self.dir):
            img_files.extend(os.path.join(root, file) for file in files if os.path.isfile(os.path.join(root, file)))

        for root, _, files in os.walk(self.dir):
            for file in tqdm(files, desc="Processing bad images", total=len(img_files)):
                file_path = os.path.join(root, file)
                # file_count += 1
                if os.path.isfile(file_path):
                    try:
                        Image.open(file_path).verify()  # 使用verify()进行快速检查
                    except (OSError, Image.DecompressionBombError) as e:
                        print(f"损坏的图像: {file_path} - {e}")
                        # 处理移动时的文件名冲突
                        dest_path = filter_dir
                        # dest_path = os.path.join(filter_dir, file)
                        # if os.path.exists(dest_path):
                        #     base, ext = os.path.splitext(file)
                        #     dest_path = os.path.join(filter_dir, f"{base}_{int(time.time())}{ext}")
                        # if self.move_with_retry(file_path, dest_path):
                        #     filter_count += 1
                        try:
                            os.remove(path=file_path)
                        except Exception as e:
                            print(f"删除失败: {e},已跳过！")
                            pass
        print(f" {filter_count} bad img been moved")
        return filter_count

    # 去除模糊图片
    def filter_blurred(self):
        filter_dir = "./dataSet/img_train/blurred"
        if not os.path.exists(filter_dir):
            os.mkdir(filter_dir)
        filter_number = 0
        img2_files = []
        for root, dirs, files in os.walk(self.dir):
            img2_files.extend(os.path.join(root, file) for file in files if os.path.isfile(os.path.join(root, file)))

        for root, dirs, files in os.walk(self.dir):
            img_files = [file_name for file_name in files if os.path.isfile(os.path.join(root, file_name))]
            for file in tqdm(img_files, desc="Processing blurred images", total=len(img2_files)):
                file_path = os.path.join(root, file)
                # img = cv2.imread(file_path)
                img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), -1)
                image_var = cv2.Laplacian(img, cv2.CV_64F).var()
                # print(image_var)  # 打印拉普拉斯算子计算出来的方差 值越大越清晰
                if image_var < 100:
                    # shutil.move(file_path, filter_dir)
                    try:
                        os.remove(path=file_path)
                    except Exception as e:
                        print(f"删除失败: {e},已跳过！")
                        pass
                    filter_number += 1
        print(f"{filter_number}blurred img been precessed")
        return filter_number

    # 计算两张图片的相似度
    def calc_similarity(self,
                        img1_path=None,
                        img2_path=None
                        ) -> bool:
        # img1 = cv2.imdecode(np.fromfile(img1_path, dtype=np.uint8), -1)
        img1 = cv2.imread(img1_path, 0)
        img2 = cv2.imread(img2_path, 0)
        H1 = cv2.calcHist([img1], [0], None, [256], [0, 256])  # 计算图直方图
        H1 = cv2.normalize(H1, H1, 0, 1, cv2.NORM_MINMAX, -1)  # 对图片进行归一化处理
        # img2 = cv2.imdecode(np.fromfile(img2_path, dtype=np.uint8), -1)
        H2 = cv2.calcHist([img2], [0], None, [256], [0, 256])  # 计算图直方图
        H2 = cv2.normalize(H2, H2, 0, 1, cv2.NORM_MINMAX, -1)  # 对图片进行归一化处理

        similarity = cv2.compareHist(H1, H2, 0)  # 相似度比较
        # print('similarity:', similarity1)
        if similarity > 0.80:  # 0.98是阈值，可根据需求调整
            return True
        else:
            return False

    #
    # # 去除相似度高的图片
    def filter_similar(self):
        filter_dir = "./dataSet/img_train/similar/"
        if not os.path.exists(filter_dir):
            os.mkdir(filter_dir)
        filter_number = 0
        for root, dirs, files in os.walk(self.dir):
            img_files = [file_name for file_name in files if os.path.isfile(os.path.join(root, file_name))]
            filter_list = []  # 重复的img
            for index in tqdm(range(len(img_files) - 4), desc="Processing similar images", total=len(img_files)):
                # for index in range(len(img_files))[:-4]:

                if img_files[index] in filter_list:
                    continue
                for idx in range(len(img_files))[(index + 1):(index + 5)]:
                    img1_path = os.path.join(root, img_files[index])
                    img2_path = os.path.join(root, img_files[idx])
                    if self.calc_similarity(img1_path=img1_path, img2_path=img2_path):
                        filter_list.append(img_files[idx])
                        filter_number += 1
            for item in tqdm(filter_list, desc="moving similar images", total=filter_number):
                src_path = os.path.join(root, item)
                # self.move_with_retry(src=src_path, dst=filter_dir)
                try:
                    # shutil.move(src_path, filter_dir)
                    os.remove(path=src_path)
                except Exception as e:
                    print(f"删除失败: {e},已跳过！")
                    pass
        return filter_number

    def run(self, dir_path=None):
        self.dir = dir_path  # 文件路径
        print(f"is doing {self.dir}")
        # self.get_data_info()
        # self.filter_bad()
        # self.filter_blurred()
        self.filter_similar()
