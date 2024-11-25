import cv2
import os
import pandas as pd

# 全局变量存储特征点坐标
points = []


# 鼠标点击事件回调函数，存储点击的特征点坐标
def on_mouse_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) < 4:
            points.append((x, y))
            print(f"特征点 {len(points)}: ({x}, {y})")


# 加载和可视化图片
def load_and_visualize_image(image_path):
    global points
    points = []

    img = cv2.imread(image_path)
    if img is None:
        print(f"无法加载图像: {image_path}")
        return None

    cv2.imshow('Image', img)
    cv2.setMouseCallback('Image', on_mouse_click)   # 回调到 on_mouse_click 函数

    # 等待用户标记完四个特征点
    while len(points) < 4:
        cv2.waitKey(1)

    cv2.destroyAllWindows()
    return points


# 保存标签到CSV
def save_points_to_csv(image_name, points, csv_file):
    lu, ro, ld, rd = points
    df = pd.DataFrame({
        'img_name': [image_name],
        'lu': [lu],
        'ro': [ro],
        'ld': [ld],
        'rd': [rd]
    })
    if not os.path.exists(csv_file) or os.stat(csv_file).st_size == 0:
        df.to_csv(csv_file, mode='a', header=True, index=False)  # 文件不存在或为空时，写入表头
    else:
        df.to_csv(csv_file, mode='a', header=False, index=False)  # 文件存在且不为空时，追加数据
    print(f"特征点已保存到 {csv_file}")


# 获取已经标注的图片列表
def get_annotated_images(csv_file):
    if os.path.exists(csv_file) and os.stat(csv_file).st_size > 0:
        df = pd.read_csv(csv_file)
        return set(df['img_name'].tolist())
    return set()


# 主程序，处理文件夹中的图片
def annotate_images_in_folder(folder_path, csv_file):
    # 获取已经标注的图片
    annotated_images = get_annotated_images(csv_file)

    # 遍历文件夹中的所有图片文件
    for image_name in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image_name)

        # 检查文件是否是图片，并且是否已经标注过
        if image_name.lower().endswith(('.png', '.jpg', '.jpeg')) and image_name not in annotated_images:
            print(f"正在标注: {image_name}")
            points = load_and_visualize_image(image_path)
            if points:
                save_points_to_csv(image_name, points, csv_file)
            else:
                print(f"跳过图像 {image_name}，因为未能标记足够的特征点。")
        else:
            print(f"跳过已标注的图片: {image_name}")


if __name__ == '__main__':
    folder_path = 'dataSet/images'  # 图片文件夹路径
    csv_file = './dataSet/img_labels.csv'  # 输出CSV文件名

    # 开始处理文件夹中的图片
    annotate_images_in_folder(folder_path, csv_file)
