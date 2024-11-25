import cnn_model
import torch
from torchvision import transforms
from PIL import Image
import cv2


# 加载模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = cnn_model.CNNModel().to(device)
model.load_state_dict(torch.load('model.ckpt'))  # 加载模型权重
model.eval()  # 切换为评估模式

# 定义图像预处理函数
transform = transforms.Compose([
    transforms.Resize((28, 28)),  # 与训练时的输入尺寸一致
    transforms.ToTensor()  # 转换为Tensor
])

# 加载图片并进行预处理
def load_image(image_path):
    image = Image.open(image_path).convert("RGB")  # 打开图片并转换为RGB
    image = transform(image)  # 应用预处理
    image = image.unsqueeze(0)  # 增加batch维度 (1, 3, 224, 224)
    return image.to(device)

# 预测坐标
def predict_coordinates(image_path):
    image = load_image(image_path)  # 预处理图片
    with torch.no_grad():  # 在评估模式下不计算梯度
        outputs = model(image)
    return outputs.cpu().numpy().reshape(-1, 2)  # 返回4个坐标点

# 在图片上用OpenCV画点
def draw_points_on_image(image_path, coordinates):
    # 使用 OpenCV 读取原始图片
    image = cv2.imread(image_path)
    # 将224x224的坐标点映射回原图像尺寸
    h, w, _ = image.shape
    coordinates = coordinates * [w / 28, h / 28]  # 根据图片的原始尺寸缩放坐标

    # 画出4个点
    for (x, y) in coordinates:
        cv2.circle(image, (int(x), int(y)), radius=1, color=(0, 255, 0), thickness=-1)  # 绿色圆点

    # 显示图像
    cv2.imshow("Predicted Points", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 主程序
if __name__ == '__main__':
    # 图片路径
    image_path = './dataSet/test/115.png'  # 替换为你的图片路径

    # 预测四个坐标点
    predicted_coordinates = predict_coordinates(image_path)
    print(f"Predicted coordinates: {predicted_coordinates}")

    # 在图片上画出预测的四个坐标点
    draw_points_on_image(image_path, predicted_coordinates)