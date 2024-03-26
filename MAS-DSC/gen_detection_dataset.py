import json
import os
import numpy as np
from PIL import Image
from scipy.io import savemat

# 定义文件夹路径
annotation_folder = 'images/detection'

# 初始化特征列表和标签列表
features = []
labels = []
widths = []
heights = []
xls = []
yls = []
xrs = []
yrs = []

# 遍历annotation文件夹中的所有json文件
for json_filename in os.listdir(annotation_folder):
    if json_filename.endswith('.json'):
        json_path = os.path.join(annotation_folder, json_filename)
        image_path = os.path.join(annotation_folder, json_filename.replace('.json', '.png'))  # 假设图像文件与JSON文件同名

        # 读取JSON文件
        with open(json_path, 'r') as json_file:
            data = json.load(json_file)

        # 假设每个JSON文件只有一个矩形标注
        shape = data['shapes'][0]  # 获取第一个标注
        points = shape['points']  # 矩形框的坐标点
        xl, yl, xr, yr = points[0][0], points[0][1], points[1][0], points[1][1]

        xls.append(round(xl))
        yls.append(round(yl))
        xrs.append(round(xr))
        yrs.append(round(yr))

        print(json_path, image_path)
        # 加载图像
        image = Image.open(image_path)
        width, height = image.size

        widths.append(width)
        heights.append(height)

        # 裁剪图像
        target_box = image.crop((xl, yl, xr, yr))

        # Resize图像到64x64
        resized_target_box = target_box.resize((64, 64), Image.ANTIALIAS)

        # 将图像转换为numpy数组并展平
        image_array = np.array(resized_target_box).flatten()

        # 添加到特征列表
        features.append(image_array)

        # 添加标签到标签列表
        label = shape['label']
        labels.append(int(label))

# 转换列表为numpy数组
features = np.array(features)
print(len(features))
labels = np.array(labels).reshape(-1, 1)
print(len(labels))
widths = np.array(widths).reshape(-1, 1)
heights = np.array(heights).reshape(-1, 1)
xls = np.array(xls).reshape(-1, 1)
xrs = np.array(xrs).reshape(-1, 1)
yls = np.array(yls).reshape(-1, 1)
yrs = np.array(yrs).reshape(-1, 1)

# 创建MATLAB兼容的结构
mat_data = {
    'fea': features,
    'gnd': labels,
    'width': widths,
    'height': heights,
    'xl': xls,
    'yl': yls,
    'xr': xrs,
    'yr': yrs
}

# 保存为MAT文件
savemat('detection.mat', mat_data)
