import os
import cv2
import numpy as np
from scipy.io import savemat
import sys
sys.path.append('.')

#胶质瘤图片路径
glioma_path = "images/glioma"

#转移瘤图片路径
metastases_path = "images/metastases"

# 加载图片和标签
images = []
labels = []

for filename in os.listdir(glioma_path):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        # 读取图像
        image_path = os.path.join(glioma_path, filename)
        image = cv2.imread(image_path)

        # 调整图像大小为256 * 256
        resized_image = cv2.resize(image, (256, 256))

        # 添加到图像列表
        images.append(resized_image)

        # 添加标签 胶质瘤的标签是1
        label = 1
        labels.append(label)

for filename in os.listdir(metastases_path):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        # 读取图像
        image_path = os.path.join(metastases_path, filename)
        image = cv2.imread(image_path)

        # 调整图像大小为256 * 256
        resized_image = cv2.resize(image, (256, 256))

        # 添加到图像列表
        images.append(resized_image)

        # 添加标签 转移瘤的标签是2
        label = 2
        labels.append(label)

# 转换为NumPy数组
images = np.array(images)
labels = np.array(labels)

# 创建MAT文件数据结构
data = {'fea': images, 'gnd': labels}

# 保存为MAT文件
mat_file_path = "datasets/brain.mat"
savemat(mat_file_path, data)
