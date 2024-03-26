import scipy.io
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
import joblib

# 读取.mat文件
mat_data = scipy.io.loadmat('datasets/detection.mat')

# 提取数据和标签
width = mat_data['width']
height = mat_data['height']
xl = mat_data['xl']
yl = mat_data['yl']
xr = mat_data['xr']
yr = mat_data['yr']
gnd = mat_data['gnd']

# 转换数据格式
X = np.column_stack((width, height, xl, yl, xr, yr))  # 将列数据合并为特征矩阵
y = np.squeeze(gnd) - 1  # 提取gnd并减去1作为输出标签

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建SVM分类器
clf = svm.SVC()

# 训练分类器
clf.fit(X_train, y_train)

# 在测试集上评估分类器
accuracy = clf.score(X_test, y_test)
print("Accuracy:", accuracy)

# 保存模型
joblib.dump(clf, 'svm_model.joblib')




