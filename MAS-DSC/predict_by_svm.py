import scipy.io
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler

def pre_pred_by_svm(model_path, dataset_path, threshold):
    # 加载模型
    clf = joblib.load(model_path)

    # 读取.mat文件
    mat_data = scipy.io.loadmat(dataset_path)

    # 提取数据
    width = mat_data['width']
    height = mat_data['height']
    xl = mat_data['xl']
    yl = mat_data['yl']
    xr = mat_data['xr']
    yr = mat_data['yr']

    # 转换数据格式
    X = np.column_stack((width, height, xl, yl, xr, yr))  # 将列数据合并为特征矩阵

    # 预测结果, 带小数
    decision_values = clf.decision_function(X)
    scaler = MinMaxScaler(feature_range=(0, 1))
    predictions = scaler.fit_transform(decision_values.reshape(-1, 1))

    # 打印推测结果
    predictions = np.squeeze(predictions)
    print("SVM predictions:")
    print(predictions)
    # 应用阈值判定
    predictions_processed = np.where(predictions >= threshold, 1, 0)
    print("Processed predictions:")
    print(predictions_processed)
    return predictions_processed

if __name__ == "__main__":
    y_pred = pred_by_svm('svm_model.joblib', 'datasets/detection.mat', 0.7)
    print('----------------')
    print(y_pred)


