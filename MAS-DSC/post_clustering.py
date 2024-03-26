import numpy as np
from sklearn import cluster
from scipy.sparse.linalg import svds
from sklearn.preprocessing import normalize
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, adjusted_mutual_info_score

nmi = normalized_mutual_info_score
ami = adjusted_mutual_info_score
ari = adjusted_rand_score


# acc(y_true, y_pred) 函数：
# 功能：计算聚类准确率。
# 参数：y_true 是真实的类别标签，y_pred 是预测的类别标签。
# 返回：聚类准确率，范围在 [0,1] 之间。

# err_rate(gt_s, s) 函数：
# 功能：计算错误率。
# 参数：gt_s 是真实的类别标签，s 是预测的类别标签。
# 返回：错误率。

# thrC(C, alpha) 函数：
# 功能：对系数矩阵进行阈值处理。
# 参数：C 是系数矩阵，alpha 是阈值参数。
# 返回：经过阈值处理后的系数矩阵Cp。

# post_proC(C, K, d, ro) 函数：
# 功能：对系数矩阵进行进一步的处理。
# 参数：C 是系数矩阵，K 是聚类数目，d 是子空间的维度，ro 是参数。
# 返回：处理后的聚类结果和矩阵grp, L。
#
# spectral_clustering(C, K, d, alpha, ro) 函数：
# 功能：执行谱聚类。先执行thrC()再执行post_proC()
# 参数：C 是系数矩阵，K 是聚类数目，d 是子空间的维度，alpha 是阈值参数，ro 是参数。
# 返回：谱聚类的结果。
def acc(y_true, y_pred):
    """
    Calculate clustering accuracy.
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    # from sklearn.utils.linear_assignment_ import linear_assignment
    from scipy.optimize import linear_sum_assignment as linear_assignment
    ind_row, ind_col = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in zip(ind_row, ind_col)]) * 1.0 / y_pred.size


def err_rate(gt_s, s):
    return 1.0 - acc(gt_s, s)


def thrC(C, alpha):
    if alpha < 1:
        N = C.shape[1]
        Cp = np.zeros((N, N))
        S = np.abs(np.sort(-np.abs(C), axis=0))
        Ind = np.argsort(-np.abs(C), axis=0)
        for i in range(N):
            cL1 = np.sum(S[:, i]).astype(float)
            stop = False
            csum = 0
            t = 0
            while (stop == False):
                csum = csum + S[t, i]
                if csum > alpha * cL1:
                    stop = True
                    Cp[Ind[0:t + 1, i], i] = C[Ind[0:t + 1, i], i]
                t = t + 1
    else:
        Cp = C

    return Cp


def post_proC(C, K, d, ro):
    # C: coefficient matrix, K: number of clusters, d: dimension of each subspace
    n = C.shape[0]
    C = 0.5 * (C + C.T)
    # C = C - np.diag(np.diag(C)) + np.eye(n, n)  # good for coil20, bad for orl
    r = d * K + 1
    U, S, _ = svds(C, r, v0=np.ones(n))
    U = U[:, ::-1]
    S = np.sqrt(S[::-1])
    S = np.diag(S)
    U = U.dot(S)
    U = normalize(U, norm='l2', axis=1)
    Z = U.dot(U.T)
    Z = Z * (Z > 0)
    L = np.abs(Z ** ro)
    L = L / L.max()
    L = 0.5 * (L + L.T)
    spectral = cluster.SpectralClustering(n_clusters=K, eigen_solver='arpack', affinity='precomputed',
                                          assign_labels='discretize')

    spectral.fit(L)
    grp = spectral.fit_predict(L)
    return grp, L


def spectral_clustering(C, K, d, alpha, ro):
    C = thrC(C, alpha)
    y, _ = post_proC(C, K, d, ro)

    return y
