from sklearn.decomposition import PCA


def pca():
    """
    主成分分析进行数据降维
    :return: None
    """
    X = [[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]]
    pa = PCA(n_components=0.95)
    data = pa.fit_transform(X)
    print(data)

    return None


if __name__ == '__main__':
    pca()
