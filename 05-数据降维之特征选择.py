from sklearn.feature_selection import VarianceThreshold


def var():
    """
    数据降维之特征选择--作用：删除低方差的特征
    :return:
    """
    x = [[0, 2, 0, 3], [0, 1, 4, 3], [0, 1, 1, 3]]
    selector = VarianceThreshold(threshold=1.0)
    data = selector.fit_transform(x)
    print(data)

    return None


if __name__ == '__main__':
    var()