import numpy as np
from sklearn.preprocessing import Imputer


def imputer():
    """
    缺失值处理
    :return:None
    """
    im = Imputer(missing_values="NaN", strategy="mean", axis=0)
    data = im.fit_transform([[1, 2], [np.nan, 3], [7, 6]])
    print(data)

    return None


if __name__ == '__main__':
    imputer()
