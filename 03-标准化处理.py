from sklearn.preprocessing import StandardScaler


def stand():
    """
    标准化缩放
    :return:
    """
    stds = StandardScaler()
    data = stds.fit_transform([[1., -1., 3.], [2., 4., 2.], [4., 6., -1]])
    print(data)

    return None


if __name__ == '__main__':
    stand()