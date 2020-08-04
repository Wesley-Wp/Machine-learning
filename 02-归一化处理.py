from sklearn.preprocessing import MinMaxScaler


def MaxMin():
    """
    归一化处理
    :return: None
    """
    mm = MinMaxScaler()
    data = mm.fit_transform([[90, 2, 10, 40], [60, 4, 15, 45], [75, 3, 13, 46]])
    print(data)

    return None


if __name__ == '__main__':
    MaxMin()
