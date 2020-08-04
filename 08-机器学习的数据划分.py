from sklearn.datasets import load_iris, fetch_20newsgroups, load_boston
from sklearn.model_selection import train_test_split


# 鸢尾花数据集
def iris_data():
    """
    数据集分割：训练集的特征值和目标值，测试集的特征值和目标值
    :return: None
    """

    li = load_iris()

    # print("获取特征值")
    # print(li.data)
    # print("获取目标值")
    # print(li.target)
    # print("描述")
    # print(li.DESCR)

    x_train, x_test, y_train, y_test = train_test_split(li.data, li.target, test_size=0.25)

    print("训练集的特征值：", x_train, "训练集的目标值：", y_train)
    print("测试集的特征值：", x_test, "测试集的目标值：", y_test)
    return None


# 新闻数据集（大数据集）
def new_data():
    news = fetch_20newsgroups(subset="all")
    print(news.data)
    print(news.target)


# 波士顿房价数据集(回归数据集）
def boston_data():
    lb = load_boston()

    print("获取特征值")
    print(lb.data)
    print("获取目标值")
    print(lb.target)
    print("描述")
    print(lb.DESCR)


if __name__ == '__main__':
    # iris_data()
    # new_data()
    boston_data()