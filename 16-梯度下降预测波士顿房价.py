from sklearn.datasets import load_boston
from sklearn.linear_model import SGDRegressor, Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error


def myliner():
    """
    线性回归预测波士顿房价
    :return: None
    """
    # 加载数据
    lb = load_boston()

    # 分割数据到训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(lb.data, lb.target, test_size=0.25)

    # 进行标准化处理(特征值和目标值都标准化)
    std_x = StandardScaler()
    x_train = std_x.fit_transform(x_train)
    x_test = std_x.transform(x_test)
    # 目标值标准化
    std_y = StandardScaler()
    y_train = std_y.fit_transform(y_train.reshape(-1, 1))
    y_test = std_y.transform(y_test.reshape(-1, 1))

    # 估计器预测(梯度下降求解预测结果)
    sgd = SGDRegressor()
    sgd.fit(x_train, y_train)
    print(sgd.coef_)

    # 预测测试集房子的价格
    y_predict = std_y.inverse_transform(sgd.predict(x_test))
    print('测试集里每个房子的价格：', y_predict)

    print('梯度下降均方误差：', mean_squared_error(std_y.inverse_transform(y_test), y_predict))

    # 岭回归预测结果
    rd = Ridge(alpha=1.0)
    rd.fit(x_train, y_train)
    print(rd.coef_)

    # 预测测试集房子的价格
    y_rd_predict = std_y.inverse_transform(sgd.predict(x_test))
    print('测试集里每个房子的价格：', y_rd_predict)

    print('岭回归均方误差：', mean_squared_error(std_y.inverse_transform(y_test), y_rd_predict))

    # 加载模型
    return None


if __name__ == '__main__':
    myliner()
