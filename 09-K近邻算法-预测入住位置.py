from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd


def knnels():
    """
    k-近邻预测用户签到位置
    :return:None
    """
    # 读取数据
    data = pd.read_csv("./datafile/FBlocation/train.csv")
    print(data.head(10))
    # 处理数据
    # 1、筛选数据
    data = data.query("x > 1.0 & x < 1.25 & y > 2.5 y < 2.75")

    # 处理时间数据
    time_value = pd.to_datetime(data['time'], unit="s")
    print(time_value)

    # 把时间格式转换成字典格式
    time_value = pd.DatetimeIndex(time_value)

    # 构建一些特征
    data['day'] = time_value.day
    data['hour'] = time_value.hour
    data['weekday'] = time_value.weekday

    # 删除时间戳特征
    data = data.drop(['time'], axis=1)  # 这里1表示列，sklearn里面0表示列

    print(data)

    # 把签到数量少于n个目标位置删除
    place_count = data.groupby('place_id').count()
    tf = place_count[place_count.row_id > 3].reset_index()  # 索引从0开始
    data = data[data['place_id'].isin(tf.place_id)]

    # 取出数据中的特征值和目标值
    x = data.drop(['place_id'], axis=1)
    y = data['place_id']

    # 分割数据集（训练集和测试集）
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

    # 特征工程（标准化）
    stand = StandardScaler()

    # 对测试集和训练集的特征值进行标准化处理
    x_train = stand.fit_transform(x_train)

    x_test = stand.transform(x_test)

    # 算法流程
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(x_train, y_train)

    # 得出预测结果
    y_predict = knn.predict(x_test)
    print("预测的目标签到位置为：", y_predict)

    # 准确度
    sc = knn.score(x_test, y_test)
    print("预测的准确率：", sc)

    return None
