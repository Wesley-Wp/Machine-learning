from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split

import pandas as pd


def decision():
    # 读取数据
    data = pd.read_csv("./datafile/titanic.txt")

    # 处理数据，找出特征值和目标值
    x = data[['pclass', 'age', 'sex']]

    y = data['survived']

    # 缺失值处理
    x['age'].fillna(x['age'].mean(), inplace=True)

    # 分割数据（把测试数据分割成训练集和测试集）
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

    # 进行数据处理（特征，类别进行onehot编码）
    dicv = DictVectorizer(sparse=False)
    x_train = dicv.fit_transform(x_train.to_dict(orient='record'))
    x_test = dicv.transform(x_test.to_dict(orient='record'))

    print(dicv.get_feature_names())
    # 使用决策树预测
    dec = DecisionTreeClassifier(max_depth=5)
    dec.fit(x_train, y_train)

    # 预测准确率
    print('预测的准确率：', dec.score(x_test, y_test))

    # 导出树的结构
    export_graphviz(dec, out_file='./datafile/tree.dot',
                    feature_names=['age', 'pclass=1st', 'pclass=2nd', 'pclass=3rd', 'sex=female', 'sex=male'])
    return None


if __name__ == '__main__':
    decision()
