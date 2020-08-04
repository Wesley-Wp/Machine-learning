from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV

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
    # 随机森林进行预测（超参数调优）
    rf = RandomForestClassifier()
    param = {'n_estimators': [120, 200, 300, 500, 800, 1200], 'max_depth': [5, 8, 15, 25, 30]}

    # 网格搜索与交叉验证
    gs = GridSearchCV(rf, param_grid=param, cv=2)

    gs.fit(x_train, y_train)

    print('准确率：', gs.score(x_test, y_test))
    print('查看选择的参数模型：', gs.best_params_)

    return None


if __name__ == '__main__':
    decision()
