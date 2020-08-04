from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

# 准备数据
lr = load_iris()

# 把数据集划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(lr.data, lr.target, test_size=0.25)

# 把训练集和测试集标准化
ss = StandardScaler()
x_train = ss.fit_transform(x_train)
x_test = ss.transform(x_test)

# 建立k近邻模型
knn = KNeighborsClassifier(n_neighbors=5)

# 输入训练数据进行学习模型
knn.fit(x_train, y_train)

# 对测试数据进行预测
y_predict = knn.predict(x_test)

print('预测结果：', y_predict)

# 模型评估
# 准确率
acc = knn.score(x_test, y_test)
print("准确率：", acc)
print("其他指标:\n", classification_report(y_test, y_predict, target_names=lr.target_names))


