# -*- coding:utf-8 -*-
"""
# @Time  :2020/8/2 22:34
#@Author :Wesley
#@File   :18-决策树预测房价.py
#@IDE    :PyCharm
#@Email  :984@qq.com
"""
from sklearn.datasets.california_housing import fetch_california_housing
from sklearn import tree
from sklearn.model_selection import GridSearchCV
import pydotplus
from IPython.display import Image

housing = fetch_california_housing()
# print(housing.data.shape)
dtr = tree.DecisionTreeRegressor(max_depth=2)
dtr.fit(housing.data[:,[6,7]],housing.target)

dot_data = tree.export_graphviz(dtr,
                                out_file="./datafile/trees.dot",
                                feature_names=housing.feature_names[6:8],
                                filled=True,
                                impurity=False,
                                rounded=True)
graph = pydotplus.graph_from_dot_data(dot_data)
graph.get_nodes()[7].set_fillcolor('#FFF2DD')
Image(graph.create_png())
graph.write_png("dtr_white_background.png")
