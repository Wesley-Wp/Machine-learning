import pandas as pd
from sklearn.decomposition import PCA

# 读取四张表的数据
prior = pd.read_csv("./datafile/order_products__prior.csv")
product = pd.read_csv("./datafile/products.csv")
orders = pd.read_csv("./datafile/orders.csv")
aisles = pd.read_csv("./datafile/aisles.csv")

# 合并四张表到一张表
mg = pd.merge(prior, product, on=['product_id', 'product_id'])
mg1 = pd.merge(mg, orders, on=['order_id', 'order_id'])
mg2 = pd.merge(mg1, aisles, on=['aisle_id', 'aisle_id'])
# ret = mg2.head(10)


# 交叉表（特殊的分组表）
cross = pd.crosstab(mg2['user_id'], mg2['aisle'])
# cross.head(10)
# print(ret)
# print(cross.head())

# 进行主成分分析
pca = PCA(n_components=0.9)
data = pca.fit_transform(cross)










