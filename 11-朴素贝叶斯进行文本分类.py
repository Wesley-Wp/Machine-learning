from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report


def nabayes():
    """
    朴素贝叶斯进行文本分类
    :return: None
    """
    # 加载新闻数据
    news = fetch_20newsgroups(subset='all')

    # 进行数据分割（训练集和数据集）
    x_train, x_test, y_train, y_test = train_test_split(news.data, news.target, test_size=0.25)

    # 对数据集进行特征抽取
    tf = TfidfVectorizer()

    # 以训练集中的数据，统计每篇文章重要词的重要性
    x_train = tf.fit_transform(x_train)

    print(tf.get_feature_names())
    x_test = tf.transform(x_test)

    # 进行朴素贝叶斯算法的预测
    mlt = MultinomialNB(alpha=1.0)
    print(x_train.toarray())
    mlt.fit(x_train, y_train)

    y_predict = mlt.predict(x_test)
    print('预测的类别为：', y_predict)

    # 得出准确率
    print('准确率为:', mlt.score(x_test, y_test))
    print("召回率:", classification_report(y_test, y_predict, target_names=news.target))


if __name__ == '__main__':
    nabayes()
