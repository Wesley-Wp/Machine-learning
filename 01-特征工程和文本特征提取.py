import jieba
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler


def dictvec():
    """字典数据抽取"""
    # 实例化
    dict = DictVectorizer(sparse=False)
    # 调用fit_transfrom
    data = dict.fit_transform([
        {'city': '北京', 'temperature': 100},
        {'city': '上海', 'temperature': 60},
        {'city': '深圳', 'temperature': 30}]
    )
    print(dict.get_feature_names())
    print(dict.inverse_transform(data))
    print(data)

    return None


def countvec():
    """
    文本特征提取
    :return:
    """
    # 实例化
    cv = CountVectorizer()
    data = cv.fit_transform(["life is short,i like python", "life is too long,i dislike python"])

    print(cv.get_feature_names())
    print(data.toarray())


def cutword():
    """
    分割中文
    :return:
    """
    con1 = jieba.cut("今天很残酷，明天更残酷，后天很美好，但绝对大部分是死在明天晚上，所以每个人不要放弃今天。")

    con2 = jieba.cut("我们看到的从很远星系来的光是在几百万年之前发出的,这样当我们看到宇宙时，我们是在看它的过去。")

    con3 = jieba.cut("如果只用一种方式了解某样事物，你就不会真正了解它。了解事物真正含义的秘密取决于如何将其与我们所了解的事物相联系。")

    # 转换成列表
    content1 = list(con1)
    content2 = list(con2)
    content3 = list(con3)

    # 把列表转换为字符串
    c1 = " ".join(content1)
    c2 = " ".join(content2)
    c3 = " ".join(content3)

    return c1, c2, c3


def hanzivec():
    """
    中文特征提取
    :return:
    """
    c1, c2, c3 = cutword()
    print(c1, c2, c3)
    cv = CountVectorizer()
    data = cv.fit_transform([c1, c2, c3])
    print(cv.get_feature_names())
    print(data.toarray())


def tfidfvec():
    """
    中文特征提取
    :return:
    """
    c1, c2, c3 = cutword()
    print(c1, c2, c3)
    tf = TfidfVectorizer()
    data = tf.fit_transform([c1, c2, c3])
    print(tf.get_feature_names())
    print(data.toarray())


if __name__ == '__main__':
    # dictvec()
    # countvec()
    # hanzivec()
    tfidfvec()

