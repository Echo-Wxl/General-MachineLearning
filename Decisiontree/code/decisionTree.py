"""
sklearn建立一颗决策树
1. 实例化，建立评估模型对象
2. 通过模型接口训练模型
3. 通过模型接口获得需要的信息
"""
from sklearn import tree
from sklearn.datasets import load_wine
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def buildTree():
    """
    总结：
    1. 八个参数：Criterion, 两个随机性相关的参数（random_state, splitter），
        五个剪枝参数（max_depth, min_samples_split，min_samples_leaf，max_feature，min_impurity_decrease）
    2. 一个属性：feature_importances_
    3. 四个接口：fit, score, apply, predict
    :return:
    """
    # 加载红酒数据集
    wine = load_wine()
    print(wine.data.shape)

    print(wine.target)

    # 将wine整理成一张表
    pd.concat([pd.DataFrame(wine.data), pd.DataFrame(wine.target)], axis=1)
    print(wine.feature_names)
    print(wine.target_names)

    # 划分训练集和测试集
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(wine.data, wine.target, test_size=0.3)
    print(Xtrain.shape)
    print(Xtest.shape)

    # 建立模型
    # random_state 用来设置分支中的随机模式的参数； splitter用来控制决策树中的随机选项
    # max_depth 限制树的最大深度；min_samples_leaf & min_samples_split 一个节点在分之后的每个子节点最少包含多少个训练样本
    clf = tree.DecisionTreeClassifier(criterion="entropy", random_state=30)
    clf.fit(Xtrain, Ytrain)
    score = clf.score(Xtest, Ytest)
    print(score)

    # 画出一棵树
    import graphviz
    feature_name = wine.feature_names
    dot_data = tree.export_graphviz(clf, feature_names=feature_name, class_names=['class_0', 'class_1', 'class_2'],
                                    filled=True, rounded=True)
    graph = graphviz.Source(dot_data)
    print(graph)

    # 探索决策树的特征重要性
    print(clf.feature_importances_)
    feature_importance = [*zip(feature_name, clf.feature_importances_)]
    print(feature_importance)

    # apply（）返回每个测试样本所在的叶子节点的索引
    print(clf.apply(Xtest))
    # predict（）返回每个测试样本的分类结果:返回的是一个大小为n的一维数组，一维数组中的第i个值为模型预测第i个预测样本的标签
    print(clf.predict(Xtest))
    # predict_proba返回的是一个n行k列的数组，第i行第j列上的数值是模型预测第i个预测样本的标签为j的概率。此时每一行的和应该等于1
    print(clf.predict_proba(Xtest))


def max_parameter(Xtrain, Ytrain, Xtest, Ytest):
    """
    网格搜索确定最优的剪枝参数
    :return:
    """
    test = list()
    for i in range(10):
        clf = tree.DecisionTreeClassifier(max_depth=i + 1, criterion="entropy", random_state=30, splitter="random")
        clf = clf.fit(Xtrain, Ytrain)
        score = clf.score(Xtest, Ytest)
        test.append(score)

    plt.plot(range(1, 11), test, color="red", label="max_depth")
    plt.legend()
    plt.show()


def cross_val():
    """
    K折交叉验证
    :return:
    """
    from sklearn.datasets import load_boston
    from sklearn.model_selection import cross_val_score
    from sklearn.tree import DecisionTreeRegressor
    boston = load_boston()
    regressor = DecisionTreeRegressor(random_state=0)
    val_score = cross_val_score(regressor, boston.data, boston.target, cv=10, scoring="neg_mean_squared_error")
    print(val_score)


def titanic_example():
    import pandas as pd
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.model_selection import GridSearchCV
    from sklearn.model_selection import cross_val_score
    import matplotlib.pyplot as plt
    import numpy as np

    data = pd.read_csv(r"data.csv", index_col=0)
    print(data.head())
    print(data.info())

    # 删除缺失值过多的列，和观察判断来说和预测的y没有关系的列
    data.drop(["Cabin", "Name", "Ticket"], inplace=True, axis=1)

    # 处理缺失值，对缺失值较多的列进行填补，有一些特征只确实一两个值，可以采取直接删除记录的方法
    data["Age"] = data["Age"].fillna(data["Age"].mean())
    data = data.dropna()

    # 将二分类变量转换为数值型变量
    # astype能够将一个pandas对象转换为某种类型，和apply(int(x))不同，astype可以将文本类转换为数字，用这个方式可以很便捷地将二分类特征转换为0~1
    data["Sex"] = (data["Sex"] == "male").astype("int")

    # 将三分类变量转换为数值型变量
    labels = data["Embarked"].unique().tolist()
    data["Embarked"] = data["Embarked"].apply(lambda x: labels.index(x))

    # 查看处理后的数据集
    print(data.head())

    # 提取标签和特征矩阵， 分测试集和训练集
    X = data.iloc[:, data.columns != "Survived"]
    y = data.iloc[:, data.columns == "Survived"]

    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, y, test_size=0.3)

    # 修正测试集和训练集的索引
    for i in [Xtrain, Xtest, Ytrain, Ytest]:
        i.index = range(i.shape[0])

    # 查看分好的训练集和测试集
    print(Xtrain.head())

    # 建立模型
    clf = DecisionTreeClassifier(random_state=25)
    clf = clf.fit(Xtrain, Ytrain)
    score_ = clf.score(Xtest, Ytest)
    print(score_)
    score = cross_val_score(clf, X, y, cv=10).mean()
    print(score)

    # 在不同max_depth下观察模型的拟合状况
    tr = []
    te = []
    for i in range(10):
        clf = DecisionTreeClassifier(random_state=25
                                     , max_depth=i + 1
                                     , criterion="entropy"
                                     )
        clf = clf.fit(Xtrain, Ytrain)
        score_tr = clf.score(Xtrain, Ytrain)
        score_te = cross_val_score(clf, X, y, cv=10).mean()
        tr.append(score_tr)
        te.append(score_te)
    print(max(te))
    plt.plot(range(1, 11), tr, color="red", label="train")
    plt.plot(range(1, 11), te, color="blue", label="test")
    plt.xticks(range(1, 11))
    plt.legend()
    plt.show()

    # 用网格搜索调整参数
    gini_thresholds = np.linspace(0, 0.5, 20)

    parameters = {'splitter': ('best', 'random')
        , 'criterion': ("gini", "entropy")
        , "max_depth": [*range(1, 10)]
        , 'min_samples_leaf': [*range(1, 50, 5)]
        , 'min_impurity_decrease': [*np.linspace(0, 0.5, 20)]}

    clf = DecisionTreeClassifier(random_state=25)
    GS = GridSearchCV(clf, parameters, cv=10)
    GS.fit(Xtrain, Ytrain)

    print(GS.best_params_)
    print(GS.best_score_)


if __name__ == '__main__':
    # buildTree()
    # cross_val()
    titanic_example()
