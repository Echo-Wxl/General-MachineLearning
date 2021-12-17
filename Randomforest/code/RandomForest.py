from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def randomForest():
    wine = load_wine()
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(wine.data, wine.target, test_size=0.3)

    # 决策树和随机森林的对比
    dtc = DecisionTreeClassifier(random_state=0)
    rfc = RandomForestClassifier(random_state=0)
    dtc = dtc.fit(Xtrain, Ytrain)
    rfc = rfc.fit(Xtrain, Ytrain)
    score_c = dtc.score(Xtest, Ytest)
    score_r = rfc.score(Xtest, Ytest)
    print("Single Tree:{}".format(score_c), "Random Forest:{}".format(score_r))

    # 交叉验证:将数据集划分为n份，依次取每一份做测试集，每n-1份做训练集，多次训练模型以观测模型稳定性的方法
    label = "RandomForest"
    for model in [RandomForestClassifier(n_estimators=25), DecisionTreeClassifier()]:
        score = cross_val_score(model, wine.data, wine.target, cv=10)
        print("{}:".format(label)), print(score.mean())
        plt.plot(range(1, 11), score, label=label)
        plt.legend()
        label = "DecisionTree"
    plt.show()

    # 其他属性和接口
    rfc = RandomForestClassifier(n_estimators=25)
    rfc = rfc.fit(Xtrain, Ytrain)
    print(rfc.score(Xtest, Ytest))

    print(rfc.feature_importances_)  # 结合zip可以对照特征名字查看特征重要性，参见上节决策树
    print(rfc.apply(Xtest))  # apply返回每个测试样本所在的叶子节点的索引
    print(rfc.predict(Xtest))  # predict返回每个测试样本的分类/回归结果
    print(rfc.predict_proba(Xtest))  # predict_proba返回每个测试样本的预测概率

def n_estimators_test():
    """
    n_estimators: 随机森林中决策树的数量
    :return:
    """
    wine = load_wine()
    super_n = []
    for i in range(100):
        rfc = RandomForestClassifier(n_estimators=i + 1, n_jobs=-1)
        rfc_s = cross_val_score(rfc, wine.data, wine.target, cv=10).mean()
        super_n.append(rfc_s)
    print(max(super_n), super_n.index(max(super_n)) + 1)  # 打印出：最高精确度取值，依次随机森林的数目
    plt.figure(figsize=[20, 5])
    plt.plot(range(1, 201), super_n)
    plt.show()


def random_test():
    """
    随机森林的随机性
    :return:
    """
    wine = load_wine()
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(wine.data, wine.target, test_size=0.3)
    rfc = RandomForestClassifier(n_estimators=20, random_state=2)
    rfc = rfc.fit(Xtrain, Ytrain)

    # 随机森林的重要属性之一：estimators，查看森林中树的状况
    print(rfc.estimators_[0].random_state)

    for i in range(len(rfc.estimators_)):
        print(rfc.estimators_[i].random_state)

    # 无需划分训练集和测试集
    rfc = RandomForestClassifier(n_estimators=25, oob_score=True)  # 默认为False
    rfc = rfc.fit(wine.data, wine.target)

    # 重要属性oob_score_
    print(rfc.oob_score_)


def rfc_fill_nan():
    """
    使用随机森林填补缺失值
    :return:
    """
    from sklearn.datasets import load_boston
    from sklearn.impute import SimpleImputer  # 填补缺失值的类
    from sklearn.ensemble import RandomForestRegressor

    """1. 加载波士顿房价数据集"""
    dataset = load_boston()  # 总共506*13=6578个数据

    X_full, y_full = dataset.data, dataset.target
    n_samples = X_full.shape[0]  # 506
    n_features = X_full.shape[1]  # 13

    """2. 为完整数据集插入缺失值"""
    # 首先确定我们希望放入的缺失数据的比例，在这里我们假设是50%，那总共就要有3289个数据缺失
    rng = np.random.RandomState(0)  # 设置一个随机种子，方便观察
    missing_rate = 0.5
    n_missing_samples = int(np.floor(n_samples * n_features * missing_rate))  # 3289

    # 所有数据要随机遍布在数据集的各行各列当中，而一个缺失的数据会需要一个行索引和一个列索引
    # 如果能够创造一个数组，包含3289个分布在0~506中间的行索引，和3289个分布在0~13之间的列索引，那我们就可以利用索引来为数据中的任意3289个位置赋空值
    # 然后我们用0，均值和随机森林来填写这些缺失值，然后查看回归的结果如何

    missing_features = rng.randint(0, n_features, n_missing_samples)  # randint（下限，上限，n）指在下限和上限之间取出n个整数
    missing_samples = rng.randint(0, n_samples, n_missing_samples)

    # 我们现在采样了3289个数据，远远超过我们的样本量506，所以我们使用随机抽取的函数randint。
    # 但如果我们需要的数据量小于我们的样本量506，那我们可以采用np.random.choice来抽样，choice会随机抽取不重复的随机数，
    # 因此可以帮助我们让数据更加分散，确保数据不会集中在一些行中!
    # 这里我们不采用np.random.choice,因为我们现在采样了3289个数据，远远超过我们的样本量506，使用np.random.choice会报错

    X_missing = X_full.copy()
    y_missing = y_full.copy()

    X_missing[missing_samples, missing_features] = np.nan
    X_missing = pd.DataFrame(X_missing)
    # 转换成DataFrame是为了后续方便各种操作，numpy对矩阵的运算速度快到拯救人生，但是在索引等功能上却不如pandas来得好用
    print(X_missing.head())

    """3.使用0和均值填补缺失值"""
    # 使用均值进行填补
    imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')  # 实例化
    X_missing_mean = imp_mean.fit_transform(X_missing)  # 特殊的接口fit_transform = 训练fit + 导出predict

    # 使用0进行填补
    imp_0 = SimpleImputer(missing_values=np.nan, strategy="constant", fill_value=0)  # constant指的是常数
    X_missing_0 = imp_0.fit_transform(X_missing)

    """4.使用随机森林填补缺失值
    
    任何回归都是从特征矩阵中学习，然后求解连续型标签y的过程，之所以能够实现这个过程，是因为回归算法认为，特征
    矩阵和标签之前存在着某种联系。实际上，标签和特征是可以相互转换的，比如说，在一个“用地区，环境，附近学校数
    量”预测“房价”的问题中，我们既可以用“地区”，“环境”，“附近学校数量”的数据来预测“房价”，也可以反过来，
    用“环境”，“附近学校数量”和“房价”来预测“地区”。而回归填补缺失值，正是利用了这种思想。
    
    对于一个有n个特征的数据来说，其中特征T有缺失值，我们就把特征T当作标签，其他的n-1个特征和原本的标签组成新
    的特征矩阵。那对于T来说，它没有缺失的部分，就是我们的Y_test，这部分数据既有标签也有特征，而它缺失的部
    分，只有特征没有标签，就是我们需要预测的部分。
    
    特征T不缺失的值对应的其他n-1个特征 + 本来的标签：X_train
    特征T不缺失的值：Y_train
    
    特征T缺失的值对应的其他n-1个特征 + 本来的标签：X_test
    特征T缺失的值：未知，我们需要预测的Y_test
    
    那如果数据中除了特征T之外，其他特征也有缺失值怎么办？
    答案是遍历所有的特征，从缺失最少的开始进行填补（因为填补缺失最少的特征所需要的准确信息最少）。
    填补一个特征时，先将其他特征的缺失值用0代替，每完成一次回归预测，就将预测值放到原本的特征矩阵中，再继续填
    补下一个特征。每一次填补完毕，有缺失值的特征会减少一个，所以每次循环后，需要用0来填补的特征就越来越少。当
    进行到最后一个特征时（这个特征应该是所有特征中缺失值最多的），已经没有任何的其他特征需要用0来进行填补了，
    而我们已经使用回归为其他特征填补了大量有效信息，可以用来填补缺失最多的特征。
    
    遍历完所有的特征后，数据就完整了。
    """
    X_missing_reg = X_missing.copy()

    # 找出数据集中，缺失值从小到大排列的特征们的顺序，并且有了这些的索引
    sortindex = np.argsort(X_missing_reg.isnull().sum(axis=0)).values  # np.argsort()返回的是从小到大排序的顺序所对应的索引

    for i in sortindex:
        # 构建我们的新特征矩阵（没有被选中去填充的特征 + 原始的标签）和新标签（被选中去填充的特征）
        df = X_missing_reg
        fillc = df.iloc[:, i]  # 新标签
        df = pd.concat([df.iloc[:, df.columns != i], pd.DataFrame(y_full)], axis=1)  # 新特征矩阵

        # 在新特征矩阵中，对含有缺失值的列，进行0的填补
        df_0 = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0).fit_transform(df)

        # 找出我们的训练集和测试集
        Ytrain = fillc[fillc.notnull()]  # Ytrain是被选中要填充的特征中（现在是我们的标签），存在的那些值：非空值
        Ytest = fillc[fillc.isnull()]  # Ytest 是被选中要填充的特征中（现在是我们的标签），不存在的那些值：空值。注意我们需要的不是Ytest的值，需要的是Ytest所带的索引
        Xtrain = df_0[Ytrain.index, :]  # 在新特征矩阵上，被选出来的要填充的特征的非空值所对应的记录
        Xtest = df_0[Ytest.index, :]  # 在新特征矩阵上，被选出来的要填充的特征的空值所对应的记录

        # 用随机森林回归来填补缺失值
        rfc = RandomForestRegressor(n_estimators=100)  # 实例化
        rfc = rfc.fit(Xtrain, Ytrain)  # 导入训练集进行训练
        Ypredict = rfc.predict(Xtest)  # 用predict接口将Xtest导入，得到我们的预测结果（回归结果），就是我们要用来填补空值的这些值

        # 将填补好的特征返回到我们的原始的特征矩阵中
        X_missing_reg.loc[X_missing_reg.iloc[:, i].isnull(), i] = Ypredict

    # 检验是否有空值
    print(X_missing_reg.isnull().sum())

    """5.对填补好的数据集进行建模"""
    # 对所有数据进行建模，取得MSE结果
    X = [X_full, X_missing_mean, X_missing_0, X_missing_reg]
    mse = [] # 存放负的均方误差
    for x in X:
        estimator = RandomForestRegressor(random_state=0, n_estimators=100)  # 实例化
        scores = cross_val_score(estimator, x, y_full, scoring='neg_mean_squared_error', cv=5).mean()
        mse.append(scores*-1)

    print([*zip(['Full data','Zero Imputation','Mean Imputation','Regressor Imputation'], mse)])

    """6.画图比较"""
    x_labels = ['Full data',
                'Zero Imputation',
                'Mean Imputation',
                'Regressor Imputation']
    colors = ['r', 'g', 'b', 'orange']

    plt.figure(figsize=(12, 6))  # 画出画布
    ax = plt.subplot(111)  # 添加子图
    for i in np.arange(len(mse)):
        ax.barh(i, mse[i], color=colors[i], alpha=0.6, align='center')  # bar为条形图，barh为横向条形图，alpha表示条的粗度
    ax.set_title('Imputation Techniques with Boston Data')
    ax.set_xlim(left=np.min(mse) * 0.9, right=np.max(mse) * 1.1)  # 设置x轴取值范围
    ax.set_yticks(np.arange(len(mse)))
    ax.set_xlabel('MSE')
    ax.set_yticklabels(x_labels)
    plt.show()


def rfc_in_breastCancer():
    """
    随机森林在乳腺癌数据集上的调参
    :return:
    """
    from sklearn.model_selection import GridSearchCV # 网格搜索调参
    from sklearn.datasets import load_breast_cancer

    #导入数据集
    data = load_breast_cancer()
    print(data.data.shape)

    # 建模:初始化准确率
    rfc = RandomForestClassifier(n_estimators=100, random_state=90)
    score_pre = cross_val_score(rfc, data.data, data.target, cv=10).mean()  # 交叉验证的分类默认scoring='accuracy'
    print(score_pre)

    # 先确定n_estimators
    scorel = []
    for i in range(0, 200, 10):
        rfc = RandomForestClassifier(n_estimators=i + 1,
                                     n_jobs=-1,
                                     random_state=90)
        score = cross_val_score(rfc, data.data, data.target, cv=10).mean()
        scorel.append(score)
    print(max(scorel), (scorel.index(max(scorel)) * 10) + 1)
    plt.figure(figsize=[20, 5])
    plt.plot(range(1, 201, 10), scorel)
    plt.show()

    scorel = []
    for i in range(35, 45):
        rfc = RandomForestClassifier(n_estimators=i,
                                     n_jobs=-1,
                                     random_state=90)
        score = cross_val_score(rfc, data.data, data.target, cv=10).mean()
        scorel.append(score)
    print(max(scorel), ([*range(35, 45)][scorel.index(max(scorel))]))
    plt.figure(figsize=[20, 5])
    plt.plot(range(35, 45), scorel)
    plt.show()

    # 调整max_depth
    param_grid = {'max_depth': np.arange(1, 20, 1)}
    #   一般根据数据的大小来进行一个试探，乳腺癌数据很小，所以可以采用1~10，或者1~20这样的试探
    #   但对于像digit recognition那样的大型数据来说，我们应该尝试30~50层深度（或许还不足够
    #   更应该画出学习曲线，来观察深度对模型的影响
    rfc = RandomForestClassifier(n_estimators=39, random_state=90)
    GS = GridSearchCV(rfc, param_grid, cv=10)  # 网格搜索
    GS.fit(data.data, data.target)
    print(GS.best_params_)  # 显示调整出来的最佳参数
    print(GS.best_score_)  # 返回调整好的最佳参数对应的准确率


    # 调整max_features
    param_grid = {'max_features': np.arange(5, 30, 1)}
    """
    max_features是唯一一个即能够将模型往左（低方差高偏差）推，也能够将模型往右（高方差低偏差）推的参数。我
    们需要根据调参前，模型所在的位置（在泛化误差最低点的左边还是右边）来决定我们要将max_features往哪边调。
    现在模型位于图像左侧，我们需要的是更高的复杂度，因此我们应该把max_features往更大的方向调整，可用的特征
    越多，模型才会越复杂。max_features的默认最小值是sqrt(n_features)，因此我们使用这个值作为调参范围的
    最小值。

    """
    rfc = RandomForestClassifier(n_estimators=39, random_state=90)
    GS = GridSearchCV(rfc, param_grid, cv=10)
    GS.fit(data.data, data.target)
    print(GS.best_params_)  # 显示调整出来的最佳参数
    print(GS.best_score_)  # 返回调整好的最佳参数对应的准确率

    # 调整min_samples_leaf
    param_grid = {'min_samples_leaf': np.arange(1, 1 + 10, 1)}
    # 对于min_samples_split和min_samples_leaf,一般是从他们的最小值开始向上增加10或20
    # 面对高维度高样本量数据，如果不放心，也可以直接+50，对于大型数据，可能需要200~300的范围
    # 如果调整的时候发现准确率无论如何都上不来，那可以放心大胆调一个很大的数据，大力限制模型的复杂度
    rfc = RandomForestClassifier(n_estimators=39
                                 , random_state=90
                                 )
    GS = GridSearchCV(rfc, param_grid, cv=10)
    GS.fit(data.data, data.target)
    print(GS.best_params_)  # 显示调整出来的最佳参数
    print(GS.best_score_)  # 返回调整好的最佳参数对应的准确率

    # 调整min_samples_split
    param_grid = {'min_samples_split': np.arange(2, 2 + 20, 1)}
    rfc = RandomForestClassifier(n_estimators=39
                                 , random_state=90
                                 )
    GS = GridSearchCV(rfc, param_grid, cv=10)
    GS.fit(data.data, data.target)
    print(GS.best_params_)  # 显示调整出来的最佳参数
    print(GS.best_score_)  # 返回调整好的最佳参数对应的准确率

    # 调整Criterion
    param_grid = {'criterion': ['gini', 'entropy']}
    rfc = RandomForestClassifier(n_estimators=39
                                 , random_state=90
                                 )
    GS = GridSearchCV(rfc, param_grid, cv=10)
    GS.fit(data.data, data.target)
    print(GS.best_params_)  # 显示调整出来的最佳参数
    print(GS.best_score_)  # 返回调整好的最佳参数对应的准确率


if __name__ == '__main__':
    # randomForest()
    # n_estimators_test()
    # rfc_fill_nan()
    rfc_in_breastCancer()