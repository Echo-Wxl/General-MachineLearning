# 随机森林

## 1 集成学习

集成学习不是单独的机器学习算法，是通过在数据上构建多个模型，集成学习会考虑多个模型的建模结果，汇总之后得到一个综合的结果，以此来获取比单个模型更好的回归或分类表现。集成学习常用的思想主要有两种，bagging和boosting。

**bagging**：从训练数据集中有放回的随机抽取T次，每次m个训练样本，基于这m个训练样本训练T个基学习器，最后将这T个基学习器的结果汇总；如果是分类任务，则进行投票；如果是回归任务，则取均值。最典型的算法就是随机森林。

**boosting**：boosting是将弱分类器提升为强学习器的算法。首先从初始数据集训练一个基学习器，再根据基学习器的表现调整训练数据，使得基学习器分类错误的样本在后续受到更多关注，然后基于调整后的样本分布训练下一个基学习器；反复迭代，直到基学习器的数目达到指定的值，最终将T个基学习器加权求和。典型算法包括Adaboost，GBDT，XGBoost等。

## 2 原理

对原始数据集进行有放回的随机抽样T次，每次m个训练样本，然后训练T棵决策树，将这些决策树的结果汇总。如果是分类任务，则进行投票；如果是回归任务，则对结果取平均。

## 3 本质

随机森林算法的本质就是训练多棵决策树，但是建立决策树的过程中存在两个核心问题：

**每次建立的决策树均为不同的树**：保证每次随机采样特征和样本数量，训练不同的决策树。bagging思想旨在降低模型方差，各子模型完全独立，可以显著降低方差；若各子模型完全相同，则不会降低方差。

**单棵决策树的准确度大于50%**：如果单棵决策树的性能低于50%，说明是无效的树，不能对预测结果起到正向作用。

## 4 为什么选择决策树作为基学习器？

1. 决策树可以较为方便地将样本的权重整合到训练过程中，不需要使用过采样的方法来调整样本权重；
2. 决策树的表达能力和泛化能力，可以通过调节树的深度来折中；
3. 不同子集样本对于生成决策树基分类器随机性较大，这种不稳定的学习器更适合作为基分类器。

## 5 优缺点

​	**优点**

- 能够处理高纬度数据，而且不用降维
- 不容易过拟合
- 并行训练，模型准确率高

​	**缺点**

- 回归算法的效果并不理想，因为不能给出连续的输出，不能做出超出训练样本之外的输出

## 6 案例

### 6.1 随机森林与决策树的对比

```python
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
```

### 6.2 重要参数

​	**1、常用参数：**

> criterion :不纯度的衡量指标，有基尼系数和信息熵两种选择
> ​max_depth: 树的最大深度，超过最大深度的树枝都会被剪掉
> ​min_samples_leaf:一个节点在分枝后的每个子节点都必须包含至少min_samples_leaf个训练样本，否则分枝就不会发生
> ​min_samples_split:一个节点必须要包含至少min_samples_split个训练样本，这个节点才允许被分枝，否则分枝就不会发生
> ​max_features: 限制分枝时考虑的特征个数，超过限制个数的特征都会被舍弃，默认值为总特征个数开平方取整
> ​min_impurity_decrease 限制信息增益的大小，信息增益小于设定数值的分枝不会发生

​	2、**n_estimators: 随机森林中决策树的数目**

```python
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
```

​	**3、 随机性：random_state、bootstrap、oob_score**

​	skearn中的分类树DecisionTreeClassifier自带随机性，每次生成的决策树都不一样，这个功能是由参数random_state控制的。

​	随机森林中的random_state用法和决策树中相似，只不过在分类树中，一个random_state只控制生成一棵树，而随机森林中的random_state控制的是生成森林的模式，而非让一个森林中只有一棵树。

```python
def estimators_():
    rfc = RandomForestClassifier(n_estimators=20,random_state=2)
    rfc = rfc.fit(Xtrain, Ytrain)

    #随机森林的重要属性之一：estimators，查看森林中树的状况
    rfc.estimators_[0].random_state

    for i in range(len(rfc.estimators_)):
        print(rfc.estimators_[i].random_state)
```

​	当random_state固定时，随机森林中生成是一组固定的树，但每棵树依然是不一致的，这是用”随机挑选特征进行分枝“的方法得到的随机性。当这种随机性越大的时候，袋装法的效果一般会越来越好。用袋装法集成时，基分类器应当是相互独立的，是不相同的。

​	除了random_state外，还需要其他的随机性。要让基分类器尽量都不一样，一种很容易理解的方法是使用不同的训练集来进行训练，而袋装法正是通过有放回的随机抽样技术来形成不同的训练数据，bootstrap就是用来控制抽样技术的参数`bootstrap参数默认True，表示采用有放回的随机抽样。`

​	然而有放回抽样也会有自己的问题。由于是有放回，一些样本可能在同一个自助集中出现多次，而其他一些却可能
被忽略，一般来说，自助集大约平均会包含63%的原始数据。因为每一个样本被抽到某个自助集中的概率为
$$
1-(1-\frac{1}{n})^n
$$
​	当n足够大时,这个概率收敛于$$1-(\frac{1}{e})$$，约等于0.632。因此，会有约37%的训练数据被浪费掉，没有参与建模这些数据被称为袋外数据( out of bag data,简写为oob)。除了我们最开始就划分好的测试集之外,这些数据也以被用来作为集成算法的测试集。也就是说，在使用随机森林时，我们可以不划分测试集和训练集，只需要用袋外数据来测试我们的模型即可。
​	如果希望用袋外数据来测试，则需要在实例化时就将 oob score这个参数调整为True，训练完毕之后，我们可以用随机森林的另一个重要属性：oob score来查看我们的在袋外数据上测试的结果：

```python
#无需划分训练集和测试集
rfc = RandomForestClassifier(n_estimators=25,oob_score=True)#默认为False
rfc = rfc.fit(wine.data,wine.target)
 
#重要属性oob_score_
rfc.oob_score_#0.9719101123595506
```

​	**4、其他属性，接口**

```python
def others():
    # 其他属性和接口
    rfc = RandomForestClassifier(n_estimators=25)
    rfc = rfc.fit(Xtrain, Ytrain)
    print(rfc.score(Xtest, Ytest))

    print(rfc.feature_importances_)  # 结合zip可以对照特征名字查看特征重要性，参见上节决策树
    print(rfc.apply(Xtest))  # apply返回每个测试样本所在的叶子节点的索引
    print(rfc.predict(Xtest))  # predict返回每个测试样本的分类/回归结果
    print(rfc.predict_proba(Xtest))  # predict_proba返回每个测试样本的预测概率
```

### 6.3 用随机森林回归填补缺失值

```python
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
```











