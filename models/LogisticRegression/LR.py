import numpy  as np
import matplotlib.pyplot as plt
# 设置随机数种子，方便复现
np.random.seed(42)

#生成随机数据
#两个特征的均值和方差，通过mean和cov
# np.random.multivariate_normal 是 NumPy 中用于生成多元正态分布随机样本的函数
mean_1 = [2,2]
cov_1  = [[2,0],[0,2]]
mean_2 = [-2,-2]
cov_2 = [[1,0],[0,1]]
# 表示两个随机变量的均值为 2，方差为 2，且相关系数为 0。
x1 = np.random.multivariate_normal(mean_1,cov_1,50)
y1 = np.zeros(50)
print(x1)
# 此时生成的是 [[ 2.70245989e+00  1.80446475e+00].....]

# 绘制x1,y1
# # plt.scatter(x1[:,0],x1[:,1],alpha=0.5)
# # plt.show()
x2 = np.random.multivariate_normal(mean_2,cov_2,50)
y2 = np.ones(50)



X = np.concatenate((x1,x2),axis=0)
y = np.concatenate((y1,y2),axis=0)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set1, edgecolors='k')
plt.xlabel("this is X")
plt.ylabel("this is Y")
plt.title("this is X and Y")
plt.show()

def sigmoid(x):
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))


class RogisticRegression:
    def __init__(self, learning_rate=0.01, max_iter=1000):  # 传入学习率和最大迭代次数
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        # 权重和偏置
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        num_samples, num_features = X.shape
        """
        对于二维数组，X.shape 返回一个元组 (rows, columns)：
            rows 表示数组的行数（通常对应样本数量）
            columns 表示数组的列数（通常对应特征数量）
        """
        # 初始化权重和偏置
        self.weights = np.zeros(num_features)
        self.bias = 0
        # 梯度下降
        for _ in range(self.max_iter):
            linear_model = np.dot(X, self.weights) + self.bias
            y_pred = sigmoid(linear_model)
            # 求导
            dw = (1 / num_samples) * np.dot(X.T, y_pred - y)
            db = (1 / num_samples) * np.sum(y_pred - y)

            self.weights = self.learning_rate * dw
            self.bias = self.learning_rate * db

    def predict_prod(self, x):
        linear_model = np.dot(x, self.weights) + self.bias
        y_pred = sigmoid(linear_model)
        return y_pred

    def predict(self, X, threshold=0.5):
        y_pred_prod = self.predict_prod(X)
        y_pred = np.zeros_like(y_pred_prod)
        y_pred[y_pred_prod > threshold] = 1
        return y_pred

logreg = RogisticRegression()
logreg.fit(X,y)
X_new = np.array([[1.5,1.5],[-4.0,-2.0]])
y_pred_prob = logreg.predict_prod(X_new)
y_pred = logreg.predict(X_new)
#输出结果
print("Predicted Probabilities:", y_pred_prob)
print("Predicted Labels:", y_pred)