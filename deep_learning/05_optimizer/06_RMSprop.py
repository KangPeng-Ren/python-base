import torch
import matplotlib.pyplot as plt
import numpy as np

"""
均方根传播
"""


def f(X):
    """
    定义函数：f(x) = 0.05 * (x_1)^2 + (x_2)^2
    :param X: 参数 x_1和 x_2
    :return: 函数值
    """
    return 0.05 * X[0] ** 2 + X[1] ** 2


# 定义函数，实现梯度下降
def gradient_descent(X, optimizer, num_iters):
    # 拷贝当前X的值，放入列表中
    X_arr = X.detach().numpy().copy()

    for i in range(num_iters):
        # 前向传播，得到损失值
        y = f(X)
        # 反向传播
        y.backward()
        # 更新参数
        optimizer.step()
        # 梯度清零
        optimizer.zero_grad()

        # 将更新之后的X保存到列表中
        X_arr = np.vstack([X_arr, X.detach().numpy()])

    return X_arr


# 主流程
# 1. 参数X初始化
X = torch.tensor([-7.0, 2.0], requires_grad=True)

# 2. 定义超参数
lr = 0.1
# num_iters = 500
num_iters = 80

# 3. 优化器对比
# 3.1 SGD
X1 = X.clone().detach().requires_grad_(True)
optimizer = torch.optim.SGD([X1], lr=lr)
X_arr1 = gradient_descent(X1, optimizer, num_iters)
plt.plot(X_arr1[:, 0], X_arr1[:, 1], 'r')

# 3.2 Adagrad
X2 = X.clone().detach().requires_grad_(True)
optimizer = torch.optim.RMSprop([X2], lr=lr, alpha=0.9)
X_arr2 = gradient_descent(X2, optimizer, num_iters)
plt.plot(X_arr2[:, 0], X_arr2[:, 1], 'b')

# 等高线绘制
x1_grad, x2_grad = np.meshgrid(np.linspace(-7, 7, 100), np.linspace(-2, 2, 100))
y_grad = 0.05 * x1_grad ** 2 + x2_grad ** 2
plt.contour(x1_grad, x2_grad, y_grad, levels=30, colors='gray')
plt.legend(['SGD', 'RMSProp'])
plt.show()