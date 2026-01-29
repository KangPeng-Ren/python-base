import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import MultiStepLR

"""
里程碑学习率衰减
"""


def f(X):
    """
    定义函数：f(x) = 0.05 * (x_1)^2 + (x_2)^2
    :param X: 参数 x_1和 x_2
    :return: 函数值
    """
    return 0.05 * X[0] ** 2 + X[1] ** 2

# 主流程
# 1. 参数X初始化
X = torch.tensor([-7.0, 2.0], requires_grad=True)

# 2. 定义超参数
lr = 0.9
num_iters = 500

# 3. 定义优化器
optimizer = torch.optim.SGD([X], lr=lr)

# 4. 定义学习率衰减策略
lr_scheduler = MultiStepLR(optimizer, milestones=[10, 50, 200], gamma=0.7)

# 5. 梯度下降
X_arr = X.detach().numpy().copy()
lr_list = []
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
    lr_list.append(optimizer.param_groups[0]['lr'])

    # 更新学习率
    lr_scheduler.step()


plt.rcParams['font.sans-serif'] = ['KaiTi']
plt.rcParams['axes.unicode_minus'] = False
# 等高线绘制
fig, ax = plt.subplots(1, 2, figsize=(12, 4))

x1_grad, x2_grad = np.meshgrid(np.linspace(-7, 7, 100), np.linspace(-2, 2, 100))
y_grad = 0.05 * x1_grad ** 2 + x2_grad ** 2

ax[0].contour(x1_grad, x2_grad, y_grad, levels=30, colors='gray')
ax[0].plot(X_arr[:, 0], X_arr[:, 1], 'r')
ax[0].set_title('梯度下降过程')

ax[1].plot(lr_list, 'k')
ax[1].set_title('学利率衰减')
plt.show()