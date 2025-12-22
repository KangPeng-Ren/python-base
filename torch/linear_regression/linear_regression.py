import torch
from torch import nn, optim  # 神经网络模型和优化器
from torch.utils.data import TensorDataset, DataLoader  # 构建数据集和数据加载器
import matplotlib.pyplot as plt

# 1. 构建数据集，创建数据加载器
X = torch.randn(100, 1)
# 预设真实系数
w = torch.tensor([2.5])
b = torch.tensor([5.2])
# 定义随机噪声
noise = torch.randn(100, 1) * 0.1
# 定义拟合目标值y
y = w * X + b + noise
# 构建DataSet数据集
dataset = TensorDataset(X, y)
# 构建DataLoader
dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

# 2. 构建模型
model = nn.Linear(in_features=1, out_features=1)  # 输入神经元个数和输出神经元个数

# 3. 定义损失函数和优化器
loss = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)  # 优化目标：模型参数（权重和偏置，并非w、b[真实系数]）

# 4. 模型训练
epochs = 1000  # 超参数：训练轮次
loss_list = []
for epoch in range(epochs):
    total_loss = 0  # 记录本轮总损失

    # 每一个轮次遍历DataLoader
    for X_train, y_train in dataloader:
        # 4.1 前向传播（模型预测）
        y_pred = model(X_train)
        # 4.2 计算损失
        loss_value = loss(y_pred, y_train)
        total_loss += loss_value.item() * X_train.shape[0]
        # 4.3 反向传播
        loss_value.backward()
        # 4.4 更新参数
        optimizer.step()
        # 4.5 梯度清零，为下一次迭代做准备
        optimizer.zero_grad()

    # 计算本轮平均损失
    loss_list.append(total_loss / len(dataset))

# 模型训练完成，打印模型参数
print(model.weight)
print(model.bias)
'''
Parameter containing:
tensor([[2.5019]], requires_grad=True)
Parameter containing:
tensor([5.1905], requires_grad=True)
'''

# 画图进行显示
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
# 训练损失随轮次的变化
ax[0].plot(loss_list)
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('Loss')
# 绘制散点图和拟合直线
ax[1].scatter(X, y)
y_pred = model.weight.item() * X + model.bias.item()
ax[1].plot(X, y_pred, color='red')
plt.show()