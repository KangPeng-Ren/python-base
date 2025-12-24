import torch
from torch import nn, optim


# 定义模型，迭代一次，观察参数改变即可
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # 定义一层全连接层
        self.linear = nn.Linear(in_features=5, out_features=3)
        # 权重初始化
        self.linear.weight.data = torch.tensor([
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9],
            [1.0, 1.1, 1.2],
            [1.3, 1.4, 1.5],
        ]).T
        self.linear.bias.data = torch.tensor([1.0, 2.0, 3.0])

    # 前向传播
    def forward(self, x):
        x = self.linear(x)
        return x


# 主流程
# 1. 定义数据
X = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]], dtype=torch.float)  # 输入数据 2x5
target = torch.tensor([[0, 0, 0], [0, 0, 0]], dtype=torch.float)  # 输出数据 2x3

# 2. 创建模型
model = Model()

# 3. 前向传播
output = model(X)

# 4. 定义损失函数
loss = nn.MSELoss()
loss_value = loss(output, target)

# 5. 反向传播计算梯度
loss_value.backward()

# 6. 定义优化器
optimizer = optim.SGD(model.parameters(), lr=0.1)

# 7. 利用优化器更新参数
optimizer.step()
optimizer.zero_grad()

# 打印模型参数
for param in model.state_dict():
    print(param)
    print(model.state_dict()[param])