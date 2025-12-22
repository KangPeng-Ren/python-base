import torch
import torch.nn as nn
from torchsummary import summary


# 自定义神经网络类
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()  # 调用基类初始化方法

        # 定义三个线性层
        self.linear1 = nn.Linear(3, 4)
        nn.init.xavier_normal_(self.linear1.weight)  # 权重参数初始化
        self.linear2 = nn.Linear(4, 4)
        nn.init.kaiming_normal_(self.linear2.weight)  # 权重参数初始化
        self.linear3 = nn.Linear(4, 2)  # 默认初始化

    # 定义前向传播
    def forward(self, x):
        x = torch.tanh(self.linear1(x))
        x = torch.relu(self.linear2(x))
        return torch.softmax(self.linear3(x), dim=1)

# 定义运算设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 测试
# 1. 定义输入数据
x = torch.randn(10, 3)
x.to(device)

# 2. 创建神经网络模型
model = Model()
model.to(device)

# 3. 前向传播，通过对象名直接调用forward()
y = model(x)

print('当前神经网络输出为：', y)

# 查看神经网络参数，格式混乱
# print(model.linear1.weight)
# print(model.linear1.bias)
# print(model.linear2.weight)
# print(model.linear2.bias)
# print(model.linear3.weight)
# print(model.linear3.bias)

# 调用parameters查看所有参数，效果同上，写法简单
# for param in model.parameters():
#     print(param)

for param in model.named_parameters():
    print(param)  # 二元组，包含参数的名称和参数的值

print('=========================================')

# 调用state_dict()状态字典得到所有参数的字典表示
state_dict = model.state_dict()
print(state_dict)

print('=========================================')
print('查看模型架构和参数数量：')
summary(model, input_size=(3,), batch_size=10, device='cpu')
