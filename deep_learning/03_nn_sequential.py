import torch
import torch.nn as nn
from torchsummary import summary


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1. 定义数据
x = torch.randn(10, 3).to(device)

# 2. 构建模型
model = nn.Sequential(
    nn.Linear(3, 4),
    nn.Tanh(),
    nn.Linear(4, 4),
    nn.ReLU(),
    nn.Linear(4, 2),
    nn.Softmax(dim=1)
)
model.to(device)


# 3. 参数初始化
def init_params(layer):  # 定义参数初始化函数
    if isinstance(layer, nn.Linear):
        nn.init.xavier_uniform_(layer.weight)
        nn.init.constant_(layer.bias, 0.1)


model.apply(init_params)

# 4. 前向传播
y = model(x)
print(y)

summary(model, input_size=(3, ), batch_size=10, device=device.type)