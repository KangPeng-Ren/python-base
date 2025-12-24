import torch
import matplotlib.pyplot as plt

x = torch.linspace(-10, 10, 1000, requires_grad=True)
y = torch.tanh(x)

# 绘制Tanh函数图像
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(x.detach(), y.detach(), color='purple')
ax[0].set_title('tanh')
ax[0].axhline(y=1, color='gray', linestyle='--', alpha=0.5, linewidth=1)
ax[0].axhline(y=-1, color='gray', linestyle='--', alpha=0.5, linewidth=1)

# 设置边界
ax[0].spines['top'].set_visible(False)
ax[0].spines['right'].set_visible(False)
ax[0].spines['left'].set_position('zero')
ax[0].spines['bottom'].set_position('zero')

# 反向传播，计算x的梯度
y.sum().backward()

# 绘制导函数图像 f`(x) = 1 - (f(x)) ^ 2
ax[1].plot(x.data, x.grad, 'purple')
ax[1].set_title('tanh`')

# 设置边界
ax[1].spines['top'].set_visible(False)
ax[1].spines['right'].set_visible(False)
ax[1].spines['left'].set_position('zero')
ax[1].spines['bottom'].set_position('zero')


plt.show()