import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.optim.sgd import SGD
import numpy as np

def generate_grid(ax, nums, index, axis_x, axis_y):
    sub_ax = ax.add_subplot(*nums, index)
    sub_ax.scatter(axis_x, axis_y)

if __name__ == '__main__':
    debug = []
    input = torch.randn(3, 5, requires_grad=True)
    target = torch.randint(5, (3,), dtype=torch.int64)
    optimizer = SGD([input], lr=0.01)
    print("target: ", target.cpu().numpy())
    for i in range(100):
        torch.no_grad()
        loss = F.cross_entropy(input, target)
        loss.backward()
        optimizer.step()
        output = F.softmax(input, dim=1)
        output = output.detach().cpu().numpy().reshape(-1)
        debug.append(output)
    debug = np.asfarray(debug)
    ax = plt.figure(0)
    nums = (3, 5)
    for j in range(1, 16):
        generate_grid(ax, nums, j, range(100), debug[:, j-1])
    plt.show()
