import torch
import optoth.activations

import numpy as np
import matplotlib.pyplot as plt

x = torch.from_numpy(np.linspace(-4,4, 501, dtype=np.float32)[:, None]).cuda()
print(x.shape)

op = optoth.activations.TrainableActivation(1, -2, 2, 63, 'spline', init='student-t', norm='l1', init_scale=1).cuda()

y = op(x)

op.weight.proj()

z = op(x)
print(torch.sum(torch.abs(op.weight)))

plt.plot(x[:,0].detach().cpu().numpy(), y[:,0].detach().cpu().numpy())
plt.plot(x[:,0].detach().cpu().numpy(), z[:,0].detach().cpu().numpy(), '--')
plt.show()