import numpy as np
import matplotlib.pyplot as plt

import torch
import optoth.rot
import optoth.activations


np_k = np.zeros((2, 1, 11, 11), dtype=np.float32)
np_k[0, :, 5, 4] = -1
np_k[0, :, 5, 6] = 1

np_k[1, :, 5, 5] = -4
np_k[1, :, 5, 6] = 1
np_k[1, :, 6, 5] = 1
np_k[1, :, 5, 4] = 1
np_k[1, :, 4, 5] = 1

np_w = np.zeros((2, 31), dtype=np.float32)
np_w[0] = np.linspace(-2, 2, 31, dtype=np.float32) * 3
np_w[1] = np.linspace(-2, 2, 31, dtype=np.float32) * 1

np_angles = np.linspace(0, 2*np.pi, num=8, endpoint=False, dtype=np.float32)

np_x = np.zeros((1, 1, 128, 128), dtype=np.float32)
np_x[..., 32:-32,32:-32] = 1

th_k = torch.tensor(np_k).cuda()
th_angles = torch.tensor(np_angles).cuda()


th_x = torch.tensor(np_x).cuda()

rot_fun = optoth.rot.RotationFunction()
act_fun = optoth.activations.TrainableActivation(2, -2, 2, 31, base_type="linear", init="linear", init_scale=3.0, group=8)
act_fun.weight.data = torch.tensor(np_w)
act_fun.cuda()

th_k_rot = rot_fun.apply(th_k, th_angles).reshape(-1, 1, 11, 11)

np_k_rot = th_k_rot.cpu().numpy()
print(np.mean(np_k_rot, (1,2,3)))

th_kx = act_fun(torch.nn.functional.conv2d(th_x, th_k_rot))

plt.imshow(np.hstack([x for x in th_kx[0].detach().cpu().numpy()]))
plt.show()
