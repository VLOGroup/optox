import numpy as np
import matplotlib.pyplot as plt

import torch
import optoth.rot


np_k = np.zeros((2, 1, 7, 7), dtype=np.float32)
np_k[0, :, :, 3] = -1
np_k[0, :, :, 4] = 1

np_k[1, :, :, 3] = -4
np_k[1, :, :, 4] = 4
np_angles = np.linspace(0, 2*np.pi, num=11, endpoint=False, dtype=np.float32)
# np_angles = np.asarray([0, 45/180.*np.pi], dtype=np.float32)


th_k = torch.tensor(np_k).cuda()
th_angles = torch.tensor(np_angles).cuda()

th_k.requires_grad_(True)

rot_fun = optoth.rot.RotationFunction()

th_k_rot = rot_fun.apply(th_k, th_angles)


np_k_rot = th_k_rot.detach().cpu().numpy()
print(np_k_rot.shape)
k_plt = np.hstack([np_k_rot[1,i,0] for i in range(len(np_angles))])

plt.imshow(k_plt)
plt.show()
