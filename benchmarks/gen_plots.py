import numpy as np
import matplotlib.pyplot
import matplotlib.pyplot as plt

plt.plot(np.arange(len(loss_data)), loss_data)
# plt.plot(np.arange(n_iters), std_loss_data, 'bs', np.arange(n_iters), node_loss_data, 'g^')

# for k, v in loss_hist.items():

plt.show()

analysis.profile2d(loss_hist, total_time)
plt.figure()
analysis.profile2d(std_loss_hist, total_time_p)
plt.figure()
analysis.profile2d(domain_loss_hist, total_time_p)
