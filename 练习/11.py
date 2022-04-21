import numpy as np
from scipy import io
data = np.random.random(3)
np.save('W:\桌面\{}.mat'.format('PSD95 647 RIM 594 M1'), data)
io.savemat('W:\桌面\{}.mat'.format('PSD95 647 RIM 594 M1'), {'data':data})