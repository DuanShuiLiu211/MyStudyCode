"""
These script demonstrates access to MINFLUX list mode data that has been
exported as a .npy file.
"""

import numpy as np

# Import the saved data.
mfx = np.load('W:\桌面\PSD95 647 RIM 594 M2.npy')

# Get localization coordinates (3D) of valid events...
loc = mfx[mfx['vld']]['itr']['loc'][:,-1,:]

# ... and split them into individual components (x,y,z).
loc_x, loc_y, loc_z = loc.T

# Get the mean effective detection rate at tcp offset (out-of-center) positions
# during the final iteration (denoted by the index -1) of valid events.
efo = mfx[mfx['vld']]['itr']['efo'][:,-1]
