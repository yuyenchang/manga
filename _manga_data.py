#########################
##  _manga_data.py     ##
##  Yu-Yen Chang       ##
##  2022.03.01         ##
#########################

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.image as mpimg
import os.path
from astropy.io import fits, ascii
from skimage.transform import resize

################################################################################

## Choose image size
d0 = 4690 #len(data)
d11 = 1
d21 = 3
d31 = 2
d2 = 50 #562 
d3 = 50 #562
d22 = 281
d23 = 281 
nrot = 8

load_image = False ## False: if 'img_2d_rot.npy' is available
if load_image:

	# Load imaging cubes
	images1 = np.zeros((d0, d11, d2, d3))
	for i in range(0, d0): 
		x = files1[i]
		image = np.zeros((d11, d2, d3)) - 9999 ## Set no files to -9999
		if os.path.exists(x): 
			hdul = fits.open(x)
			data13 = hdul[3].data
			data1 = data13[np.newaxis, 102, :, :]                    ## vel Ha
			# data1 = np.concatenate((
			# 	data13[np.newaxis,  45, :, :], data13[np.newaxis, 273, :, :],     #flux Ha
			# 	data13[np.newaxis, 102, :, :], data13[np.newaxis, 330, :, :],     #vel Ha
			# 	data13[np.newaxis, 159, :, :], data13[np.newaxis, 387, :, :],     #disp Ha
			# 	data13[np.newaxis, 216, :, :], data13[np.newaxis, 444, :, :]), 0) #EW Ha
			image = resize(data1, (d11, d2, d3))
		images1[i, :, :, :] = image
	inan = np.isnan(images1)   ## Set NaN to 0
	images1[inan] = 0          ## Set NaN to 0

	## Load imaging by Lihwai (SDSS gri)
	images2 = np.zeros((d0, d21, d22, d23))
	for i in range(0, d0):
		x = files2[i] 
		image = np.zeros((d21, d22, d23)) - 9999  ## Set no files to -9999
		if os.path.exists(x): 
			img = mpimg.imread(x)
			data2 = np.moveaxis(img[:, :, 0:3], 2, 0)
			# image = resize(data2, (d21, d22, d23))
			image = data2
		images2[i, :, :, :] = image

	## Add drdv
	drm = np.zeros((d0, 1, d2, d3))
	# drm = np.zeros((d0, 1))
	for i in range(0, d0): 
		drm[i, :, :, :] = np.full((d2, d3), dr[i])
		# drm[i, :] = dr[i]
	dvm = np.zeros((d0, 1, d2, d3))
	# dvm = np.zeros((d0,1))
	for i in range(0, d0): 
		dvm[i, :, :, :] = np.full((d2, d3), dv[i])
		# dvm[i,:] = dv[i]
	imagse3 = np.concatenate((drm, dvm), 1) 
	images3 = images3[:,:,0,0]
	images3 = images3[:,:,np.newaxis,np.newaxis]

	## save all input data
	np.save('img_2d_1', images1)
	np.save('img_2d_2_rs', images2)
	np.save('img_2d_3', images3)

################################################################################

images1 = np.load('img_2d_1.npy') #(4690, 1, 50, 50)
images2 = np.load('img_2d_2_rs.npy') #(4690, 3, 281, 281)
images3 = np.load('img_2d_3.npy') #(4690, 2, 1, 1)
