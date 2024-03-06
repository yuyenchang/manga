
#########################
##  _manga.py    ##
##  Yu-Yen Chang       ##
##  2022.03.01         ##
#########################

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.image as mpimg
import os.path
from astropy.io import fits, ascii
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import confusion_matrix
from sklearn.utils import resample
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from skimage.transform import resize
from xgboost import XGBClassifier
from xgboost import DMatrix
import xgboost as xgb
import lightgbm as lgb
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution1D, MaxPooling1D
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
import tensorflow as tf
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.semi_supervised import LabelPropagation
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import random

################################################################################

class GAL:

	kind = 'GAL'

	def __init__(self, name):
		self.name = name    # Variable unique to each instance


	# def boot(self, galf0, images, stages, code):
	def boot(self, X_train, X_test, y_train, y_test, stages, code):

		nn = images[:,0].shape[0] 
		output_b = []
		for i in range(100):

			## bootstrapping for training sample
			nn_train = X_train[:,0].shape[0]
			nn_index_train = resample(list(range(nn_train)))
			X_train_b = X_train[nn_index_train,:]
			y_train_b = y_train[nn_index_train]
			## bootstrapping for testing sample
			nn_test = X_test[:,0].shape[0]
			nn_index_test = resample(list(range(nn_test)))
			X_test_b = X_test[nn_index_test,:]
			y_test_b = y_test[nn_index_test]
			if code ==  1: output_t = self.ml(X_train_b, X_test_b, y_train_b, y_test_b, stages, code)
			if code ==  2: output_t = self.ml(X_train_b, X_test_b, y_train_b, y_test_b, stages, code)
			if code ==  3: output_t = self.ml(X_train_b, X_test_b, y_train_b, y_test_b, stages, code)
			if code ==  4: output_t = self.ml(X_train_b, X_test_b, y_train_b, y_test_b, stages, code)
			if code ==  5: output_t = self.ml(X_train_b, X_test_b, y_train_b, y_test_b, stages, code)
			if code ==  6: output_t = self.ml(X_train_b, X_test_b, y_train_b, y_test_b, stages, code)
			if code ==  7: output_t = self.ml(X_train_b, X_test_b, y_train_b, y_test_b, stages, code)
			if code ==  8: output_t = self.ml(X_train_b, X_test_b, y_train_b, y_test_b, stages, code)
			if code ==  9: output_t = self.ml_xgb(X_train_b, X_test_b, y_train_b, y_test_b, stages)
			if code == 10: output_t = self.ml_keras(X_train_b, X_test_b, y_train_b, y_test_b, stages)
			output_b.append([float(x) for x in output_t])
		output_m = np.mean(output_b,0)
		output_s = np.std(output_b,0)	

		if code ==  1: output_m = self.ml(X_train, X_test, y_train, y_test, stages, code)
		if code ==  2: output_m = self.ml(X_train, X_test, y_train, y_test, stages, code)
		if code ==  3: output_m = self.ml(X_train, X_test, y_train, y_test, stages, code)
		if code ==  4: output_m = self.ml(X_train, X_test, y_train, y_test, stages, code)
		if code ==  5: output_m = self.ml(X_train, X_test, y_train, y_test, stages, code)
		if code ==  6: output_m = self.ml(X_train, X_test, y_train, y_test, stages, code)
		if code ==  7: output_m = self.ml(X_train, X_test, y_train, y_test, stages, code)
		if code ==  8: output_m = self.ml(X_train, X_test, y_train, y_test, stages, code)
		if code ==  9: output_m = self.ml_xgb(X_train, X_test, y_train, y_test, stages, code)
		if code == 10: output_m = self.ml_keras(X_train, X_test, y_train, y_test, stages)
		# if code == 11: output_m = self.ml_keras(X_train, X_test, y_train, y_test, stages)
		self.acc_m = '{:02.2f}'.format(np.float(output_m[0]))
		self.p_m = '{:02.2f}'.format(np.float(output_m[1]))
		self.r_m = '{:02.2f}'.format(np.float(output_m[2]))
		self.f1_m = '{:02.2f}'.format(np.float(output_m[3]))
		self.p0_m = '{:02.2f}'.format(np.float(output_m[4]))
		self.p1_m = '{:02.2f}'.format(np.float(output_m[5]))
		self.p2_m = '{:02.2f}'.format(np.float(output_m[6]))
		self.p3_m = '{:02.2f}'.format(np.float(output_m[7]))
		self.p4_m = '{:02.2f}'.format(np.float(output_m[8]))
		self.acc_s = '{:02.2f}'.format(output_s[0])
		self.p_s = '{:02.2f}'.format(output_s[1])
		self.r_s = '{:02.2f}'.format(output_s[2])
		self.f1_s = '{:02.2f}'.format(output_s[3])
		self.p0_s = '{:02.2f}'.format(output_s[4])
		self.p1_s = '{:02.2f}'.format(output_s[5])
		self.p2_s = '{:02.2f}'.format(output_s[6])
		self.p3_s = '{:02.2f}'.format(output_s[7])
		self.p4_s = '{:02.2f}'.format(output_s[8])

		print(code)
		print(self.acc_m + '$\pm$' + self.acc_s, '&',
			self.p_m + '$\pm$' + self.p_s,'&', self.r_m + '$\pm$' + self.r_s,'&',
			self.f1_m + '$\pm$' + self.f1_s,'&', self.p0_m + '$\pm$' + self.p0_s,'&', 
			self.p1_m + '$\pm$' + self.p1_s,'&', self.p2_m + '$\pm$' + self.p2_s,'&', 
			self.p3_m + '$\pm$' + self.p3_s,'&', self.p4_m + '$\pm$' + self.p4_s,'\\\\')

	def ml(self, X_train, X_test, y_train, y_test, stages, code):

		self.X_train = X_train
		self.X_test = X_test
		self.y_train = y_train
		self.y_test = y_test

		## define parameters
		if code ==  1: self.model = LGBMClassifier()#learning_rate = 0.05, num_leaves = 62, n_estimators = 200)
		if code ==  2: self.model = LogisticRegression(max_iter = 1000, solver = 'sag', verbose = 0, tol = 0.01)#(max_iter = 100, verbose = 0, solver = 'sag')
		if code ==  3: self.model = DecisionTreeClassifier(max_depth = 5)
		if code ==  4: self.model = RandomForestClassifier(max_depth = 5, n_estimators = 10, max_features = 1) #(n_estimators = 100 verbose = 0, min_impurity_decrease=0.01)
		if code ==  5: self.model = KNeighborsClassifier(n_neighbors = 3)
		if code ==  6: self.model = MLPClassifier(alpha=1, max_iter = 1000)
		if code ==  7: self.model = GaussianNB()#n_estimators = 50)
		if code ==  8: self.model = AdaBoostClassifier()
#		if code ==  00: self.model = SVC(kernel="linear", C=0.025)


		self.result = self.model.fit(self.X_train, self.y_train)

		## predict Y value vs real Y test value
		self.Y_predict = self.model.predict(self.X_test)
		self.y_predict = self.Y_predict

	    ## Ouputs
		self.acc = '{:02.2f}'.format(metrics.accuracy_score(self.y_test, self.y_predict))
		self.p = '{:02.2f}'.format(metrics.precision_score(self.y_test, self.y_predict, average='macro', zero_division=0))
		self.r = '{:02.2f}'.format(metrics.recall_score(self.y_test, self.y_predict, average='macro', zero_division=0))
		self.f1 = '{:02.2f}'.format(metrics.f1_score(self.y_test, self.y_predict, average='macro', zero_division=0))

		pt = metrics.precision_score(self.y_test, self.y_predict, average=None, zero_division=0)
		self.pp = np.zeros(5)
		self.pp[0] = pt[0]
		if pt.shape[0] > 1: self.pp[1] = pt[1]
		if pt.shape[0] > 2: self.pp[2] = pt[2] 
		if pt.shape[0] > 3: self.pp[3] = pt[3] 
		if pt.shape[0] > 4: self.pp[4] = pt[4] 
		self.pp[0] = '{:02.2f}'.format(self.pp[0])
		self.pp[1] = '{:02.2f}'.format(self.pp[1])
		self.pp[2] = '{:02.2f}'.format(self.pp[2])
		self.pp[3] = '{:02.2f}'.format(self.pp[3])
		self.pp[4] = '{:02.2f}'.format(self.pp[4])

		# print(self.acc,'&',self.p,'&',self.r,'&',self.f1,'&',
			# self.pp[0],'&',self.pp[1],'&',self.pp[2],'&',self.pp[3],'&',self.pp[4],'\\\\')

		return [self.acc, self.p, self.r, self.f1,self.pp[0],self.pp[1],self.pp[2],self.pp[3],self.pp[4]]


	def ml_xgb(self, X_train, X_test, y_train, y_test, stages):

		self.X_train = X_train
		self.X_test = X_test
		self.y_train = y_train
		self.y_test = y_test

		num = 100         # num_round for training iterations
		## define parameters
		dtrain = DMatrix(self.X_train, label=self.y_train)
		dtest = DMatrix(self.X_test, label=self.y_test)
		param = {
			'eval_metric': 'mlogloss',      #'merror' to 'mlogloss'
		    'max_depth': 10,                 # the maximum depth of each tree
		    'eta': 0.3,                     # the training step for each iteration
		    'objective': 'multi:softprob',  # error evaluation for multiclass training
		    'num_class': stages}            # the number of classes that exist in this datset
		num_round = num                     # the number of training iterations
		self.bst = xgb.train(param, dtrain, num_round, evals = [ (dtrain,'train'), (dtest,'test')], early_stopping_rounds = 5, verbose_eval = 0)

		## predict Y value vs real Y test value
		self.Y_predict = self.bst.predict(dtest)
		self.y_predict = np.argmax(self.Y_predict ,axis=1)
	    
	    ## Ouputs
		self.acc = '{:02.2f}'.format(metrics.accuracy_score(self.y_test, self.y_predict))
		self.p = '{:02.2f}'.format(metrics.precision_score(self.y_test, self.y_predict, average='macro', zero_division=0))
		self.r = '{:02.2f}'.format(metrics.recall_score(self.y_test, self.y_predict, average='macro', zero_division=0))
		self.f1 = '{:02.2f}'.format(metrics.f1_score(self.y_test, self.y_predict, average='macro', zero_division=0))

		pt = metrics.precision_score(self.y_test, self.y_predict, average=None, zero_division=0)
		self.pp = np.zeros(5)
		self.pp[0] = pt[0]
		self.pp[1] = pt[1]
		if pt.shape[0] > 2: self.pp[2] = pt[2] 
		if pt.shape[0] > 3: self.pp[3] = pt[3] 
		if pt.shape[0] > 4: self.pp[4] = pt[4] 
		self.pp[0] = '{:02.2f}'.format(self.pp[0])
		self.pp[1] = '{:02.2f}'.format(self.pp[1])
		self.pp[2] = '{:02.2f}'.format(self.pp[2])
		self.pp[3] = '{:02.2f}'.format(self.pp[3])
		self.pp[4] = '{:02.2f}'.format(self.pp[4])

		# print(self.acc,'&',self.p,'&',self.r,'&',self.f1,'&',
		# 	self.pp[0],'&',self.pp[1],'&',self.pp[2],'&',self.pp[3],'&',self.pp[4],'\\\\')

		return [self.acc, self.p, self.r, self.f1,self.pp[0],self.pp[1],self.pp[2],self.pp[3],self.pp[4]]


	def ml_keras(self, X_train, X_test, y_train, y_test, stages):
		self.X_train = X_train
		self.X_test = X_test
		self.y_train = y_train
		self.y_test = y_test

		ep = 100            ## Epochs for model fit
	
		## Preprocess input data
		self.X_train = self.X_train.astype('float32')
		self.X_test = self.X_test.astype('float32')
		xmax = np.nanmax(self.X_train)
		xmin = np.nanmin(self.X_train)
		self.X_train = (self.X_train - xmin) / (xmax - xmin)
		self.X_test = (self.X_test - xmin) / (xmax - xmin)

		## Preprocess class labels
		self.Y_train = np_utils.to_categorical(self.y_train, stages)
		self.Y_test = np_utils.to_categorical(self.y_test, stages)

		## Define model architecture
		model = Sequential()
		model.add(Convolution2D(32, (3, 3), padding='same', input_shape=ishape))
		model.add(Convolution2D(32, (3, 3), padding='same'))
		model.add(Flatten())
		model.add(Dense(128, activation='relu'))
		model.add(Dropout(0.5))
		model.add(Dense(stages, activation='softmax'))

		## Compile model
		model.compile(loss='categorical_crossentropy',
		              optimizer='RMSprop',
		              metrics=[tf.keras.metrics.Precision()])
		             
		## Fit model on training data
		callback = keras.callbacks.EarlyStopping(patience=10)
		history = model.fit(
			self.X_train, self.Y_train, callbacks = [callback], 
			validation_split=0.33, epochs=ep, batch_size=32, verbose=1)

		## Evaluate model on test data
		score = model.evaluate(self.X_test, self.Y_test, verbose = 0)

		## Predict Y value vs real Y test value
		self.Y_predict = model.predict(
			self.X_test, batch_size=32, verbose=0, steps=None)
		self.y_predict = np.argmax(self.Y_predict, axis=1)

	    ## Ouputs
		self.acc = '{:02.2f}'.format(metrics.accuracy_score(self.y_test, self.y_predict))
		self.p = '{:02.2f}'.format(metrics.precision_score(self.y_test, self.y_predict, average='macro', zero_division=0))
		self.r = '{:02.2f}'.format(metrics.recall_score(self.y_test, self.y_predict, average='macro', zero_division=0))
		self.f1 = '{:02.2f}'.format(metrics.f1_score(self.y_test, self.y_predict, average='macro', zero_division=0))

		pt = metrics.precision_score(self.y_test, self.y_predict, average=None, zero_division=0)
		self.pp = np.zeros(5)
		self.pp[0] = pt[0]
		self.pp[1] = pt[1]
		if pt.shape[0] > 2: self.pp[2] = pt[2] 
		if pt.shape[0] > 3: self.pp[3] = pt[3] 
		if pt.shape[0] > 4: self.pp[4] = pt[4] 
		self.pp[0] = '{:02.2f}'.format(self.pp[0])
		self.pp[1] = '{:02.2f}'.format(self.pp[1])
		self.pp[2] = '{:02.2f}'.format(self.pp[2])
		self.pp[3] = '{:02.2f}'.format(self.pp[3])
		self.pp[4] = '{:02.2f}'.format(self.pp[4]) 

		# print(self.acc,'&',self.p,'&',self.r,'&',self.f1,'&',
		# 	self.pp[0],'&',self.pp[1],'&',self.pp[2],'&',self.pp[3],'&',self.pp[4],'\\\\')

		return [self.acc, self.p, self.r, self.f1,self.pp[0],self.pp[1],self.pp[2],self.pp[3],self.pp[4]]

################################################################################

## Open the catalog and read the data
hdul = fits.open('drpallMPL6_180809_pairinfo.cat_.fits')
data = hdul[1].data
header = hdul[1].header
# data = data[0:1000]
# i_sel = (data['FLAG_SF'] == 1)
# i_sel = ((data['PROJSEP'] > -1) & (data['FLAG_SF'] == 1))  
# data = data[i_sel]

plateifu = data['PLATEIFU']     ## plate + IFU number
plate = data['PLATE']           ## plate number
ifu = data['IFU']               ## IFU number
flag_pair = data['FLAG_PAIR']   ## pair information flag
flag_class = data['FLAG_CLASS'] ## merger stage flag
flag_sf = data['FLAG_SF']       ## star-forming galaxy flag
dr = data['PROJSEP']            ## projected separation (kpc assuming h = 0.7)
dv = data['DELTAV']             ## difference in line-of-sight velocity (km/s)

## Define merger flag
i_flag = np.where(flag_class > 0) ## Mergers according to flag_class
flag = flag_class - flag_class
flag[i_flag] = 1

## The relative path of the input data
files1 = '1_sanchez_DR15_190426/' + plate.strip() + '/manga-' + \
	plateifu.strip() + '.Pipe3D.cube.fits.gz'
files2 = '2_MPL6/' + plate.strip() + '/' + ifu.strip() + '.png' 
files3 = '3_images/SDSS_manga-' + plateifu.strip() +'.png' 

################################################################################

# ## Select subsamples
# # i_sel = i_sel[0:100]
# # i_sel = np.where(data['FLAG_PAIR'] > 0)[0]   ## Mergers according to flag_flag
# # i_sel = np.where(data['FLAG_CLASS'] > 0)[0]  ## Mergers according to flag_class
# # i_sel = np.where(data['PROJSEP'] > -1)[0]    ## with NSA redshift
# # i_sel = np.where(data['FLAG_SF'] == 1)[0]    ## Star-forming galaxies
# # i_sel = np.where((data['PROJSEP'] > -1) & (data['FLAG_SF'] == 1))[0]
# # images = images[i_sel,:,:,:]
# # flag_class = flag_class[i_sel]
# # flag_pair = flag_pair[i_sel]
# # flag = flag[i_sel]

# i_999 = np.where(flag_class < 0) 
# flag_class[i_999[0]] = 0

# ## merger and stage flags
# flag_s5 = flag_class

# flag_s3 = flag_class - flag_class #flag_s3 = np.zeros_like(flag_class, dtype=int)
# i_1_3 = np.where((flag_class == 1) | (flag_class == 3)) #stage 1=3
# flag_s3[i_1_3[0]] = 1
# i_2_4 = np.where((flag_class == 2) | (flag_class == 4)) #stage 2=4
# flag_s3[i_2_4[0]] = 2

# flag_s2 = flag              

# # Load image, rotate+flip image, save image from _manga_data.py
# images1 = np.load('img_2d_1.npy') #(4690, 1, 50, 50)
# images2 = np.load('img_2d_2_rs.npy') #(4690, 3, 281, 281)
# images3 = np.load('img_2d_3.npy') #(4690, 2, 1, 1)

# # separate the sample
# my_list = np.array(range(0, 4690)) 
# random.shuffle(my_list)
# ind_train = my_list[0:3126]   #2/3 sample (N=3126)
# ind_test = my_list[3126:4690] #1/3 sample (N=1564)
## ind_train = np.load('npy/ind_train_rot90.npy')
## ind_test = np.load('npy/ind_test_rot90.npy')
# d1 = ind_train.shape
# d1_test = ind_test.shape

# # test
# images1_test = images1[ind_test,:,:,:]
# images2_test = images2[ind_test,:,:,:]
# images3_test = images3[ind_test,:,:,:]
# images1_test = np.reshape(images1_test, (images1_test.shape[0], images1_test.shape[1]*images1_test.shape[2]*images1_test.shape[3]))
# images2_test = np.reshape(images2_test, (images2_test.shape[0], images2_test.shape[1]*images2_test.shape[2]*images2_test.shape[3]))
# images3_test = np.reshape(images3_test, (images3_test.shape[0], images3_test.shape[1]*images3_test.shape[2]*images3_test.shape[3]))
# images_test = np.concatenate((images1_test, images2_test, images3_test), 1)
# flag_s5_test = flag_s5[ind_test]
# flag_s3_test = flag_s3[ind_test]
# flag_s2_test = flag_s2[ind_test]

# # train
# images1 = images1[ind_train,:,:,:]
# images2 = images2[ind_train,:,:,:]
# images3 = images3[ind_train,:,:,:]
# # images1 = np.concatenate((images1, np.rot90(images1,k=1,axes=(2,3)),  ## rotate and combine
# # 	np.rot90(images1,k=2,axes=(2,3)), np.rot90(images1,k=3,axes=(2,3)),
# # 	np.flip(images1,axis=(2,3)), np.flip(np.rot90(images1,k=1,axes=(2,3)),axis=(2,3)), 
# # 	np.flip(np.rot90(images1,k=2,axes=(2,3)),axis=(2,3)), np.flip(np.rot90(images1,k=3,axes=(2,3)),axis=(2,3))), 0)
# # images2 = np.concatenate((images2, np.rot90(images2,k=1,axes=(2,3)), ## rotate and combine
# # 	np.rot90(images2,k=2,axes=(2,3)), np.rot90(images2,k=3,axes=(2,3)),
# # 	np.flip(images2,axis=(2,3)), np.flip(np.rot90(images2,k=1,axes=(2,3)),axis=(2,3)), 
# # 	np.flip(np.rot90(images2,k=2,axes=(2,3)),axis=(2,3)), np.flip(np.rot90(images2,k=3,axes=(2,3)),axis=(2,3))), 0)
# # images3 = np.concatenate((images3,images3,images3,images3,images3,images3,images3,images3))
# images1 = np.concatenate((images1, np.rot90(images1,k=1,axes=(2,3))), 0)
# images2 = np.concatenate((images2, np.rot90(images2,k=1,axes=(2,3))), 0)
# images3 = np.concatenate((images3,images3))
# images1 = np.reshape(images1, (images1.shape[0], images1.shape[1]*images1.shape[2]*images1.shape[3]))
# images2 = np.reshape(images2, (images2.shape[0], images2.shape[1]*images2.shape[2]*images2.shape[3]))
# images3 = np.reshape(images3, (images3.shape[0], images3.shape[1]*images3.shape[2]*images3.shape[3]))
# images = np.concatenate((images1, images2, images3), 1)
# flag_s5 = flag_s5[ind_train]
# flag_s3 = flag_s3[ind_train]
# flag_s2 = flag_s2[ind_train]
# # flag_s5 = np.concatenate((flag_s5,flag_s5,flag_s5,flag_s5,flag_s5,flag_s5,flag_s5,flag_s5),0)
# # flag_s3 = np.concatenate((flag_s3,flag_s3,flag_s3,flag_s3,flag_s3,flag_s3,flag_s3,flag_s3),0)
# # flag_s2 = np.concatenate((flag_s2,flag_s2,flag_s2,flag_s2,flag_s2,flag_s2,flag_s2,flag_s2),0)
# flag_s5 = np.concatenate((flag_s5,flag_s5),0)
# flag_s3 = np.concatenate((flag_s3,flag_s3),0)
# flag_s2 = np.concatenate((flag_s2,flag_s2),0)

# images = images.astype('float16')
# images_test = images_test.astype('float16')
# flag_s5 = flag_s5.astype('float16')
# flag_s5_test = flag_s5_test.astype('float16')
# flag_s3 = flag_s3.astype('float16')
# flag_s3_test = flag_s3_test.astype('float16')
# flag_s2 = flag_s2.astype('float16')
# flag_s2_test = flag_s2_test.astype('float16')

# np.save('npy/f64_rs_rot90_images.npy', images)
# np.save('npy/f64_rs_rot90_images_test.npy', images_test)
# np.save('npy/f64_rs_rot90_flag_s5.npy', flag_s5)
# np.save('npy/f64_rs_rot90_flag_s5_test.npy', flag_s5_test)
# np.save('npy/f64_rs_rot90_flag_s3.npy', flag_s3)
# np.save('npy/f64_rs_rot90_flag_s3_test.npy', flag_s3_test)
# np.save('npy/f64_rs_rot90_flag_s2.npy', flag_s2)
# np.save('npy/f64_rs_rot90_flag_s2_test.npy', flag_s2_test)

images = np.load('npy/f64_rs_rot90_images.npy')
images_test = np.load('npy/f64_rs_rot90_images_test.npy')
flag_s5 = np.load('npy/f64_rs_rot90_flag_s5.npy')
flag_s5_test = np.load('npy/f64_rs_rot90_flag_s5_test.npy')
flag_s3 = np.load('npy/f64_rs_rot90_flag_s3.npy')
flag_s3_test = np.load('npy/f64_rs_rot90_flag_s3_test.npy')
flag_s2 = np.load('npy/f64_rs_rot90_flag_s2.npy')
flag_s2_test = np.load('npy/f64_rs_rot90_flag_s2_test.npy')

################################################################################

# np.save('npy/f64_rs_rot90_images.npy', images)
# np.save('npy/f64_rs_rot90_images_test.npy', images_test)
# np.save('npy/f64_rs_rot90_flag_s5.npy', flag_s5)
# np.save('npy/f64_rs_rot90_flag_s5_test.npy', flag_s5_test)
# np.save('npy/f64_rs_rot90_flag_s3.npy', flag_s3)
# np.save('npy/f64_rs_rot90_flag_s3_test.npy', flag_s3_test)
# np.save('npy/f64_rs_rot90_flag_s2.npy', flag_s2)
# np.save('npy/f64_rs_rot90_flag_s2_test.npy', flag_s2_test)

# images = np.load('npy/f64_rs_rot90_images.npy')
# images_test = np.load('npy/f64_rs_rot90_images_test.npy')
# flag_s5 = np.load('npy/f64_rs_rot90_flag_s5.npy')
# flag_s5_test = np.load('npy/f64_rs_rot90_flag_s5_test.npy')
# flag_s3 = np.load('npy/f64_rs_rot90_flag_s3.npy')
# flag_s3_test = np.load('npy/f64_rs_rot90_flag_s3_test.npy')
# flag_s2 = np.load('npy/f64_rs_rot90_flag_s2.npy')
# flag_s2_test = np.load('npy/f64_rs_rot90_flag_s2_test.npy')

# images = images[:,0:2500] # MaNGA Ha
# images = images[:,2500:81461] # SDSS g
# images = images[:,81461:160422] # SDSS r
# images = images[:,160422:239383] # SDSS i
# images = images[:,239383:239384] # dr
# images = images[:,239384:239385] # dv

# images = np.append(images[:,0:2500],images[:,81461:239385], axis=1) # Haridrdv
# images = np.append(images[:,0:81461],images[:,160422:239385], axis=1) # Hagidrdv
# images = np.append(images[:,0:160422],images[:,239383:239385], axis=1) # Hagrdrdv

# images = images[:,2500:239385] # gridrdv
# images = np.append(images[:,0:2500],images[:,239383:239385], axis=1) # Hadrdv
# images = images[:,0:239383] # Hagri
# images = images[:,239383:239385] # drdv
# images = images[:,239383:239384] # dr
# images = images[:,239384:239385] # dv
# images = images[:,2500:239383] # gri
# images = images[:,2500:81461] # SDSS g
# images = images[:,81461:160422] # SDSS r
# images = images[:,160422:239383] # SDSS i
# images = images[:,0:2500] # Ha

# print("Hagri")
# images = images[:,0:239383] # Hagri
# images_test = images_test[:,0:239383] # Hagri
# flag_s5 = flag_s5[0:239383]
# flag_s5_test = flag_s5_test[0:239383]

################################################################################
## Run the model and print accuracy, precision, recall, and F-1 score
gal0 = GAL('gal0')

# output = gal0.ml(images, images_test, flag_s5, flag_s5_test, 5, 1)
# output = gal0.boot(images, images_test, flag_s5, flag_s5_test, 5, 1) ## boot

# for i in range(1, 9):
# 	output_5 = gal0.ml(images, images_test, flag_s5, flag_s5_test, 5, i) ## codes
# 	output_3 = gal0.ml(images, images_test, flag_s3, flag_s5_test, 3, i) ## codes
# 	output_2 = gal0.ml(images, images_test, flag_s2, flag_s5_test, 2, i) ## codes
# output_5 = gal0.ml_xgb(images, images_test, flag_s5, flag_s5_test, 5) ## XGBoost
# output_3 = gal0.ml_xgb(images, images_test, flag_s3, flag_s3_test, 3) ## XGBoost
# output_2 = gal0.ml_xgb(images, images_test, flag_s2, flag_s2_test, 2) ## XGBoost
# output_5 = gal0.ml_keras(images, images_test, flag_s5, flag_s5_test, 5) ## Keras
# output_3 = gal0.ml_keras(images, images_test, flag_s3, flag_s3_test, 3) ## Keras
# output_2 = gal0.ml_keras(images, images_test, flag_s2, flag_s2_test, 2) ## Keras

print('f64_rs_rot90_images; boot=100')
for i in range(1, 10):
	output_5 = gal0.boot(images, images_test, flag_s5, flag_s5_test, 5, i) ## boot
	output_3 = gal0.boot(images, images_test, flag_s3, flag_s3_test, 3, i) ## boot
	output_2 = gal0.boot(images, images_test, flag_s2, flag_s2_test, 2, i) ## boot

## End of file
################################################################################

