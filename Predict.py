import numpy as np
from Network import *
from Loss_function import *
import tensorflow as tf
session = tf.Session()
train = np.load('E:\\D1_Paper_Cardiac_Segmentation\\Test_data\\LGE_NPY\\lge200.npy')

 # 200 rv
i1 = train[0:166, :, :, :]
i2 = train[166:333, :, :, :]
i3 = train[333:498, :, :, :]
i4 = train[498:666, :, :, :]#517
'''
 # 500 rv
i1 = train[0:168, :, :, :]
i2 = train[168:336, :, :, :]
i3 = train[336:504, :, :, :]
i4 = train[504:672, :, :, :]

 # 600 rv
i1 = train[0:127, :, :, :]
i2 = train[127:254, :, :, :]
i3 = train[254:382, :, :, :]
i4 = train[382:511, :, :, :]
'''
model = UNet_pp()
model.load_weights('E:\\D1_Paper_Cardiac_Segmentation\\code\\UNet_pp_lge200f1.hdf5')
preds = model.predict(i1)
preds = preds.astype('float64')
preds[preds >= 0.5] = 1
preds[preds < 0.5]=0
np.save('E:\\D1_Paper_Cardiac_Segmentation\\Test_results\\UNet_pp\\lge\\200\\f1\\200f1.npy',preds)
