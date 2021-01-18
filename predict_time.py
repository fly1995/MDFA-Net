from Network import *
from Loss_function import *
import numpy as np
from Loss_function import *
import tensorflow as tf
from Metrics import *
from keras.preprocessing.image import array_to_img
session = tf.Session()
import time
def normolize(input):
    epsilon = 1e-6
    mean = np.mean(input)
    std = np.std(input)
    return (input-mean)/(std+epsilon)


t0 = time.clock()

#img = np.load('E:\D1_Paper_Cardiac_Segmentation\External_validation\\test_data\\c0.npy')#(102, 256, 256, 1) 不加map的测试
img = np.load('E:\D1_Paper_Cardiac_Segmentation\External_validation\map_results\D_UNet_pp\\c0_weighted_map\\c0_weighted_map.npy')#(102, 256, 256, 1)
img = normolize(img)#important
label = np.load('E:\D1_Paper_Cardiac_Segmentation\External_validation\\test_data\\label.npy')#(102, 256, 256, 6)
label = label.astype('float64')
model = MDFA_Net()
model.load_weights('E:\D1_Paper_Cardiac_Segmentation\code\\(c0)MDFA_Net_wm.hdf5')
preds4 = model.predict(img)

t1 = time.clock()
print("Total running time: %s s" % (str(t1 - t0)))
