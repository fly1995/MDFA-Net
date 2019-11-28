#!/usr/bin/python
#coding:utf-8
import numpy as np
from keras import backend as K
import tensorflow as tf
import math
import cv2
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import jieba

pi=3


def dice_coef(y_true, y_pred):
    smooth = 1.0
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) /(K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true,y_pred):
    return (1-(dice_coef(y_true, y_pred)))


def jaccard(y_true, y_pred):
    smooth = 1.0
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + smooth)/(K.sum(y_true_f) + K.sum(y_pred_f) - intersection + smooth)


def Precision(predicted, actual):
    TP = tf.count_nonzero(predicted * actual)
    FP = tf.count_nonzero(predicted * (actual - 1))
    tf_precision = (TP+1) / (TP + FP+1)
    return tf_precision


def Recall(predicted, actual):
    TP = tf.count_nonzero(predicted * actual)
    FN = tf.count_nonzero((predicted - 1) * actual)
    tf_recall = (TP+1) / (TP + FN+1)
    return tf_recall


def F1_score(predicted, actual):
    TP = tf.count_nonzero(predicted * actual)
    FP = tf.count_nonzero(predicted * (actual - 1))
    FN = tf.count_nonzero((predicted - 1) * actual)
    tf_precision = (TP+1) / (TP + FP+1)
    tf_recall =( TP+1) / (TP + FN+1)
    tf_f1_score = 2 * tf_precision * tf_recall / (tf_precision + tf_recall)
    return tf_f1_score


def circle_r(y_true, y_pred):
    y_true_s = 0
    y_pred_s = 0
    for i in range(96):
        for j in range(96):
            if y_true[:,:,:,0][i][j] != 0:
                y_true_s = y_true_s + 1
            elif y_pred[:,:,:,0][i][j] !=0:
                y_pred_s = y_pred_s + 1

    y_true_r = math.sqrt(y_true_s/pi)
    y_pred_r = math.sqrt(y_pred_s/pi)
    result = y_true_r - y_pred_r
    return  result

def Centroid(y_true, y_pred):

    true_sum_i=0
    true_sum_j=0
    true_num=0

    pred_sum_i=0
    pred_sum_j=0
    pred_num=0

    for i in range(96):
        for j in range(96):
            if y_true[:, :, :, 0][i][j] != 0:
                true_num=true_num+1
                true_sum_i=true_sum_i+i
                true_sum_j=true_sum_j+j

    for i in range(96):
        for j in range(96):
            if y_pred[:, :, :, 0][i][j] != 0:
                pred_num=pred_num+1
                pred_sum_i=pred_sum_i+i
                pred_sum_j=pred_sum_j+j

    true_average_i=true_sum_i/true_num
    true_average_j=true_sum_j/true_num

    pred_average_i=pred_sum_i/pred_num
    pred_average_j=pred_sum_j/pred_num

    true_min=0
    true_max=0

    pred_min=0
    pred_max=0

    for i in range(96):
        for j in range(96):
            if y_true[:, :, :, 0][i][j] != 0 and math.sqrt(pow((true_average_i-i),2)+pow((true_average_j-j),2))<true_min:
                true_min=math.sqrt(pow((true_average_i-i),2)+pow((true_average_j-j),2))

            elif y_true[:, :, :, 0][i][j] != 0 and math.sqrt(pow((true_average_i-i),2)+pow((true_average_j-j),2))>true_max:
                true_max=math.sqrt(pow((true_average_i-i),2)+pow((true_average_j-j),2))


    for i in range(96):
        for j in range(96):
            if y_pred[:, :, :, 0][i][j] != 0 and math.sqrt(pow((pred_average_i-i),2)+pow((pred_average_j-j),2))<pred_min:
                pred_min=math.sqrt(pow((pred_average_i-i),2)+pow((pred_average_j-j),2))

            elif y_pred[:, :, :, 0][i][j] != 0 and math.sqrt(pow((pred_average_i-i),2)+pow((pred_average_j-j),2))>pred_max:
                pred_max=math.sqrt(pow((pred_average_i-i),2)+pow((pred_average_j-j),2))


    result1 = (true_min-pred_min)+(true_max-pred_max)
    result2 = (true_max-pred_max)
    return  result1


def sum_loss(y_true,y_pred):
    sum = 0.8*dice_coef(y_true,y_pred) + 0.2*Centroid(y_true,y_pred)
    return sum


def focal_loss1(y_true, y_pred):
    gamma = 2
    alpha = 0.85
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))

    pt_1 = K.clip(pt_1, 1e-3, .999)
    pt_0 = K.clip(pt_0, 1e-3, .999)

    return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.sum(
        (1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))


# focal loss with multi label
def focal_loss2(classes_num, gamma=2., alpha=.25, e=0.8320):
    # classes_num contains sample number of each classes
    def focal_loss_fixed(target_tensor, prediction_tensor):
        '''
        prediction_tensor is the output tensor with shape [None, 100], where 100 is the number of classes
        target_tensor is the label tensor, same shape as predcition_tensor
        '''
        import tensorflow as tf
        from tensorflow.python.ops import array_ops
        from keras import backend as K

        #1# get focal loss with no balanced weight which presented in paper function (4)
        zeros = array_ops.zeros_like(prediction_tensor, dtype=prediction_tensor.dtype)
        one_minus_p = array_ops.where(tf.greater(target_tensor,zeros), target_tensor - prediction_tensor, zeros)
        FT = -1 * (one_minus_p ** gamma) * tf.log(tf.clip_by_value(prediction_tensor, 1e-8, 1.0))

        #2# get balanced weight alpha
        classes_weight = array_ops.zeros_like(prediction_tensor, dtype=prediction_tensor.dtype)

        total_num = float(sum(classes_num))
        classes_w_t1 = [ total_num / ff for ff in classes_num ]
        sum_ = sum(classes_w_t1)
        classes_w_t2 = [ ff/sum_ for ff in classes_w_t1 ]   #scale
        classes_w_tensor = tf.convert_to_tensor(classes_w_t2, dtype=prediction_tensor.dtype)
        classes_weight += classes_w_tensor

        alpha = array_ops.where(tf.greater(target_tensor, zeros), classes_weight, zeros)

        #3# get balanced focal loss
        balanced_fl = alpha * FT
        balanced_fl = tf.reduce_sum(balanced_fl)

        #4# add other op to prevent overfit
        # reference : https://spaces.ac.cn/archives/4493
        nb_classes = len(classes_num)
        fianal_loss = (1-e) * balanced_fl + e * K.categorical_crossentropy(K.ones_like(prediction_tensor)/nb_classes, prediction_tensor)

        return fianal_loss
    return focal_loss_fixed
