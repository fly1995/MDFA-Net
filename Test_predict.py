#!/usr/bin/python
#coding:utf-8
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import glob
from Network import *
from Loss_function import *
import tensorflow as tf
import time
import SimpleITK as sitk
from scipy.spatial.distance import pdist
from sklearn.metrics import accuracy_score
from sklearn import metrics


def test_predict(file_dir,weight_dir,save_dir,case_num):
    i=0
    imgs = glob.glob(file_dir+case_num )
    for imgname in imgs:
        midname = imgname[imgname.rindex("/") + 1:]
        model = MR_Net()
        model.load_weights(weight_dir)
        img = load_img(file_dir + midname, grayscale=True)
        x = img_to_array(img)
        x = x/255.0
        x= np.expand_dims(x,axis=0)
        preds = model.predict(x)
        #np.save(save_dir+midname[0:-4]+'.npy',preds)
        preds=preds[0]
        preds=array_to_img(preds)
        i += 1
        preds.save(save_dir+midname)
    print(i)


def test_dice(groundtruth_dir,pred_dir,case_num):
    i=0
    hausdorff_sum = 0
    dice_sum = 0
    jaccard_sum =0
    precision_sum = 0
    recall_sum=0
    f1_score_sum=0

    p = glob.glob(groundtruth_dir + case_num)
    for imgname in p:
        midname = imgname[imgname.rindex("/") + 1:]
        gt = load_img(groundtruth_dir+midname, grayscale=True)
        gt = img_to_array(gt)
        gt = gt/255.0
        for x in range(160):
            for y in range(160):
                if gt[:, :, 0][x][y] > 0.5:
                    gt[:, :, 0][x][y] = 1
                else:
                    gt[:, :, 0][x][y] = 0

        pred = load_img(pred_dir + midname, grayscale = True)
        pred = img_to_array(pred)
        pred = pred/255.0
        for x in range(160):
            for y in range(160):
                if pred[:, :, 0][x][y] > 0.5:
                    pred[:, :, 0][x][y] = 1
                else:
                    pred[:, :, 0][x][y] = 0
        i = i+1

        dice = dice_coef(pred, gt)
        dice_sum = dice_sum+dice
        dice_average = dice_sum/i

        ja = jaccard(pred, gt)
        jaccard_sum =jaccard_sum+ja
        ja_average=jaccard_sum/i

        precision = Precision(pred,gt)
        precision_sum = precision_sum+precision
        precision_average =precision_sum/i

        recall = Recall(pred,gt)
        recall_sum = recall_sum+recall
        recall_average =recall_sum/i

        f1 = F1_score(pred,gt)
        f1_score_sum = f1_score_sum+f1
        f1_score_average =f1_score_sum/i
        '''
        with tf.Session() as sess:
            print(midname + ' Dice is: ' + str(sess.run(dice)))
            print(midname + ' Jaccard is: ' + str(sess.run(ja)))
        labelPred = sitk.GetImageFromArray(pred, isVector=False)
        labelTrue = sitk.GetImageFromArray(gt, isVector=False)
        hausdorff = sitk.HausdorffDistanceImageFilter()
        hausdorff.Execute(labelPred,labelTrue)
        hausdorff_sum = hausdorff_sum + hausdorff.GetHausdorffDistance()
        hausdorff_average = hausdorff_sum/i
        print(midname+' Hausdorff is: '+str(hausdorff.GetHausdorffDistance()))
                '''

    with tf.Session() as sess:
        print(str(sess.run(dice_average)),end='\t')
        print(str(sess.run(ja_average)),end='\t')
        print(str(sess.run(precision_average)),end='\t')
        print(str(sess.run(recall_average)),end='\t')
        print(str(sess.run(f1_score_average)),end='\t')


start_time = time.time()

case_num = 'patient36*.jpg'
weight_dir = '/home/cqupt/FLY/2019_Cardiac_Seg_Challenge/model_and_csv/MR_Net/MR_Nett2600.hdf5'
test_file_dir = '/home/cqupt/FLY/2019_Cardiac_Seg_Challenge/test_data/t2/600img/'
groundtruth_dir='/home/cqupt/FLY/2019_Cardiac_Seg_Challenge/test_data/t2/600gt/'
pred_result_dir = '/home/cqupt/FLY/2019_Cardiac_Seg_Challenge/test_result/MR_Net/t2/600/'

test_predict(test_file_dir,weight_dir,pred_result_dir,case_num)
test_dice(groundtruth_dir,pred_result_dir,case_num)
print("%s" % (time.time() - start_time),end='\t')
