import glob
from keras.preprocessing.image import array_to_img, img_to_array, load_img
import numpy as np
import nibabel
import SimpleITK as sitk
import tensorflow as tf

file_dir='/home/cqupt/FLY/2019_Cardiac_Seg_Challenge/test_data/c0/merge/'
save_dir=file_dir

'''
img = glob.glob(file_dir + '*.jpg')
for imgname in img:
    midname = imgname[imgname.rindex("/") + 1:]
    img = load_img(file_dir+midname,grayscale=True)
    img_npy = img_to_array(img)

    img_merge = np.zeros((160, 160, 3))
    img_merge[:, :, 0:1] = img_npy
    img_merge[:, :, 1:2] = img_npy
    img_merge[:, :, 2:3] = img_npy
    img_merge = sitk.GetImageFromArray(img_merge)

    sitk.WriteImage(img_merge,save_dir+midname[0:-4] + '.nii')
'''

img = glob.glob(file_dir + '*.jpg')
for imgname in img:
    midname = imgname[imgname.rindex("/") + 1:]
    img = load_img(file_dir+midname)

    img_npy = img_to_array(img)
    '''
    img_npy =img_npy/255
    for i in range(img_npy.shape[0]):
        for j in range(img_npy.shape[1]):
            if img_npy[i][j].any()>0.5:
                img_npy[i][j]=1
            else:
                img_npy[i][j]=0
    '''
    img_npy = sitk.GetImageFromArray(img_npy)
    sitk.WriteImage(img_npy,save_dir+midname[0:-4] + '.nii')


