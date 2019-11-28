
from keras.preprocessing.image import  array_to_img, img_to_array, load_img
import numpy as np
import glob


file_dir200='/home/cqupt/FLY/2019_Cardiac_Seg_Challenge/test_result/NDD_Net_result/version2/t2/200/'
file_dir500='/home/cqupt/FLY/2019_Cardiac_Seg_Challenge/test_result/NDD_Net_result/version2/t2/500/'
file_dir600='/home/cqupt/FLY/2019_Cardiac_Seg_Challenge/test_result/NDD_Net_result/version2/t2/600/'
save_merge_dir='/home/cqupt/FLY/2019_Cardiac_Seg_Challenge/test_result/NDD_Net_result/version2/t2/merge/'



imgname200 = glob.glob(file_dir200 + '/*' + '.jpg')
imgname200 = [x.split('/')[-1] for x in imgname200]
imgname500 = glob.glob(file_dir500 + '/*' + '.jpg')
imgname500 = [x.split('/')[-1] for x in imgname500]
imgname600 = glob.glob(file_dir600 + '/*' + '.jpg')
imgname600 = [x.split('/')[-1] for x in imgname600]

out = list(set(imgname200).intersection(imgname500,imgname600))
print(len(out))

for patient_name in out:

    img200= load_img(file_dir200+patient_name,grayscale=True)
    img200 = img_to_array(img200)
    img500=load_img(file_dir500+patient_name,grayscale=True)
    img500 = img_to_array(img500)
    img600=load_img(file_dir600+patient_name,grayscale=True)
    img600 = img_to_array(img600)

    img_merge= np.zeros((160,160,3))
    img_merge[:,:,0:1]=img200
    img_merge[:,:,1:2]=img500
    img_merge[:,:,2:3]=img600
    img_merge =array_to_img(img_merge)

    img_merge.save(save_merge_dir+patient_name)
