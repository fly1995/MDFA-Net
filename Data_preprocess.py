import numpy as np
import glob
import nibabel
import scipy.ndimage
import cv2
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

#读取原始nii数据的格式：每个病例中元素的最大值、最小值以及病例的形状
def read_data_format(file_dir):
    imgname = glob.glob(file_dir + '/*' + '.nii.gz')
    for file_name in imgname:
        midname = file_name[file_name.rindex("/") + 1:]
        img = nibabel.load(file_dir + midname).get_data()
        img_max = np.amax(img)
        img_min = np.amin(img)
        print('%s: max is %d, min is %d' %(midname, img_max, img_min))
        print(img.shape)

#read_data_format('/home/cqupt/FLY/2019_Cardiac_Seg_Challenge/original_data/nii/t2gt/')


#将有标签的切片筛选出来(初步筛选)
def select_slice(file_dir, label_dir,save_file_dir, save_label_dir):
    imgname = glob.glob(file_dir+'/*'+'.nii.gz')

    for file_name in imgname:
        j = 0
        midname = file_name[file_name.rindex("/") + 1:]
        img = nibabel.load(file_dir + midname).get_data()
        label = nibabel.load(label_dir + midname[0:-7]+'_manual.nii.gz').get_data()
        print(img.shape)

        newimg = np.zeros((img.shape[0],img.shape[1],img.shape[2]))
        newlabel = np.zeros((img.shape[0],img.shape[1],label.shape[2]))

        for i in range(label.shape[2]):#筛选当前切片总和大于0的
            if np.sum(label[:, :, i:i+1]) >0:
                newimg[:, :, j:j+1] = img[:, :, i:i+1]
                newlabel[:, :, j:j+1] = label[:, :, i:i+1]
                j = j + 1

        finalimg = newimg[:, :, 0: j]
        finallabel = newlabel[:, :, 0: j]
        print(finalimg.shape)

        np.save(save_file_dir + midname[0:-7] + '.npy', finalimg)
        np.save(save_label_dir + midname[0:-7] + '.npy', finallabel)

'''
select_slice('/home/cqupt/FLY/2019_Cardiac_Seg_Challenge/original_data/npy/lge_npy/train/',
                 '/home/cqupt/FLY/2019_Cardiac_Seg_Challenge/original_data/npy/lgegt_npy/train/',
                '/home/cqupt/FLY/2019_Cardiac_Seg_Challenge/train_data/lge/',
                 '/home/cqupt/FLY/2019_Cardiac_Seg_Challenge/train_data/lgegt/')
'''

#将标签（心脏、左心室、右心室分离到三个numpy中）
def Label_separation(file_dir,save_label_dir200,save_label_dir500,save_label_dir600):
    imgname = glob.glob(file_dir + '/*' + '.npy')
    for file_name in imgname:
        midname = file_name[file_name.rindex("/") + 1:]
        label = np.load(file_dir+midname)
        label200 = np.zeros((label.shape[0], label.shape[1], label.shape[2]))
        label500 = np.zeros((label.shape[0], label.shape[1], label.shape[2]))
        label600 = np.zeros((label.shape[0], label.shape[1], label.shape[2]))
        for i in range(label.shape[0]):
            for j in range(label.shape[1]):
                for k in range(label.shape[2]):
                    if label[i][j][k] == 200:
                        label200[i][j][k] = 200
                    elif label[i][j][k] == 500:
                        label500[i][j][k] = 500
                    elif label[i][j][k] == 600:
                        label600[i][j][k] = 600
                    else:
                        label200[i][j][k] == 0
                        label500[i][j][k] == 0
                        label600[i][j][k] == 0

        np.save(save_label_dir200 + midname, label200)
        np.save(save_label_dir500 + midname, label500)
        np.save(save_label_dir600 + midname, label600)
''''
file_dir='/home/cqupt/FLY/2019_Cardiac_Seg_Challenge/original_data/npy/lgegt_npy/train/'
save_label_dir200='/home/cqupt/FLY/2019_Cardiac_Seg_Challenge/train_data/lgegt/200/'
save_label_dir500='/home/cqupt/FLY/2019_Cardiac_Seg_Challenge/train_data/lgegt/500/'
save_label_dir600='/home/cqupt/FLY/2019_Cardiac_Seg_Challenge/train_data/lgegt/600/'
Label_separation(file_dir,save_label_dir200,save_label_dir500,save_label_dir600)
'''


#将生成的numpy转换成jpg并保存下来
def print_npy_to_jpg(file_dir,save_dir):
  imgname = glob.glob(file_dir + '*' + '.npy')
  for file_name in imgname:
    midname = file_name[file_name.rindex("/") + 1:]
    npy = np.load(file_dir+midname)

    for i in range(npy.shape[2]):
      img = npy[:,:,i]
      img = np.expand_dims(img,axis=2)
      img = array_to_img(img)
      img.save(save_dir+midname[0:-4]+'%d.jpg'%i)
'''
file_dir='/home/cqupt/FLY/2019_Cardiac_Seg_Challenge/train_data/lge/200/'
save_dir='/home/cqupt/FLY/2019_Cardiac_Seg_Challenge/train_data/lge/200/'
print_npy_to_jpg(file_dir,save_dir)
'''


def resize_and_crop(file_dir):
    imgname = glob.glob(file_dir + '/*' + '.jpg')
    for file_name in imgname:
        midname = file_name[file_name.rindex("/") + 1:]
        img = load_img(file_dir + midname, grayscale=True)

        img_resize = img.resize((256,256))
        img_resize.save(file_dir+midname)

        img_resize = img_to_array(img_resize)
        img_crop = img_resize[48:208, 48:208]
        img_crop = array_to_img(img_crop)
        img_crop.save(file_dir+midname)


#resize_and_crop('/home/cqupt/FLY/2019_Cardiac_Seg_Challenge/train_data/lgegt/600/')


#第二次筛选有标记的切片，根据mask进行筛选，左心室、右心室分别筛选
def select_img_accord_to_name(mask_file_dir,img_file_dir,save_file_dir):#筛选有标记的切片，先手动筛出没有标记的，然后再执行该代码
    imgs = glob.glob(mask_file_dir+'*.jpg')
    for imgname in imgs:
        midname= imgname[imgname.rindex('/')+1:]
        img=cv2.imread(img_file_dir+midname)
        cv2.imwrite(save_file_dir+midname,img)
'''
mask_file_dir = '/home/cqupt/FLY/2019_Cardiac_Seg_Challenge/train_data/lge/200/'
img_file_dir = '/home/cqupt/FLY/2019_Cardiac_Seg_Challenge/train_data/lgegt/200/'
save_file_dir = '/home/cqupt/桌面/test/'
select_img_accord_to_name(mask_file_dir,img_file_dir,save_file_dir)
'''

def rotate(image, angle, center=None, scale=1.0):#旋转图片
    (h, w) = image.shape[:2]
    # If no rotation center is specified, the center of the image is set as the rotation center
    if center is None:
        center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated


def data_aug(file_dir,save_path):#数据增广函数
    i=0
    imgs = glob.glob(file_dir + "/*." + 'jpg')

    for imgname in imgs:
        midname = imgname[imgname.rindex("/") + 1:]
        print(midname)
        image = load_img(file_dir + "/" + midname, grayscale=True)
        image = img_to_array(image)
        rotated2 = rotate(image, 90)
        cv2.imwrite(save_path + midname[0:-4] + '_ro90.jpg', rotated2)
        rotated4 = rotate(image, 180)
        cv2.imwrite(save_path + midname[0:-4] + '_ro180.jpg', rotated4)
        rotated6 = rotate(image, 270)
        cv2.imwrite(save_path + midname[0:-4] + '_ro270.jpg', rotated6)
        rotated8 = rotate(image, 360)
        cv2.imwrite(save_path + midname, rotated8)
        sp = cv2.flip(image, 1, dst=None)  # 水平镜像
        cv2.imwrite(save_path + midname[0:-4] + '_sp.jpg', sp)
        cz = cv2.flip(image, 0, dst=None)  # 垂直镜像
        cv2.imwrite(save_path + midname[0:-4] + '_cz.jpg', cz)
        dj = cv2.flip(image, -1, dst=None)  # 对角镜像
        cv2.imwrite(save_path + midname[0:-4] + '_dj.jpg', dj)

        i += 1
    print(i)

'''
file_dir='/home/cqupt/FLY/2019_Cardiac_Seg_Challenge/train_data/lgegt/600/'
save_path='/home/cqupt/FLY/2019_Cardiac_Seg_Challenge/train_data/lgegt/600/'
data_aug(file_dir,save_path)
'''

def jpg2npy(file_dir_image,file_dir_mask):#将所有图片加载 到一个numpy中，制作训练集,现在是三通道
    imgs = glob.glob(file_dir_mask + '*.jpg')
    i = 0
    imgdatas_image = np.zeros((389, 160,160, 1))
    imgdatas_mask = np.zeros((389, 160,160, 1))
    for imgname in imgs:
        midname = imgname[imgname.rindex('/') + 1:]
        print(midname)

        img = load_img(file_dir_image + midname.replace('_manual',''), grayscale=True)#保证img和mask文件名相同
        img = img_to_array(img)
        img = img/255.0
        imgdatas_image[i, :, :, :] = img

        img_mask = load_img(file_dir_mask + midname, grayscale=True)
        img_mask = img_to_array(img_mask)
        img_mask = img_mask/255

        for x in range(160):
            for y in range(160):
                if img_mask[x][y].any() >0.5:
                    img_mask[x][y] = 1
                else:
                    img_mask[x][y] = 0

        imgdatas_mask[i, :, :, :] = img_mask

        i += 1
    print(i)
    np.save(file_dir_image+'lge_image.npy', imgdatas_image)
    np.save(file_dir_mask+'lge_mask.npy', imgdatas_mask)

'''
file_dir_image = '/home/cqupt/FLY/2019_Cardiac_Seg_Challenge/train_data/lge/600/'
file_dir_mask = '/home/cqupt/FLY/2019_Cardiac_Seg_Challenge/train_data/lgegt/600/'
jpg2npy(file_dir_image,file_dir_mask)
'''


#将生成的numpy转换成jpg并保存下来
def test_npy(file_dir,save_dir):
    npy = np.load(file_dir)
    for i in range(5):
      img = npy[i,:,:,:]
      img = array_to_img(img)
      img.save(save_dir+'%d.jpg'%i)
'''
file_dir='/home/cqupt/FLY/2019_Cardiac_Seg_Challenge/train_data/lgegt/200/lge200_aug/mask.npy'
save_dir='/home/cqupt/桌面/TEST/'
test_npy(file_dir, save_dir)
'''

'''
x1= np.load('/home/cqupt/FLY/2019_Cardiac_Seg_Challenge/train_data/lgegt/600/lge600_aug_fold1/mask.npy')
print(x1.shape)
x2 = np.load('/home/cqupt/FLY/2019_Cardiac_Seg_Challenge/train_data/lgegt/600/lge600_aug_fold2/mask.npy')
x3= np.load('/home/cqupt/FLY/2019_Cardiac_Seg_Challenge/train_data/lgegt/600/lge600_aug_fold3/mask.npy')
x4 = np.load('/home/cqupt/FLY/2019_Cardiac_Seg_Challenge/train_data/lgegt/600/lge600_aug_fold4/mask.npy')
x5 =np.load('/home/cqupt/FLY/2019_Cardiac_Seg_Challenge/train_data/lgegt/600/lge600_aug_fold5/mask.npy')

x = np.concatenate([x1,x2,x3,x4,x5],axis=0)
print(x.shape)
np.save("//home/cqupt/FLY/2019_Cardiac_Seg_Challenge/train_data/lgegt/600/mask.npy",x)
'''
