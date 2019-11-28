#coding=utf-8
import os
import string


def re_file():
    path =("/home/cqupt/FLY/C0LET2_nii45_for_challenge19/train_gt_original")
    #filelist = os.listdir(path) #该文件夹下所有的文件（包括文件夹）
    for root, dirs,files in os.walk(path):
        for  name in files: #遍历所有文件
            pathname = os.path.splitext(os.path.join(root,name))
            if(pathname[1] != ".txt" and pathname[1] != ".png" and pathname[1] != ".exe"and pathname[1] != ".bin"): #删除不是.txt的文件
                os.remove(os.path.join(root,name))


def re_name():
    path = ("/home/cqupt/FLY/2019_Cardiac_Seg_Challenge/train_data/lge/600/")
    #filelist = os.listdir(path) #该文件夹下所有的文件（包括文件夹）
    for root, dirs, files in os.walk(path):
        for name in files:  # 遍历所有文件
            pos = name.find("_")  # 把文件带有.tab.out的字符删除
            if (pos == -1):
                continue
            newname = name[0:pos] + name[pos + 1:]
            os.rename(os.path.join(root, name), os.path.join(root, newname))

re_name()


