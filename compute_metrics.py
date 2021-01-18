import numpy as np
from Loss_function import *
import tensorflow as tf
from  Metrics import *
from hausdorff import  hausdorff_distance
' Lambda1'

gt = np.load('E:\\D1_Paper_Cardiac_Segmentation\\Test_data\\LGE_NPY\\lge500.npy')
print(gt.shape)
pred1 = np.load("E:\\D1_Paper_Cardiac_Segmentation\\Test_results\\UNet_pp\\lge\\500\\f1\\500f1.npy")
pred2 = np.load("E:\\D1_Paper_Cardiac_Segmentation\\Test_results\\UNet_pp\\lge\\500\\f2\\500f2.npy")
pred3 = np.load("E:\\D1_Paper_Cardiac_Segmentation\\Test_results\\UNet_pp\\lge\\500\\f3\\500f3.npy")
pred4 = np.load("E:\\D1_Paper_Cardiac_Segmentation\\Test_results\\UNet_pp\\lge\\500\\f4\\500f4.npy")
print(pred1.shape)
print(pred2.shape)
print(pred3.shape)
print(pred4.shape)
pred = np.concatenate([pred1, pred2, pred3, pred4], axis=0)
print(pred.shape)

result1 = []#dice
resullge = []#jaccard
result3 = []#hausdorff

for i in range(gt.shape[0]):
    x1 = dice2_coef(gt[i:i+1], pred[i:i+1])
    result1.append(x1)
    x2 = jaccard(gt[i:i+1], pred[i:i+1])
    resullge.append(x2)
with tf.Session() as sess:
    y1 = sess.run(result1)
    y2 = sess.run(resullge)
np.savetxt('E:\\D1_Paper_Cardiac_Segmentation\\Test_results\\UNet_pp\\lge\\500\\dice.csv',y1)
np.savetxt('E:\\D1_Paper_Cardiac_Segmentation\\Test_results\\UNet_pp\\lge\\500\\jaccard.csv',y2)

print("Dice:mean std min max",end='\t')
mean = sum(y1)/len(y1)
print(str(mean),end='\t')
std = np.std(y1)
print(str(std),end='\t')
print(str(min(y1)),end='\t')
print(str(max(y1)),end='\n')

print("Jaccard:mean std min max",end='\t')
mean = sum(y2)/len(y2)
print(str(mean),end='\t')
std = np.std(y2)
print(str(std),end='\t')
print(str(min(y2)),end='\t')
print(str(max(y2)),end='\n')

result3=[]
for i in range(gt.shape[0]):
    a = gt[i,:,:,0]
    b = pred[i,:,:,0]
    x3 = hausdorff_distance(a,b,distance='manhattan')#distance="euclidean",distance="chebyshev",distance="cosine"
    result3.append(x3)
y3 = result3
np.savetxt('E:\\D1_Paper_Cardiac_Segmentation\\Test_results\\UNet_pp\\lge\\500\\hausdorff.csv',y3)

print("Hausdorff:mean std min max",end='\t')
mean = sum(y3)/len(y3)
print(str(mean),end='\t')
std = np.std(y3)
print(str(std),end='\t')
print(str(min(y3)),end='\t')
print(str(max(y3)),end='\n')

surface = Surface(gt, pred, connectivity=2)
assd = surface.get_average_symmetric_surface_distance()
print("assd is :"+str(assd))


