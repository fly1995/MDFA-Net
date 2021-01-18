#!/usr/bin/python
#coding:utf-8
from keras import backend as K
import tensorflow as tf
import math
import scipy.spatial
import scipy.ndimage.morphology
from hausdorff import  hausdorff_distance
import numpy as np

#二分类的dice
def dice2_coef(y_true, y_pred):
    smooth = 1.0
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) /(K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice2_coef_loss(y_true,y_pred):
    return (1-(dice2_coef(y_true, y_pred)))

#多分类的dice
def multi_dice_coef(y_true, y_pred):
    sum1 = 2*tf.reduce_sum(y_true*y_pred, axis=(0, 1, 2))
    sum2 = tf.reduce_sum(y_true+y_pred, axis=(0, 1, 2))
    dice = (sum1+0.00001)/(sum2+0.00001)
    dice = tf.reduce_mean(dice)
    return dice


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

def Extrenal_metric(gt,pred):
    result1 = []  # dice
    result2 = []  # jaccard
    result3 = []  # hausdorff
    for i in range(gt.shape[0]):
        x1 = dice2_coef(gt[i:i+1], pred[i:i+1])
        result1.append(x1)
        x2 = jaccard(gt[i:i+1], pred[i:i+1])
        result2.append(x2)
    with tf.Session() as sess:
        y1 = sess.run(result1)
        y2 = sess.run(result2)
    print("Dice,Jaccard,Hausdorff")
    print('%.2f±%.2f[%.2f,%.2f]'%(sum(y1)/len(y1)*100, np.std(y1)*100, min(y1)*100, max(y1)*100), end='\t')
    print('%.2f±%.2f[%.2f,%.2f]'%(sum(y2)/len(y2)*100, np.std(y2)*100, min(y2)*100, max(y2)*100), end='\t')
    for i in range(gt.shape[0]):
        a = gt[i, :, :, 0]
        b = pred[i, :, :, 0]
        x3 = hausdorff_distance(a,b,distance='manhattan')#distance="euclidean",distance="chebyshev",distance="cosine"
        result3.append(x3)
    y3 = result3
    print('%.2f±%.2f[%.2f,%.2f]'%(sum(y3)/len(y3), np.std(y3), min(y3), max(y3)), end='\t')

    '''
    surface = Surface(gt, pred, connectivity=2)
    assd = surface.get_average_symmetric_surface_distance()
    print("ASSD: %.2f"%assd)
    '''

class Surface(object):
    # The edge points of the mask object.
    __mask_edge_points = None
    # The edge points of the reference object.
    __reference_edge_points = None
    # The nearest neighbours distances between mask and reference edge points.
    __mask_reference_nn = None
    # The nearest neighbours distances between reference and mask edge points.
    __reference_mask_nn = None
    # Distances of the two objects surface points.
    __distance_matrix = None
    def __init__(self, mask, reference, physical_voxel_spacing=[1, 1, 1], mask_offset=[0, 0, 0],reference_offset=[0, 0, 0], connectivity=1):
        self.connectivity = connectivity
        # compute edge images
        mask_edge_image = self.compute_contour(mask)
        reference_edge_image = self.compute_contour(reference)
        mask_pts = mask_edge_image.nonzero()
        mask_edge_points = list(zip(mask_pts[0], mask_pts[1], mask_pts[2]))
        reference_pts = reference_edge_image.nonzero()
        reference_edge_points = list(zip(reference_pts[0], reference_pts[1], reference_pts[2]))
        # check if there is actually an object present
        if 0 >= len(mask_edge_points):
            raise Exception('The mask image does not seem to contain an object.')
        if 0 >= len(reference_edge_points):
            raise Exception('The reference image does not seem to contain an object.')
        # add offsets to the voxels positions and multiply with physical voxel spacing
        # to get the real positions in millimeters
        physical_voxel_spacing = scipy.array(physical_voxel_spacing)
        mask_edge_points = scipy.array(mask_edge_points, dtype='float64')
        mask_edge_points += scipy.array(mask_offset)
        mask_edge_points *= physical_voxel_spacing
        reference_edge_points = scipy.array(reference_edge_points, dtype='float64')
        reference_edge_points += scipy.array(reference_offset)
        reference_edge_points *= physical_voxel_spacing
        # set member vars
        self.__mask_edge_points = mask_edge_points
        self.__reference_edge_points = reference_edge_points
    def get_maximum_symmetric_surface_distance(self):
        # Get the maximum of the nearest neighbour distances
        A_B_distance = self.get_mask_reference_nn().max()
        B_A_distance = self.get_reference_mask_nn().max()
        # compute result and return
        return min(A_B_distance, B_A_distance)
    def get_root_mean_square_symmetric_surface_distance(self):
        # get object sizes
        mask_surface_size = len(self.get_mask_edge_points())
        reference_surface_sice = len(self.get_reference_edge_points())
        # get minimal nearest neighbours distances
        A_B_distances = self.get_mask_reference_nn()
        B_A_distances = self.get_reference_mask_nn()
        # square the distances
        A_B_distances_sqrt = A_B_distances * A_B_distances
        B_A_distances_sqrt = B_A_distances * B_A_distances
        # sum the minimal distances
        A_B_distances_sum = A_B_distances_sqrt.sum()
        B_A_distances_sum = B_A_distances_sqrt.sum()
        # compute result and return
        return math.sqrt(1. / (mask_surface_size + reference_surface_sice)) * math.sqrt(
            A_B_distances_sum + B_A_distances_sum)
    def get_average_symmetric_surface_distance(self):
        # get object sizes
        mask_surface_size = len(self.get_mask_edge_points())
        reference_surface_sice = len(self.get_reference_edge_points())
        # get minimal nearest neighbours distances
        A_B_distances = self.get_mask_reference_nn()
        B_A_distances = self.get_reference_mask_nn()
        # sum the minimal distances
        A_B_distances = A_B_distances.sum()
        B_A_distances = B_A_distances.sum()
        # compute result and return
        return 1. / (mask_surface_size + reference_surface_sice) * (A_B_distances + B_A_distances)
    def get_mask_reference_nn(self):
        # Note: see note for @see get_reference_mask_nn
        if None == self.__mask_reference_nn:
            tree = scipy.spatial.cKDTree(self.get_mask_edge_points())
            self.__mask_reference_nn, _ = tree.query(self.get_reference_edge_points())
        return self.__mask_reference_nn
    def get_reference_mask_nn(self):
        if self.__reference_mask_nn is None:
            tree = scipy.spatial.cKDTree(self.get_reference_edge_points())
            self.__reference_mask_nn, _ = tree.query(self.get_mask_edge_points())
        return self.__reference_mask_nn
    def get_mask_edge_points(self):
        return self.__mask_edge_points
    def get_reference_edge_points(self):
        return self.__reference_edge_points
    def compute_contour(self, array):
        footprint = scipy.ndimage.morphology.generate_binary_structure(array.ndim, self.connectivity)
        # create an erode version of the array
        erode_array = scipy.ndimage.morphology.binary_erosion(array, footprint)
        array = array.astype(bool)
        # xor the erode_array with the original and return
        return array ^ erode_array
