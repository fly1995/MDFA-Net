from __future__ import division, print_function
from keras.layers import Input, Conv2D, Concatenate
from keras.models import Model, load_model
from keras.layers import Dense, Dropout,Input, Activation, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, ZeroPadding2D,UpSampling2D
from keras.optimizers import Adam, SGD
from Loss_function import *
from keras.layers import Conv2D, Conv2DTranspose, Input, Cropping2D, add, Dropout, Reshape, Activation
from keras import Model,layers
from keras.layers import Input,Conv2D,BatchNormalization,Activation,Reshape
from keras import optimizers
import tensorflow as tf
from keras.layers import Dropout, Lambda
from keras import backend as K
from keras.layers import Input, average
import tensorflow as tf
import numpy as np

shape1 =(160,160,1)
shape2 =(256,256,1)

#多分类的dice
def dice_coef(y_true, y_pred):
    sum1 = 2*tf.reduce_sum(y_true*y_pred, axis=(0, 1, 2))
    sum2 = tf.reduce_sum(y_true+y_pred, axis=(0, 1, 2))
    dice = (sum1+1)/(sum2+1)
    dice = tf.reduce_mean(dice)
    return dice
def dice_coef_loss(y_true,y_pred):
    return (1-(dice_coef(y_true, y_pred)))


def diceCoeff(gt, pred, smooth=1):
    pred_flat = tf.layers.flatten(pred)
    gt_flat = tf.layers.flatten(gt)
    intersection = K.sum((pred_flat * gt_flat))
    unionset = K.sum(pred_flat) + K.sum(gt_flat)
    score = (2 * intersection + smooth) / (unionset + smooth)
    return score

def myo(y_true,y_pred,):
    class_dice = []
    for i in range(1,2):
        class_dice.append(diceCoeff(y_true[:,:,:,i:i +1 ],y_pred[:,:,:,i:i + 1]))
    mean_dice = sum(class_dice) / len(class_dice)
    return mean_dice
def lv(y_true,y_pred,):
    class_dice = []
    for i in range(2,3):
        class_dice.append(diceCoeff(y_true[:,:,:,i:i +1 ],y_pred[:,:,:,i:i + 1]))
    mean_dice = sum(class_dice) / len(class_dice)
    return mean_dice
def rv(y_true,y_pred,):
    class_dice = []
    for i in range(3,4):
        class_dice.append(diceCoeff(y_true[:,:,:,i:i +1 ],y_pred[:,:,:,i:i + 1]))
    mean_dice = sum(class_dice) / len(class_dice)
    return mean_dice


def conv_bn_relu(input_tensor, flt):
    x = Conv2D(flt, kernel_size=(3, 3), activation='relu', padding='same')(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(flt, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x


def UNet_pp(inputs=Input(shape1)):#Unet
    conv1_1 = conv_bn_relu(inputs, 32)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1_1)

    conv2_1 = conv_bn_relu(pool1, 64)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2_1)

    conv3_1 = conv_bn_relu(pool2, 128)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3_1)

    conv4_1 = conv_bn_relu(pool3, 256)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4_1)

    conv5_1 = conv_bn_relu(pool4, 512)

    up1_2 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv2_1)
    conv1_2 = concatenate([conv1_1, up1_2], 3)
    conv1_2 = conv_bn_relu(conv1_2, 32)

    up2_2 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv3_1)
    conv2_2 = concatenate([conv2_1, up2_2], 3)
    conv2_2 = conv_bn_relu(conv2_2, 64)

    up3_2 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv4_1)
    conv3_2 = concatenate([conv3_1, up3_2], 3)
    conv3_2 = conv_bn_relu(conv3_2, 128)

    up4_2 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5_1)
    conv4_2 = concatenate([conv4_1, up4_2], 3)
    conv4_2 = conv_bn_relu(conv4_2, 256)

    up1_3 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv2_2)
    conv1_3 = concatenate([conv1_1, conv1_2, up1_3], 3)
    conv1_3 = conv_bn_relu(conv1_3, 32)

    up2_3 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv3_2)
    conv2_3 = concatenate([conv2_1, conv2_2, up2_3], 3)
    conv2_3 = conv_bn_relu(conv2_3, 64)

    up3_3 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv4_2)
    conv3_3 = concatenate([conv3_1, conv3_2, up3_3], 3)
    conv3_3 = conv_bn_relu(conv3_3, 128)

    up1_4 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv2_3)
    conv1_4 = concatenate([conv1_1, conv1_2, conv1_3, up1_4], 3)
    conv1_4 = conv_bn_relu(conv1_4, 32)

    up2_4 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv3_3)
    conv2_4 = concatenate([conv2_1, conv2_2, conv2_3, up2_4], 3)
    conv2_4 = conv_bn_relu(conv2_4, 64)

    up1_5 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv2_4)
    conv1_5 = concatenate([conv1_1, conv1_2, conv1_3, conv1_4, up1_5], 3)
    conv1_5 = conv_bn_relu(conv1_5, 32)

    output = Conv2D(4, (1, 1), activation='sigmoid',)(conv1_5)

    model = Model(inputs=[inputs], outputs=[output])
    model.compile(optimizer=Adam(lr=0.0001), loss=[dice_coef_loss], metrics=[dice_coef])
    #model.compile(optimizer=Adam(lr=0.0001), loss=[dice_coef_loss], metrics=[dice_coef, myo, lv, rv])
    return model

def UNet_pp_func(inputs):#Unet
    flt=32
    conv1_1 = conv_bn_relu(inputs, flt)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1_1)

    conv2_1 = conv_bn_relu(pool1, flt*2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2_1)

    conv3_1 = conv_bn_relu(pool2, flt*4)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3_1)

    conv4_1 = conv_bn_relu(pool3, flt*8)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4_1)

    conv5_1 = conv_bn_relu(pool4, flt*16)

    up1_2 = Conv2DTranspose(flt, (2, 2), strides=(2, 2), padding='same')(conv2_1)
    conv1_2 = concatenate([conv1_1, up1_2], 3)
    conv1_2 = conv_bn_relu(conv1_2, flt)

    up2_2 = Conv2DTranspose(flt*2, (2, 2), strides=(2, 2), padding='same')(conv3_1)
    conv2_2 = concatenate([conv2_1, up2_2], 3)
    conv2_2 = conv_bn_relu(conv2_2, flt*2)

    up3_2 = Conv2DTranspose(flt*4, (2, 2), strides=(2, 2), padding='same')(conv4_1)
    conv3_2 = concatenate([conv3_1, up3_2], 3)
    conv3_2 = conv_bn_relu(conv3_2, flt*4)

    up4_2 = Conv2DTranspose(flt*8, (2, 2), strides=(2, 2), padding='same')(conv5_1)
    conv4_2 = concatenate([conv4_1, up4_2], 3)
    conv4_2 = conv_bn_relu(conv4_2,flt*8)

    up1_3 = Conv2DTranspose(flt, (2, 2), strides=(2, 2), padding='same')(conv2_2)
    conv1_3 = concatenate([conv1_1, conv1_2, up1_3], 3)
    conv1_3 = conv_bn_relu(conv1_3, flt)

    up2_3 = Conv2DTranspose(flt*2, (2, 2), strides=(2, 2), padding='same')(conv3_2)
    conv2_3 = concatenate([conv2_1, conv2_2, up2_3], 3)
    conv2_3 = conv_bn_relu(conv2_3, flt*2)

    up3_3 = Conv2DTranspose(flt*4, (2, 2), strides=(2, 2), padding='same')(conv4_2)
    conv3_3 = concatenate([conv3_1, conv3_2, up3_3], 3)
    conv3_3 = conv_bn_relu(conv3_3, flt*4)

    up1_4 = Conv2DTranspose(flt, (2, 2), strides=(2, 2), padding='same')(conv2_3)
    conv1_4 = concatenate([conv1_1, conv1_2, conv1_3, up1_4], 3)
    conv1_4 = conv_bn_relu(conv1_4, flt)

    up2_4 = Conv2DTranspose(flt*2, (2, 2), strides=(2, 2), padding='same')(conv3_3)
    conv2_4 = concatenate([conv2_1, conv2_2, conv2_3, up2_4], 3)
    conv2_4 = conv_bn_relu(conv2_4, flt*2)

    up1_5 = Conv2DTranspose(flt*4, (2, 2), strides=(2, 2), padding='same')(conv2_4)
    conv1_5 = concatenate([conv1_1, conv1_2, conv1_3, conv1_4, up1_5], 3)
    conv1_5 = conv_bn_relu(conv1_5, flt)

    return conv1_5

def D_UNet_pp(inputs=Input(shape1)):
    p1 = UNet_pp_func(inputs)
    p2 = UNet_pp_func(inputs)
    merge = concatenate([p1, p2], axis=-1)
    output = Conv2D(32, (1, 1), activation='relu', )(merge)
    output = Conv2D(1, (1, 1), activation='sigmoid',)(output)

    #output = Conv2D(4, (1, 1), activation='sigmoid', )(merge)
    model = Model(inputs=[inputs], outputs=[output])
    model.compile(optimizer=Adam(lr=0.001), loss=[dice_coef_loss], metrics=[dice_coef])
    #model.compile(optimizer=Adam(lr=0.0001), loss=[dice_coef_loss], metrics=[dice_coef, myo, lv, rv])
    return  model


def U_Net(input_size = shape1):
    flt = 64
    inputs = Input(input_size)

    conv1 = Conv2D(flt, (3, 3), activation='relu', padding='same',kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(flt, (3, 3), activation='relu', padding='same',kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(flt * 2, (3, 3), activation='relu', padding='same',kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(flt * 2, (3, 3), activation='relu', padding='same',kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(flt * 4, (3, 3), activation='relu', padding='same',kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(flt * 4, (3, 3), activation='relu', padding='same',kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(flt * 8, (3, 3), activation='relu', padding='same',kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(flt * 8, (3, 3), activation='relu', padding='same',kernel_initializer='he_normal')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(flt * 16, (3, 3), activation='relu', padding='same',kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(flt * 8, (3, 3), activation='relu', padding='same',kernel_initializer='he_normal')(conv5)

    up6 = concatenate([Conv2DTranspose(flt * 8, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    #up6 = concatenate([UpSampling2D( (2, 2))(conv5), conv4], axis=3)
    conv6 = Conv2D(flt * 4, (3, 3), activation='relu', padding='same',kernel_initializer='he_normal')(up6)
    conv6 = Conv2D(flt * 4, (3, 3), activation='relu', padding='same',kernel_initializer='he_normal')(conv6)

    up7 = concatenate([Conv2DTranspose(flt * 4, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    #up7 = concatenate([UpSampling2D( (2, 2))(conv6), conv3], axis=3)
    conv7 = Conv2D(flt * 2, (3, 3), activation='relu', padding='same',kernel_initializer='he_normal')(up7)
    conv7 = Conv2D(flt * 2, (3, 3), activation='relu', padding='same',kernel_initializer='he_normal')(conv7)

    up8 = concatenate([Conv2DTranspose(flt * 2, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    #up8 = concatenate([UpSampling2D((2, 2))(conv7), conv2], axis=3)
    conv8 = Conv2D(flt, (3, 3), activation='relu', padding='same',kernel_initializer='he_normal')(up8)
    conv8 = Conv2D(flt, (3, 3), activation='relu', padding='same',kernel_initializer='he_normal')(conv8)

    up9 = concatenate([Conv2DTranspose(flt, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    #up9 = concatenate([UpSampling2D((2, 2))(conv8), conv1], axis=3)
    conv9 = Conv2D(flt, (3, 3), activation='relu', padding='same',kernel_initializer='he_normal')(up9)
    conv9 = Conv2D(flt, (3, 3), activation='relu', padding='same',kernel_initializer='he_normal')(conv9)

    conv10 = Conv2D(4, (1, 1), activation='softmax')(conv9)
    #conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])
    model.compile(optimizer=Adam(lr=0.0001), loss=[dice_coef_loss], metrics=[dice_coef])
    #model.compile(optimizer=Adam(lr=0.0001), loss=['binary_crossentropy'], metrics=[dice_coef])

    #model.compile(optimizer=Adam(lr=0.0001), loss=[dice_coef_loss], metrics=[dice_coef,myo,lv,rv])
    #plot_model(model, to_file='./model_pic/UNet.png', show_shapes=True, show_layer_names=False)

    return model

def IB3(input,flt):
    conv1 = Conv2D(flt, (1,1), activation='relu', padding='same')(input)
    conv3 = Conv2D(flt, (3,3), activation='relu', padding='same')(input)
    conv5 = Conv2D(flt, (5,5), activation='relu', padding='same')(input)
    concate = concatenate([conv3,conv5,conv1],axis=3)
    conv = Conv2D(flt, (1,1), activation='relu')(concate)
    output = conv
    return output

def NDD_Net1(x = Input(shape=shape1),features=16, depth=4, padding='same', dilation_rate = 1, kernel_size = (3, 3)):
    inputs = x
    maps = [inputs]
    ib = IB3(inputs,features*2)
    x = Conv2D(features, kernel_size=(3, 3), activation='relu',padding=padding)(ib)
    for n in range(depth):
        x = Conv2D(features, kernel_size, padding=padding)(x)
        x = Conv2D(features, kernel_size, dilation_rate=dilation_rate, padding=padding)(x)
        dilation_rate *= 2
        maps.append(x)
        x = Concatenate(axis=3)(maps)
        x = Activation('relu')(x)

    x = Conv2D(features, kernel_size=(3, 3), activation='relu', padding=padding)(x)
    probabilities = Conv2D(4, kernel_size=(1, 1), activation='softmax')(x)
    model = Model(inputs=inputs, outputs=probabilities)
    model.compile(optimizer=Adam(lr=0.0001), loss=[dice_coef_loss], metrics=[dice_coef,myo,lv,rv])
    #plot_model(model, to_file='../model_pic/NDD_Net_version2.png', show_shapes=True, show_layer_names=False)
    return model

def FCN(input_size=shape1):
    flt=64
    inputs = Input(input_size)
    conv1 = Conv2D(flt,(3,3),activation='relu',padding='same')(inputs)
    conv1 = Conv2D(flt,(3,3),activation='relu',padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(flt*2, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(flt * 2, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(flt * 4, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(flt * 4, (3, 3), activation='relu', padding='same')(conv3)
    conv3 = Conv2D(flt * 4, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(flt * 8, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(flt * 8, (3, 3), activation='relu', padding='same')(conv4)
    conv4 = Conv2D(flt * 8, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    conv5 = Conv2D(flt * 16, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(flt * 16, (3, 3), activation='relu', padding='same')(conv5)
    conv5 = Conv2D(flt * 16, (3, 3), activation='relu', padding='same')(conv5)
    pool5 = MaxPooling2D(pool_size=(2, 2))(conv5)

    conv6 = Conv2D(flt, (5, 5), activation='relu', padding='same')(pool5)
    conv7 = Conv2D(flt, (5, 5), activation='relu', padding='same')(conv6)

    up1 = UpSampling2D(size=(32, 32))(conv7)
    up2 =UpSampling2D(size=(16, 16))(pool4)
    conv_up2 =Conv2D(flt,(1,1))(up2)
    up3 =UpSampling2D(size=(8, 8))(pool3)
    conv_up3=Conv2D(flt,(1,1))(up3)
    add =layers.add([up1,conv_up2,conv_up3])

    output = Conv2D(4,(1,1),activation='softmax')(add)
    model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer=SGD(lr=0.01), loss=[dice_coef_loss], metrics=[dice_coef,myo,lv,rv])
    #plot_model(model, to_file='../model_pic/FCN32.png', show_shapes=True, show_layer_names=False)
    return model


def convblock(m, dim, layername, res=1, drop=0, **kwargs):
    n = Conv2D(filters=dim, name= layername + '_conv1', **kwargs)(m)
    n = BatchNormalization(momentum=0.95, epsilon=0.001)(n)
    n = Dropout(drop)(n) if drop else n
    n = Conv2D(filters=dim, name= layername + '_conv2', **kwargs)(n)
    n = BatchNormalization(momentum=0.95, epsilon=0.001)(n)
    return Concatenate()([m, n]) if res else n


def compute_softmax_weighted_loss(gt, y_pred):
    n_dims=y_pred.shape[-1]
    loss = 0.
    for i in range(n_dims):
        gti = gt[:,:,:,i]
        predi = y_pred[:,:,:,i]
        weighted = 1-(tf.reduce_sum(gti)/tf.reduce_sum(gt))
        focal_loss=1
        loss = loss + -tf.reduce_mean(weighted * gti * focal_loss * tf.log(tf.clip_by_value(predi, 0.005, 1 )))
    return loss


def MSCMR(input_shape=(shape1), num_classes=4, maxpool=True, weights=None):
    kwargs = dict(kernel_size=3, strides=1,activation='relu',padding='same',use_bias=True,kernel_initializer='glorot_uniform',bias_initializer='zeros',bias_regularizer=None,activity_regularizer=None,kernel_constraint=None,bias_constraint=None,trainable=True)
    num_classes = num_classes
    data = Input(shape=input_shape, dtype='float', name='data')
    # encoder
    enconv1 = convblock(data, dim=32, layername='block1', **kwargs)
    pool1 = MaxPooling2D(pool_size=3, strides=2,padding='same',name='pool1')(enconv1) if maxpool \
        else Conv2D(filters=32, strides=2, name='pool1')(enconv1)

    enconv2 = convblock(pool1, dim=64, layername='block2', **kwargs)
    pool2 = MaxPooling2D(pool_size=3, strides=2, padding='same', name='pool2')(enconv2) if maxpool \
        else Conv2D(filters=64, strides=2, name='pool2')(enconv2)

    enconv3 = convblock(pool2, dim=128, layername='block3', **kwargs)
    pool3 = MaxPooling2D(pool_size=3, strides=2, padding='same', name='pool3')(enconv3) if maxpool \
        else Conv2D( filters=128, strides=2, name='pool3')(enconv3)

    enconv4 = convblock(pool3, dim=256, layername='block4', **kwargs)
    pool4 = MaxPooling2D(pool_size=3, strides=2, padding='same', name='pool4')(enconv4) if maxpool \
        else Conv2D(filters=256, strides=2, name='pool4')(enconv4)

    enconv5 = convblock(pool4, dim=512, layername='block5notl', **kwargs)
    # decoder
    up1 = Conv2D(filters=256, kernel_size=1, padding='same', activation='relu',
                 name='up1')(UpSampling2D(size=[2, 2])(enconv5))
    merge1 = Concatenate()([up1,enconv4])
    deconv6 = convblock(merge1, dim=256, layername='deconv6', **kwargs)

    up2 = Conv2D(filters=128, kernel_size=1, padding='same', activation='relu',
                 name='up2')(UpSampling2D(size=[2,2])(deconv6))
    merge2 = Concatenate()([up2,enconv3])
    deconv7 = convblock(merge2, dim=128, layername='deconv7', **kwargs)

    up3 = Conv2D(filters=64, kernel_size=1, padding='same', activation='relu',name='up3')(UpSampling2D(size=[2, 2])(deconv7))
    merge3 = Concatenate()([up3, enconv2])
    deconv8 = convblock(merge3, dim=64, layername='deconv8', **kwargs)

    up4 = Conv2D(filters=32, kernel_size=1, padding='same', activation='relu',name='up4')(UpSampling2D(size=[2, 2])(deconv8))
    merge4 = Concatenate()([up4, enconv1])
    deconv9 = convblock(merge4, dim=32, drop=0.5, layername='deconv9', **kwargs)
    conv10 = Conv2D(filters=num_classes, kernel_size=1, padding='same', activation='relu',name='conv10')(deconv9)
    predictions = Conv2D(filters=num_classes, kernel_size=1, activation='softmax',padding='same', name='predictions')(conv10)
    model = Model(inputs=data, outputs=predictions)

    model.compile(optimizer=Adam(lr=0.0001), loss=[dice_coef_loss], metrics=[dice_coef,myo,lv,rv])

    return model


def mvn(tensor):
    epsilon = 1e-6
    mean = K.mean(tensor, axis=(1, 2), keepdims=True)
    std = K.std(tensor, axis=(1, 2), keepdims=True)
    mvn = (tensor - mean) / (std + epsilon)

    return mvn


def crop(tensors):
    h_dims, w_dims = [], []
    for t in tensors:
        b, h, w, d = K.get_variable_shape(t)
        h_dims.append(h)
        w_dims.append(w)
    crop_h, crop_w = (h_dims[1] - h_dims[0]), (w_dims[1] - w_dims[0])
    rem_h = crop_h % 2
    rem_w = crop_w % 2
    crop_h_dims = (int(crop_h / 2), int(crop_h / 2) + rem_h)
    crop_w_dims = (int(crop_w / 2), int(crop_w / 2) + rem_w)
    cropped = (Cropping2D(cropping=(crop_h_dims, crop_w_dims))(tensors[1]))
    return cropped


def fcn(input_shape=(shape1), num_classes=4):

    kwargs = dict(kernel_size=3,strides=1,activation='relu',padding='same',use_bias=True,kernel_initializer='glorot_uniform',bias_initializer='zeros',bias_regularizer=None,activity_regularizer=None,kernel_constraint=None,bias_constraint=None,trainable=True,)

    data = Input(shape=input_shape, dtype='float', name='data')
    mvn0 = Lambda(mvn, name='mvn0')(data)
    pad = ZeroPadding2D(padding=5, name='pad')(mvn0)

    conv1 = Conv2D(filters=64, name='conv1', **kwargs)(pad)
    mvn1 = Lambda(mvn, name='mvn1')(conv1)

    conv2 = Conv2D(filters=64, name='conv2', **kwargs)(mvn1)
    mvn2 = Lambda(mvn, name='mvn2')(conv2)

    conv3 = Conv2D(filters=64, name='conv3', **kwargs)(mvn2)
    mvn3 = Lambda(mvn, name='mvn3')(conv3)
    pool1 = MaxPooling2D(pool_size=3, strides=2,padding='valid', name='pool1')(mvn3)

    conv4 = Conv2D(filters=128, name='conv4', **kwargs)(pool1)
    mvn4 = Lambda(mvn, name='mvn4')(conv4)

    conv5 = Conv2D(filters=128, name='conv5', **kwargs)(mvn4)
    mvn5 = Lambda(mvn, name='mvn5')(conv5)

    conv6 = Conv2D(filters=128, name='conv6', **kwargs)(mvn5)
    mvn6 = Lambda(mvn, name='mvn6')(conv6)

    conv7 = Conv2D(filters=128, name='conv7', **kwargs)(mvn6)
    mvn7 = Lambda(mvn, name='mvn7')(conv7)
    pool2 = MaxPooling2D(pool_size=3, strides=2,padding='valid', name='pool2')(mvn7)

    conv8 = Conv2D(filters=256, name='conv8', **kwargs)(pool2)
    mvn8 = Lambda(mvn, name='mvn8')(conv8)

    conv9 = Conv2D(filters=256, name='conv9', **kwargs)(mvn8)
    mvn9 = Lambda(mvn, name='mvn9')(conv9)

    conv10 = Conv2D(filters=256, name='conv10', **kwargs)(mvn9)
    mvn10 = Lambda(mvn, name='mvn10')(conv10)

    conv11 = Conv2D(filters=256, name='conv11', **kwargs)(mvn10)
    mvn11 = Lambda(mvn, name='mvn11')(conv11)
    pool3 = MaxPooling2D(pool_size=3, strides=2,padding='valid', name='pool3')(mvn11)
    drop1 = Dropout(rate=0.5, name='drop1')(pool3)

    conv12 = Conv2D(filters=512, name='conv12', **kwargs)(drop1)
    mvn12 = Lambda(mvn, name='mvn12')(conv12)

    conv13 = Conv2D(filters=512, name='conv13', **kwargs)(mvn12)
    mvn13 = Lambda(mvn, name='mvn13')(conv13)

    conv14 = Conv2D(filters=512, name='conv14', **kwargs)(mvn13)
    mvn14 = Lambda(mvn, name='mvn14')(conv14)

    conv15 = Conv2D(filters=512, name='conv15', **kwargs)(mvn14)
    mvn15 = Lambda(mvn, name='mvn15')(conv15)
    drop2 = Dropout(rate=0.5, name='drop2')(mvn15)

    score_conv15 = Conv2D(filters=num_classes, kernel_size=1,strides=1, activation=None, padding='valid',kernel_initializer='glorot_uniform', use_bias=True,name='score_conv15')(drop2)
    upsample1 = Conv2DTranspose(filters=num_classes, kernel_size=3, strides=2, activation=None, padding='valid', kernel_initializer='glorot_uniform', use_bias=False,name='upsample1')(score_conv15)
    score_conv11 = Conv2D(filters=num_classes, kernel_size=1,strides=1, activation=None, padding='valid',kernel_initializer='glorot_uniform', use_bias=True,name='score_conv11')(mvn11)
    crop1 = Lambda(crop, name='crop1')([upsample1, score_conv11])
    fuse_scores1 = average([crop1, upsample1], name='fuse_scores1')

    upsample2 = Conv2DTranspose(filters=num_classes, kernel_size=3,strides=2, activation=None, padding='valid',kernel_initializer='glorot_uniform', use_bias=False,name='upsample2')(fuse_scores1)
    score_conv7 = Conv2D(filters=num_classes, kernel_size=1,strides=1, activation=None, padding='valid',kernel_initializer='glorot_uniform', use_bias=True,name='score_conv7')(mvn7)
    crop2 = Lambda(crop, name='crop2')([upsample2, score_conv7])
    fuse_scores2 = average([crop2, upsample2], name='fuse_scores2')

    upsample3 = Conv2DTranspose(filters=num_classes, kernel_size=3,strides=2, activation=None, padding='valid',kernel_initializer='glorot_uniform', use_bias=False,name='upsample3')(fuse_scores2)
    crop3 = Lambda(crop, name='crop3')([data, upsample3])
    predictions = Conv2D(filters=num_classes, kernel_size=1,strides=1, activation='softmax', padding='valid',kernel_initializer='glorot_uniform', use_bias=True,name='predictions')(crop3)

    model = Model(inputs=data, outputs=predictions)
    sgd = optimizers.Adam(lr=0.001)
    model.compile(optimizer=sgd, loss=[dice_coef_loss],metrics=[dice_coef,myo,lv,rv])
    #model.compile(optimizer=sgd, loss=[dice_coef_loss], metrics=[dice_coef])
   # plot_model(model, to_file='../model_pic/fcn.png', show_shapes=True, show_layer_names=False)
    return model


def MDFA_Net(x = Input(shape=shape1),num_classes=1,features=32, depth=4, padding='same', kernel_size = (3, 3)):
    inputs = x
    maps = [inputs]
    ib = IB3(inputs,features*2)
    x = Conv2D(features, kernel_size=(3, 3), activation='relu',padding=padding,kernel_initializer='glorot_uniform')(ib)
    for n in range(depth):

        x = Conv2D(features, kernel_size, padding=padding,kernel_initializer='glorot_uniform')(x)
        x = Conv2D(features, kernel_size, padding=padding,kernel_initializer='glorot_uniform')(x)
        maps.append(x)
        x = Concatenate(axis=3)(maps)
        x = Activation('relu')(x)

    x1 = Conv2D(features, kernel_size=(3, 3), activation='relu', padding=padding,kernel_initializer='glorot_uniform')(x)
########################################################################################################################
    kwargs = dict(kernel_size=3, strides=1, activation='relu', padding='same', use_bias=True,
                  kernel_initializer='glorot_uniform', bias_initializer='zeros', bias_regularizer=None,
                  activity_regularizer=None, kernel_constraint=None, bias_constraint=None, trainable=True, )


    mvn0 = Lambda(mvn, name='mvn0')(inputs)
    pad = ZeroPadding2D(padding=5, name='pad')(mvn0)

    conv1 = Conv2D(filters=32, name='conv1', **kwargs)(pad)
    mvn1 = Lambda(mvn, name='mvn1')(conv1)

    conv2 = Conv2D(filters=32, name='conv2', **kwargs)(mvn1)
    mvn2 = Lambda(mvn, name='mvn2')(conv2)

    conv3 = Conv2D(filters=32, name='conv3', **kwargs)(mvn2)
    mvn3 = Lambda(mvn, name='mvn3')(conv3)
    pool1 = MaxPooling2D(pool_size=3, strides=2, padding='valid', name='pool1')(mvn3)

    conv4 = Conv2D(filters=64, name='conv4', **kwargs)(pool1)
    mvn4 = Lambda(mvn, name='mvn4')(conv4)

    conv5 = Conv2D(filters=64, name='conv5', **kwargs)(mvn4)
    mvn5 = Lambda(mvn, name='mvn5')(conv5)

    conv6 = Conv2D(filters=64, name='conv6', **kwargs)(mvn5)
    mvn6 = Lambda(mvn, name='mvn6')(conv6)

    conv7 = Conv2D(filters=64, name='conv7', **kwargs)(mvn6)
    mvn7 = Lambda(mvn, name='mvn7')(conv7)
    pool2 = MaxPooling2D(pool_size=3, strides=2, padding='valid', name='pool2')(mvn7)

    conv8 = Conv2D(filters=128, name='conv8', **kwargs)(pool2)
    mvn8 = Lambda(mvn, name='mvn8')(conv8)

    conv9 = Conv2D(filters=128, name='conv9', **kwargs)(mvn8)
    mvn9 = Lambda(mvn, name='mvn9')(conv9)

    conv10 = Conv2D(filters=128, name='conv10', **kwargs)(mvn9)
    mvn10 = Lambda(mvn, name='mvn10')(conv10)

    conv11 = Conv2D(filters=128, name='conv11', **kwargs)(mvn10)
    mvn11 = Lambda(mvn, name='mvn11')(conv11)
    pool3 = MaxPooling2D(pool_size=3, strides=2, padding='valid', name='pool3')(mvn11)
    drop1 = Dropout(rate=0.5, name='drop1')(pool3)

    conv12 = Conv2D(filters=256, name='conv12', **kwargs)(drop1)
    mvn12 = Lambda(mvn, name='mvn12')(conv12)

    conv13 = Conv2D(filters=256, name='conv13', **kwargs)(mvn12)
    mvn13 = Lambda(mvn, name='mvn13')(conv13)

    conv14 = Conv2D(filters=256, name='conv14', **kwargs)(mvn13)
    mvn14 = Lambda(mvn, name='mvn14')(conv14)

    conv15 = Conv2D(filters=256, name='conv15', **kwargs)(mvn14)
    mvn15 = Lambda(mvn, name='mvn15')(conv15)
    drop2 = Dropout(rate=0.5, name='drop2')(mvn15)

    score_conv15 = Conv2D(filters=num_classes, kernel_size=1, strides=1, activation=None, padding='valid',
                          kernel_initializer='glorot_uniform', use_bias=True, name='score_conv15')(drop2)
    upsample1 = Conv2DTranspose(filters=num_classes, kernel_size=3, strides=2, activation=None, padding='valid',
                                kernel_initializer='glorot_uniform', use_bias=False, name='upsample1')(score_conv15)
    score_conv11 = Conv2D(filters=num_classes, kernel_size=1, strides=1, activation=None, padding='valid',
                          kernel_initializer='glorot_uniform', use_bias=True, name='score_conv11')(mvn11)
    crop1 = Lambda(crop, name='crop1')([upsample1, score_conv11])
    fuse_scores1 = average([crop1, upsample1], name='fuse_scores1')

    upsample2 = Conv2DTranspose(filters=num_classes, kernel_size=3, strides=2, activation=None, padding='valid',
                                kernel_initializer='glorot_uniform', use_bias=False, name='upsample2')(fuse_scores1)
    score_conv7 = Conv2D(filters=num_classes, kernel_size=1, strides=1, activation=None, padding='valid',
                         kernel_initializer='glorot_uniform', use_bias=True, name='score_conv7')(mvn7)
    crop2 = Lambda(crop, name='crop2')([upsample2, score_conv7])
    fuse_scores2 = average([crop2, upsample2], name='fuse_scores2')

    upsample3 = Conv2DTranspose(filters=num_classes, kernel_size=3, strides=2, activation=None, padding='valid',
                                kernel_initializer='glorot_uniform', use_bias=False, name='upsample3')(fuse_scores2)
    crop3 = Lambda(crop, name='crop3')([inputs, upsample3])

    output=concatenate([x1,crop3],axis=3)
    output=Conv2D(4, kernel_size=(1, 1), activation='softmax')(output)
    #output = Conv2D(1, kernel_size=(1, 1), activation='sigmoid')(output)
    model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer=Adam(lr=0.001), loss=[dice_coef_loss], metrics=[dice_coef,myo,lv,rv])
    #model.compile(optimizer=Adam(lr=0.001), loss=[dice_coef_loss], metrics=[dice_coef])
    #plot_model(model, to_file='E:/JBHI/model_and_csv/MDFA_Net.png', show_shapes=True, show_layer_names=False)
    model.summary()
    return model


def path2(x = Input(shape=shape1),num_classes=1,features=16, depth=4, padding='same', kernel_size = (3, 3)):
    inputs = x
    kwargs = dict(kernel_size=3, strides=1, activation='relu', padding='same', use_bias=True,
                  kernel_initializer='glorot_uniform', bias_initializer='zeros', bias_regularizer=None,
                  activity_regularizer=None, kernel_constraint=None, bias_constraint=None, trainable=True, )

    mvn0 = Lambda(mvn, name='mvn0')(inputs)
    pad = ZeroPadding2D(padding=5, name='pad')(mvn0)

    conv1 = Conv2D(filters=32, name='conv1', **kwargs)(pad)
    mvn1 = Lambda(mvn, name='mvn1')(conv1)

    conv2 = Conv2D(filters=32, name='conv2', **kwargs)(mvn1)
    mvn2 = Lambda(mvn, name='mvn2')(conv2)

    conv3 = Conv2D(filters=32, name='conv3', **kwargs)(mvn2)
    mvn3 = Lambda(mvn, name='mvn3')(conv3)
    pool1 = MaxPooling2D(pool_size=3, strides=2, padding='valid', name='pool1')(mvn3)

    conv4 = Conv2D(filters=64, name='conv4', **kwargs)(pool1)
    mvn4 = Lambda(mvn, name='mvn4')(conv4)

    conv5 = Conv2D(filters=64, name='conv5', **kwargs)(mvn4)
    mvn5 = Lambda(mvn, name='mvn5')(conv5)

    conv6 = Conv2D(filters=64, name='conv6', **kwargs)(mvn5)
    mvn6 = Lambda(mvn, name='mvn6')(conv6)

    conv7 = Conv2D(filters=64, name='conv7', **kwargs)(mvn6)
    mvn7 = Lambda(mvn, name='mvn7')(conv7)
    pool2 = MaxPooling2D(pool_size=3, strides=2, padding='valid', name='pool2')(mvn7)

    conv8 = Conv2D(filters=128, name='conv8', **kwargs)(pool2)
    mvn8 = Lambda(mvn, name='mvn8')(conv8)

    conv9 = Conv2D(filters=128, name='conv9', **kwargs)(mvn8)
    mvn9 = Lambda(mvn, name='mvn9')(conv9)

    conv10 = Conv2D(filters=128, name='conv10', **kwargs)(mvn9)
    mvn10 = Lambda(mvn, name='mvn10')(conv10)

    conv11 = Conv2D(filters=128, name='conv11', **kwargs)(mvn10)
    mvn11 = Lambda(mvn, name='mvn11')(conv11)
    pool3 = MaxPooling2D(pool_size=3, strides=2, padding='valid', name='pool3')(mvn11)
    drop1 = Dropout(rate=0.5, name='drop1')(pool3)

    conv12 = Conv2D(filters=256, name='conv12', **kwargs)(drop1)
    mvn12 = Lambda(mvn, name='mvn12')(conv12)

    conv13 = Conv2D(filters=256, name='conv13', **kwargs)(mvn12)
    mvn13 = Lambda(mvn, name='mvn13')(conv13)

    conv14 = Conv2D(filters=256, name='conv14', **kwargs)(mvn13)
    mvn14 = Lambda(mvn, name='mvn14')(conv14)

    conv15 = Conv2D(filters=256, name='conv15', **kwargs)(mvn14)
    mvn15 = Lambda(mvn, name='mvn15')(conv15)
    drop2 = Dropout(rate=0.5, name='drop2')(mvn15)

    score_conv15 = Conv2D(filters=num_classes, kernel_size=1, strides=1, activation=None, padding='valid',
                          kernel_initializer='glorot_uniform', use_bias=True, name='score_conv15')(drop2)
    upsample1 = Conv2DTranspose(filters=num_classes, kernel_size=3, strides=2, activation=None, padding='valid',
                                kernel_initializer='glorot_uniform', use_bias=False, name='upsample1')(score_conv15)
    score_conv11 = Conv2D(filters=num_classes, kernel_size=1, strides=1, activation=None, padding='valid',
                          kernel_initializer='glorot_uniform', use_bias=True, name='score_conv11')(mvn11)
    crop1 = Lambda(crop, name='crop1')([upsample1, score_conv11])
    fuse_scores1 = average([crop1, upsample1], name='fuse_scores1')

    upsample2 = Conv2DTranspose(filters=num_classes, kernel_size=3, strides=2, activation=None, padding='valid',
                                kernel_initializer='glorot_uniform', use_bias=False, name='upsample2')(fuse_scores1)
    score_conv7 = Conv2D(filters=num_classes, kernel_size=1, strides=1, activation=None, padding='valid',
                         kernel_initializer='glorot_uniform', use_bias=True, name='score_conv7')(mvn7)
    crop2 = Lambda(crop, name='crop2')([upsample2, score_conv7])
    fuse_scores2 = average([crop2, upsample2], name='fuse_scores2')

    upsample3 = Conv2DTranspose(filters=num_classes, kernel_size=3, strides=2, activation=None, padding='valid',
                                kernel_initializer='glorot_uniform', use_bias=False, name='upsample3')(fuse_scores2)
    crop3 = Lambda(crop, name='crop3')([inputs, upsample3])

    output=Conv2D(4, kernel_size=(1, 1), activation='softmax')(crop3)

    model = Model(inputs=inputs, outputs=output)

    model.compile(optimizer=Adam(lr=0.0001), loss=[dice_coef_loss], metrics=[dice_coef,myo,lv,rv])
    #plot_model(model, to_file='E:/JBHI/model_and_csv/MDFA_Net.png', show_shapes=True, show_layer_names=False)
    return model

def MR_Net(input_size = shape1):
    flt = 64
    inputs = Input(input_size)

    conv1 = Conv2D(flt, (3, 3), activation='relu', padding='same')(inputs)
    x1 = conv1
    conv1 = Conv2D(flt, (3, 3), activation='relu', padding='same')(conv1)
    conv1 = Conv2D(flt, (3, 3), activation='relu', padding='same')(conv1)
    conv1 = layers.add([x1, conv1])
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    #muti_level convolution

    conv11 = Conv2D(flt*2, (1, 1), activation='relu', padding='same')(conv1)
    conv12 = Conv2D(flt*2, (3, 3), activation='relu', padding='same')(conv1)
    conv13 = Conv2D(flt*2, (5, 5), activation='relu', padding='same')(conv1)
    conv11 = concatenate([Conv2D(flt * 2, (1, 1), strides=(1, 1), padding='same')(conv12), conv11], axis=3)
    conv11 = concatenate([Conv2D(flt * 2, (1, 1), strides=(1, 1), padding='same')(conv13), conv11], axis=3)

    conv11 = Conv2D(flt, (1, 1), activation='relu', padding='same')(conv11)

    conv2 = Conv2D(flt*2, (3, 3), activation='relu', padding='same')(pool1)
    x2 = conv2
    conv2 = Conv2D(flt*2, (3, 3), activation='relu', padding='same')(conv2)
    conv2 = Conv2D(flt * 2, (3, 3), activation='relu', padding='same')(conv2)
    conv2 = layers.add([x2, conv2])
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(flt*4, (3, 3), activation='relu', padding='same')(pool2)
    x3 = conv3
    conv3 = Conv2D(flt*4, (3, 3), activation='relu', padding='same')(conv3)
    conv3 = Conv2D(flt * 4, (3, 3), activation='relu', padding='same')(conv3)
    conv3 = layers.add([x3, conv3])
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(flt*8, (3, 3), activation='relu', padding='same')(pool3)
    x4 = conv4
    conv4 = Conv2D(flt*8, (3, 3), activation='relu', padding='same')(conv4)
    conv4 = Conv2D(flt * 8, (3, 3), activation='relu', padding='same')(conv4)
    conv4 = layers.add([x4, conv4])
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(flt*16, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(flt * 8, (3, 3), activation='relu', padding='same')(conv5)
    x5 = pool4
    conv5 = layers.add([x5, conv5])

    up6 = concatenate([Conv2DTranspose(flt*8, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    conv6 = Conv2D(flt*4, (3, 3), activation='relu', padding='same')(up6)
    x6 = conv6
    conv6 = Conv2D(flt*4, (3, 3), activation='relu', padding='same')(conv6)
    conv6 = Conv2D(flt * 4, (3, 3), activation='relu', padding='same')(conv6)
    conv6 = layers.add([x6, conv6])

    up7 = concatenate([Conv2DTranspose(flt*4, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = Conv2D(flt*2, (3, 3), activation='relu', padding='same')(up7)
    x7 = conv7
    conv7 = Conv2D(flt*2, (3, 3), activation='relu', padding='same')(conv7)
    conv7 = Conv2D(flt * 2, (3, 3), activation='relu', padding='same')(conv7)
    conv7 = layers.add([x7, conv7])

    up8 = concatenate([Conv2DTranspose(flt*2, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = Conv2D(flt, (3, 3), activation='relu', padding='same')(up8)
    x8 = conv8
    conv8 = Conv2D(flt, (3, 3), activation='relu', padding='same')(conv8)
    conv8 = Conv2D(flt, (3, 3), activation='relu', padding='same')(conv8)
    conv8 = layers.add([x8, conv8])

    up9 = concatenate([Conv2DTranspose(flt, (2, 2), strides=(2, 2), padding='same')(conv8), conv11], axis=3)
    conv9 = Conv2D(flt, (3, 3), activation='relu', padding='same')(up9)
    x9 = conv9
    conv9 = Conv2D(flt, (3, 3), activation='relu', padding='same')(conv9)
    conv9 = Conv2D(flt, (3, 3), activation='relu', padding='same')(conv9)
    conv9 = layers.add([x9, conv9])

    conv10 = Conv2D(4, (1, 1), activation='softmax')(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])

    model.compile(optimizer=Adam(lr=0.0001), loss=dice_coef_loss, metrics=[dice_coef,myo,lv,rv])

    return model
def Lambda1(x = Input(shape=shape1),num_classes=1,features=16, depth=4, padding='same', kernel_size = (3, 3)):
    inputs = x
    maps = [inputs]
    ib = IB3(inputs,features*2)
    x = Conv2D(features, kernel_size=(3, 3), activation='relu',padding=padding,kernel_initializer='glorot_uniform')(ib)

    for n in range(depth):

        x = Conv2D(features, kernel_size, padding=padding,kernel_initializer='glorot_uniform')(x)
        x = Conv2D(features, kernel_size, padding=padding,kernel_initializer='glorot_uniform')(x)
        maps.append(x)
        x = Concatenate(axis=3)(maps)
        x = Activation('relu')(x)

    x1 = Conv2D(features, kernel_size=(3, 3), activation='relu', padding=padding,kernel_initializer='glorot_uniform')(x)

########################################################################################################################
    kwargs = dict(kernel_size=3, strides=1, activation='relu', padding='same', use_bias=True,
                  kernel_initializer='glorot_uniform', bias_initializer='zeros', bias_regularizer=None,
                  activity_regularizer=None, kernel_constraint=None, bias_constraint=None, trainable=True, )

    pad = ZeroPadding2D(padding=5, name='pad')(inputs)
    conv1 = Conv2D(filters=32, name='conv1', **kwargs)(pad)
    conv2 = Conv2D(filters=32, name='conv2', **kwargs)(conv1)
    conv3 = Conv2D(filters=32, name='conv3', **kwargs)(conv2)
    pool1 = MaxPooling2D(pool_size=3, strides=2, padding='valid', name='pool1')(conv3)
    conv4 = Conv2D(filters=64, name='conv4', **kwargs)(pool1)
    conv5 = Conv2D(filters=64, name='conv5', **kwargs)(conv4)
    conv6 = Conv2D(filters=64, name='conv6', **kwargs)(conv5)
    conv7 = Conv2D(filters=64, name='conv7', **kwargs)(conv6)
    pool2 = MaxPooling2D(pool_size=3, strides=2, padding='valid', name='pool2')(conv7)
    conv8 = Conv2D(filters=128, name='conv8', **kwargs)(pool2)
    conv9 = Conv2D(filters=128, name='conv9', **kwargs)(conv8)
    conv10 = Conv2D(filters=128, name='conv10', **kwargs)(conv9)
    conv11 = Conv2D(filters=128, name='conv11', **kwargs)(conv10)
    pool3 = MaxPooling2D(pool_size=3, strides=2, padding='valid', name='pool3')(conv11)
    drop1 = Dropout(rate=0.5, name='drop1')(pool3)
    conv12 = Conv2D(filters=256, name='conv12', **kwargs)(drop1)
    conv13 = Conv2D(filters=256, name='conv13', **kwargs)(conv12)
    conv14 = Conv2D(filters=256, name='conv14', **kwargs)(conv13)
    conv15 = Conv2D(filters=256, name='conv15', **kwargs)(conv14)
    drop2 = Dropout(rate=0.5, name='drop2')(conv15)
    score_conv15 = Conv2D(filters=num_classes, kernel_size=1, strides=1, activation=None, padding='valid',
                          kernel_initializer='glorot_uniform', use_bias=True, name='score_conv15')(drop2)
    upsample1 = Conv2DTranspose(filters=num_classes, kernel_size=3, strides=2, activation=None, padding='valid',
                                kernel_initializer='glorot_uniform', use_bias=False, name='upsample1')(score_conv15)
    score_conv11 = Conv2D(filters=num_classes, kernel_size=1, strides=1, activation=None, padding='valid',
                          kernel_initializer='glorot_uniform', use_bias=True, name='score_conv11')(conv11)
    crop1 = crop([upsample1, score_conv11])
    fuse_scores1 = average([crop1, upsample1], name='fuse_scores1')

    upsample2 = Conv2DTranspose(filters=num_classes, kernel_size=3, strides=2, activation=None, padding='valid',
                                kernel_initializer='glorot_uniform', use_bias=False, name='upsample2')(fuse_scores1)
    score_conv7 = Conv2D(filters=num_classes, kernel_size=1, strides=1, activation=None, padding='valid',
                         kernel_initializer='glorot_uniform', use_bias=True, name='score_conv7')(conv7)
    crop2 = crop([upsample2, score_conv7])
    fuse_scores2 = average([crop2, upsample2], name='fuse_scores2')

    upsample3 = Conv2DTranspose(filters=num_classes, kernel_size=3, strides=2, activation=None, padding='valid',
                                kernel_initializer='glorot_uniform', use_bias=False, name='upsample3')(fuse_scores2)
    crop3 = crop([inputs, upsample3])

    output=concatenate([x1,crop3],axis=3)
    output=Conv2D(4, kernel_size=(1, 1), activation='softmax')(output)

    model = Model(inputs=inputs, outputs=output)

    model.compile(optimizer=Adam(lr=0.0001), loss=[dice_coef_loss], metrics=[dice_coef,myo,lv,rv])
    #plot_model(model, to_file='E:/JBHI/model_and_csv/MDFA_Net.png', show_shapes=True, show_layer_names=False)
    return model
