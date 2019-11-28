from keras.callbacks import ModelCheckpoint, LearningRateScheduler,EarlyStopping,CSVLogger,ReduceLROnPlateau
import keras.backend.tensorflow_backend as KTF
import tensorflow as tf
from keras.models import *
import keras.backend as K
from Network import *
from Loss_function import *


sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
KTF.set_session(tf.Session(config=tf.ConfigProto(device_count={'gpu': 0})))
K.set_image_data_format('channels_last')


def train():

    train = np.load('/home/cqupt/FLY/2019_Cardiac_Seg_Challenge/train_data/c0/200/image.npy')
    train_mask = np.load('/home/cqupt/FLY/2019_Cardiac_Seg_Challenge/train_data/c0gt/200/mask.npy')

    earlystop = EarlyStopping(monitor='val_loss', patience=50, verbose=1, mode='auto')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=50, mode='auto')
    #model = load_model('/home/cqupt/FLY/2019_Cardiac_Seg_Challenge/code/model_and_csv/(UNet)t2200.hdf5', compile=True,custom_objects={'dice_coef_loss': dice_coef_loss,'dice_coef':dice_coef})

    #model = get_unet()
    model = NDD_Net2()
    #model = FCN32()
    #model = MSCMR()
    #model = fcn()

    csv_logger = CSVLogger('../model_and_csv/NDD_Net/version2.0/(NDD_Net)c0200.csv')
    model_checkpoint = ModelCheckpoint(filepath='../model_and_csv/NDD_Net/version2.0/(NDD_Net)c0200.hdf5', monitor='loss', verbose=1, save_best_only=True, mode='min')
    model.fit(train,train_mask,batch_size=32,  validation_split=0.1, epochs=1000, verbose=1, shuffle=True, callbacks=[model_checkpoint, csv_logger,earlystop,reduce_lr])


if __name__ == '__main__':
    train()
