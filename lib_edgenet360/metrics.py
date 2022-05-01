import numpy as np
from tensorflow.keras import backend as K


def comp_iou(_y_true, _y_pred):
    smooth = K.epsilon()

    weights = K.max(_y_true-1, axis=-1)

    comp_flags = K.cast(K.not_equal(weights, K.variable(0.0)),'float32')


    y_true = K.cast(K.clip(K.argmax(_y_true),0,1),'float32') * comp_flags

    y_pred = K.cast(K.clip(K.argmax(_y_pred),0,1),'float32') * comp_flags

    #axis = (np.array(range(K.ndim(y_pred)-1))+1)*-1

    inter = K.sum(K.flatten(y_true) * K.flatten(y_pred))

    union = K.sum(K.clip((K.flatten(y_true) + K.flatten(y_pred)), 0, 1))

    comp_iou = (inter+smooth) / (union + smooth)

    return comp_iou


def seg_iou(_y_true, _y_pred):
    smooth = K.epsilon()

    weights = K.max(_y_true - 1, axis=-1)

    seg_flags = K.cast(K.not_equal(weights, K.variable(0.0)),'float32')

    seg_iou=0

    for cl in range(0,11):

        y_true = K.cast(K.equal(K.argmax(_y_true, axis=-1), K.variable(cl+1,'int64')),'float32') * seg_flags
        y_pred = K.cast(K.equal(K.argmax(_y_pred, axis=-1), K.variable(cl+1,'int64')),'float32') * seg_flags

        inter = K.sum(K.flatten(y_true) * K.flatten(y_pred))
        union = K.sum(K.clip(K.flatten(y_true) + K.flatten(y_pred), 0, 1))

        seg_iou += (inter+smooth/2)/(union+smooth) #non existing classes scores 0.5

    return seg_iou/11

def combined_iou(_y_true, _y_pred):
    return K.sqrt(K.pow(seg_iou(_y_true, _y_pred),2) + K.pow(comp_iou(_y_true, _y_pred),2))/1.4142

def nclasses(_y_true, _y_pred):

    y_pred = K.cast(K.argmax(_y_pred),'float32')

    nc, idx = tf.unique(K.flatten(y_pred))

    nc = tf.size(nc)

    ret = nc

    return ret

def minclass(_y_true, _y_pred):

    y_pred = K.cast(K.argmax(_y_pred),'float32')

    nc, idx = tf.unique(K.flatten(y_pred))

    nc = K.min(nc)

    return nc

def maxclass(_y_true, _y_pred):

    y_pred = K.cast(K.argmax(_y_pred),'float32')

    nc, idx = tf.unique(K.flatten(y_pred))

    nc = K.max(nc)

    return nc

import tensorflow as tf


def comp_iou_np(_y_true, _y_pred, vol):
    #smooth = 0.00000001

    #only occluded
    flags=   np.array((vol<0)&(vol>=-1),dtype='float32')

    y_true = np.clip(np.argmax(_y_true, axis=-1),0,1) * flags

    y_pred = np.clip(np.argmax(_y_pred, axis=-1),0,1) * flags

    axis = (-1,-2,-3)

    inter = np.sum(y_true.flatten() * y_pred.flatten())
    union = np.sum(np.clip((y_true.flatten() + y_pred.flatten()), 0, 1))

    tp = inter
    fp = np.sum(np.array(y_true.flatten() == 0) & (y_pred.flatten() > 0))
    fn = np.sum((y_true.flatten() > 0) & (y_pred.flatten() == 0))

    return inter, union, tp, fp, fn


def comp_iou_stanford(_y_true, _y_pred, surface):
    #smooth = 0.00000001

    #only occluded
    flags=   np.array((surface==0) & np.argmax(_y_true, axis=-1)>0,dtype='float32')

    y_true = np.clip(np.argmax(_y_true, axis=-1),0,1) * flags

    y_pred = np.clip(np.argmax(_y_pred, axis=-1),0,1) * flags

    inter = np.sum(y_true.flatten() * y_pred.flatten())
    union = np.sum(np.clip((y_true.flatten() + y_pred.flatten()), 0, 1))

    tp = inter
    fp = np.sum(np.array(y_true.flatten() == 0) & (y_pred.flatten() > 0))
    fn = np.sum((y_true.flatten() > 0) & (y_pred.flatten() == 0))

    return inter, union, tp, fp, fn


def seg_iou_stanford(_y_true, _y_pred, cl):
    #smooth = 0.00000001

    #occluded and visible
    flags=   np.array(np.argmax(_y_true, axis=-1)>0,dtype='float32')

    y_true = np.array(np.argmax(_y_true, axis=-1) == cl,dtype='float32') * flags
    y_pred = np.array(np.argmax(_y_pred, axis=-1) == cl,dtype='float32') * flags

    axis = (-1,-2,-3)

    inter = np.sum(y_true.flatten() * y_pred.flatten())
    union = np.sum(np.clip((y_true.flatten() + y_pred.flatten()), 0, 1))

    return inter, union



def seg_iou_np(_y_true, _y_pred, vol, cl):
    #smooth = 0.00000001

    #occluded and visible
    flags=   np.array((abs(vol)<1)|(vol==-1),dtype='float32')

    y_true = np.array(np.argmax(_y_true, axis=-1) == cl,dtype='float32') * flags
    y_pred = np.array(np.argmax(_y_pred, axis=-1) == cl,dtype='float32') * flags

    axis = (-1,-2,-3)

    inter = np.sum(y_true.flatten() * y_pred.flatten())
    union = np.sum(np.clip((y_true.flatten() + y_pred.flatten()), 0, 1))

    return inter, union


def comp_iou_np2(_y_true, _y_pred, vol):
    #smooth = 0.00000001


    #only occluded
    flags=   np.array((vol>-1) & (vol<-.5),dtype='float32')

    y_true = np.clip(np.argmax(_y_true, axis=-1),0,1) * flags

    y_pred = np.clip(np.argmax(_y_pred, axis=-1),0,1) * flags

    axis = (-1,-2,-3)

    inter = np.sum(y_true.flatten() * y_pred.flatten())
    union = np.sum(np.clip((y_true.flatten() + y_pred.flatten()), 0, 1))

    return inter, union


def seg_iou_np2(_y_true, _y_pred, vol, cl):
    #smooth = 0.00000001

    occ = np.array(np.argmax(_y_true, axis=-1), dtype='float32')
    #occluded and visible
    flags=   np.array(((vol>-1)&(vol < -0.2))|(((vol==-1)|(vol>=0.2)) & (occ>0)),dtype='float32')

    y_true = np.array(np.argmax(_y_true, axis=-1) == cl, dtype='float32') * flags
    y_pred = np.array(np.argmax(_y_pred, axis=-1) == cl, dtype='float32') * flags

    axis = (-1,-2,-3)

    inter = np.sum(y_true.flatten() * y_pred.flatten())
    union = np.sum(np.clip((y_true.flatten() + y_pred.flatten()), 0, 1))

    return inter, union

