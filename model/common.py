import numpy as np
import tensorflow as tf


DIV2K_RGB_MEAN = np.array([0.4488, 0.4371, 0.4040]) * 255


def resolve_single(model, lr, nbit=16):
    return resolve16(model, tf.expand_dims(lr, axis=0), nbit=nbit)[0]

def resolve(model, lr_batch):
    lr_batch = tf.cast(lr_batch, tf.float16)
    sr_batch = model(lr_batch)
    sr_batch = tf.clip_by_value(sr_batch, 0, 255)
    sr_batch = tf.round(sr_batch)
    sr_batch = tf.cast(sr_batch, tf.uint8)
    return sr_batch

def resolve16(model, lr_batch, nbit=16):
    if nbit==8:
        casttype=tf.uint8
    elif nbit==16:
        casttype=tf.uint16
    else:
        print("Wrong number of bits")
        exit()
    lr_batch = tf.cast(lr_batch, tf.float32)
    sr_batch = model(lr_batch)
    sr_batch = tf.clip_by_value(sr_batch, 0, 2**nbit-1)
    sr_batch = tf.round(sr_batch)
    sr_batch = tf.cast(sr_batch, casttype)
    return sr_batch

def evaluate(model, dataset, nbit=8):
    psnr_values = []
    for lr, hr in dataset:
        sr = resolve16(model, lr, nbit=nbit) #hack
        if lr.shape[-1]==1:
            sr = sr[..., 0, None]
#        psnr_value = psnr16(hr, sr)[0]
        psnr_value = psnr(hr, sr, nbit=nbit)[0]
        psnr_values.append(psnr_value)
    return tf.reduce_mean(psnr_values)


# ---------------------------------------
#  Normalization
# ---------------------------------------
#def normalize(x, rgb_mean=DIV2K_RGB_MEAN, nbit=16):
#    if True:
#        return (x - rgb_mean) / 127.5
#    elif nbit==16:
#        return (x - 2.**15)/2.**15


#def denormalize(x, rgb_mean=DIV2K_RGB_MEAN, nbit=16):
#    if True:
#        return x * 127.5 + rgb_mean


def normalize(x, rgb_mean=DIV2K_RGB_MEAN, nbit=16):
    if nbit==8:
        return (x - rgb_mean) / 127.5
    elif nbit==16:
        return (x - 2.**15)/2.**15


def denormalize(x, rgb_mean=DIV2K_RGB_MEAN, nbit=16):
    if nbit==8:
        return x * 127.5 + rgb_mean
    elif nbit==16:
        return x * 2**15 + 2**15



def normalize_01(x):
    """Normalizes RGB images to [0, 1]."""
    return x / 255.0


def normalize_m11(x):
    """Normalizes RGB images to [-1, 1]."""
    return x / 127.5 - 1


def denormalize_m11(x):
    """Inverse of normalize_m11."""
    return (x + 1) * 127.5


# ---------------------------------------
#  Metrics
# ---------------------------------------


def psnr(x1, x2, nbit=8):
    return tf.image.psnr(x1, x2, max_val=2**nbit - 1)

def psnr16(x1, x2):
    return tf.image.psnr(x1, x2, max_val=2**16-1)
# ---------------------------------------
#  See https://arxiv.org/abs/1609.05158
# ---------------------------------------


def pixel_shuffle(scale):
    return lambda x: tf.nn.depth_to_space(x, scale)


