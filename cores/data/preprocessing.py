import random
import numpy as np
import mxnet as mx


def calc_crop_params(im_arr, scale_range, crop_size):
    r, c = im_arr.shape[:2]
    scale = random.random()*(scale_range[1]-scale_range[0])+scale_range[0]

    new_crop_size = int(crop_size / scale)
    r_start = random.randint(0, max(0, r-new_crop_size))
    c_start = random.randint(0, max(0, c-new_crop_size))
    return r_start, c_start, new_crop_size



def random_flip(im_arr, l_arr):
    if np.random.random() > 0.5:
        im_arr = mx.nd.flip(im_arr, axis=1)
        l_arr = l_arr[:, ::-1]
    return im_arr, l_arr

def pad_image(im_arr, l_arr, target_dim, rgb_mean, ignored_label=255):
    r, c = im_arr.shape[:2]
    pad_r = max((target_dim-r), 0)
    pad_c = max((target_dim-c), 0)
    
    if pad_r>0 or pad_c>0:
        new_im_arr = mx.nd.zeros((r+pad_r, c+pad_c, 3), dtype=np.uint8, ctx=mx.cpu()) + rgb_mean
        new_im_arr[:r, :c, :] = im_arr
        new_l_arr = np.ones((r+pad_r, c+pad_c), dtype=np.uint8)*ignored_label
        new_l_arr[:r, :c] = l_arr
        return new_im_arr, new_l_arr
    else:
        return im_arr, l_arr