import mxnet as mx
import numpy as np

def create_softmax(preds, min_prob=0.0001):
    min_prob_const = mx.sym.ones((1, 1, 1, 1)) * min_prob
    preds_max = mx.sym.max(preds, keepdims=True, axis=1, name="softmax_max")
    preds_exp = mx.sym.exp(mx.sym.broadcast_minus(preds, preds_max, name="softmax_minus"))
    tmp1 = mx.sym.sum(preds_exp, keepdims=True, axis=1)
    tmp2 = mx.sym.broadcast_div(preds_exp, tmp1)
    tmp3 = mx.sym.broadcast_add(tmp2, min_prob_const)
    probs = mx.sym.broadcast_div(tmp3, mx.sym.sum(tmp3, axis=1, keepdims=True))
    return probs


def create_seed_loss(probs, cues):
    count = mx.sym.sum(cues, axis=(1, 2, 3), keepdims=True)
    tmp = mx.sym.sum(cues*mx.sym.log(probs, name="seedloss_log"), axis=(1, 2, 3), keepdims=True)
    loss_balanced = -mx.sym.mean(tmp/count)
    loss = mx.sym.MakeLoss(loss_balanced)
    return loss

def create_expand_loss(probs_tmp, stat_inp, fg_w, bg_w):
    stat = mx.sym.slice_axis(stat_inp, axis=3, begin=1, end=None)
    probs_bg = mx.sym.slice_axis(probs_tmp, begin=0, end=1, axis=1)
    probs = mx.sym.slice_axis(probs_tmp, begin=1, end=None, axis=1)
    probs_max = mx.sym.max(probs, axis=(2, 3)) #different imp


    probs_sort = mx.sym.sort(mx.sym.reshape(probs, (0, 0, -1)), axis=2)
    fg_w = mx.sym.BlockGrad(fg_w)
    Z_fg = mx.sym.sum(fg_w)
    tmp = mx.sym.broadcast_mul(probs_sort, fg_w)
    tmp = mx.sym.broadcast_div(tmp, Z_fg)
    probs_mean = mx.sym.sum(tmp, axis=2)


    probs_bg_sort = mx.sym.sort(mx.sym.reshape(probs_bg, (0, -1)), axis=1)
    bg_w = mx.sym.BlockGrad(bg_w)
    Z_bg = mx.sym.sum(bg_w)
    tmp = mx.sym.broadcast_mul(probs_bg_sort, bg_w)
    tmp = mx.sym.broadcast_div(tmp, Z_bg)
    probs_bg_mean = mx.sym.sum(tmp, axis=1)

    constant_one = mx.sym.ones((1, 1))
    stat = mx.sym.reshape(stat, (0, -1))
    stat_2d = mx.sym.broadcast_greater(stat, constant_one*0.5)
    tmp1 = mx.sym.broadcast_mul(stat_2d, mx.sym.log(probs_mean))
    tmp2 = mx.sym.sum(stat_2d, axis=1, keepdims=True)
    tmp3 = mx.sym.broadcast_div(tmp1, tmp2)
    tmp = mx.sym.sum(tmp3, axis=1)
    loss1 = -mx.sym.mean(tmp)

    tmp1 = mx.sym.broadcast_minus(constant_one, stat_2d)
    tmp2 = mx.sym.log(mx.sym.broadcast_minus(constant_one, probs_max))
    tmp3 = mx.sym.sum(mx.sym.broadcast_minus(constant_one, stat_2d), axis=1, keepdims=True)
    tmp4 = mx.sym.broadcast_div(tmp1*tmp2, tmp3)
    tmp = mx.sym.sum(tmp4, axis=1)
    loss2 = -mx.sym.mean(tmp)
    loss3 = -mx.sym.mean(mx.sym.log(probs_bg_mean))
    loss = loss1 + loss2 + loss3
    loss = mx.sym.MakeLoss(loss)
    return loss

def create_constrain_loss(probs, probs_smooth_log):
    probs_smooth = mx.sym.exp(probs_smooth_log, name="closs_smooth_log")
    tmp_div = mx.sym.elemwise_div(probs_smooth, probs, name="closs_div")
    tmp = probs_smooth*mx.sym.log(tmp_div)
    mean = mx.sym.mean(mx.sym.sum(tmp, axis=1))
    loss = mx.sym.MakeLoss(mean)
    return loss