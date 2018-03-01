import SEC_syms
import net_symbols as syms
import mxnet as mx
import crf_layer
def create_block(data, name, num_filter, kernel, pad=0, stride=1, dilate=1, workspace=512,
                 use_global_stats=True, lr_type="alex"):
    res = syms.conv(data=data, name="res" + name, num_filter=num_filter, pad=pad, kernel=kernel, stride=stride,
                    dilate=dilate, no_bias=True, workspace=workspace, lr_type=lr_type)
    bn = syms.bn(res, name="bn" + name, use_global_stats=use_global_stats, lr_type=lr_type)
    return bn


def create_big_block(data, name, num_filter1, num_filter2, stride=1, dilate=1, pad=1, identity_map=True,
                     workspace=512, use_global_stats=True, lr_type="alex"):
    blocka = create_block(data, name=name+"_branch2a", num_filter=num_filter1, kernel=1, stride=stride,
                          workspace=workspace, use_global_stats=use_global_stats, lr_type=lr_type)
    relu1 = syms.relu(blocka)
    blockb = create_block(relu1, name=name + "_branch2b", num_filter=num_filter1, kernel=3, dilate=dilate, pad=pad,
                          workspace=workspace, use_global_stats=use_global_stats, lr_type=lr_type)
    relu2 = syms.relu(blockb)
    blockc = create_block(relu2, name=name+"_branch2c", num_filter=num_filter2, kernel=1, workspace=workspace,
                          use_global_stats=use_global_stats, lr_type=lr_type)
    if identity_map:
        return syms.relu(data+blockc)
    else:
        branch1 = create_block(data, name=name+"_branch1", num_filter=num_filter2, kernel=1, stride=stride,
                               workspace=workspace, use_global_stats=use_global_stats, lr_type=lr_type)
        return syms.relu(branch1+blockc)


def create_body(data, use_global_stats=True, lr_type="alex", workspace=512):
    conv1 = syms.conv(data, name="conv1", num_filter=64, pad=3, kernel=7, stride=2, workspace=workspace, lr_type=lr_type)
    bn = syms.bn(conv1, name="bn_conv1", use_global_stats=use_global_stats, lr_type=lr_type)
    relu = syms.relu(bn)
    pool1 = syms.maxpool(relu, kernel=3, stride=2, pad=1)

    res2a = create_big_block(pool1, "2a", 64, 256, identity_map=False, workspace=workspace
                             , use_global_stats=use_global_stats, lr_type=lr_type)
    res2b = create_big_block(res2a, "2b", 64, 256, workspace=workspace, use_global_stats=use_global_stats, lr_type=lr_type)
    res2c = create_big_block(res2b, "2c", 64, 256, workspace=workspace, use_global_stats=use_global_stats, lr_type=lr_type)
    res3a = create_big_block(res2c, "3a", 128, 512, stride=2,identity_map=False, workspace=workspace,
                             use_global_stats=use_global_stats, lr_type=lr_type)
    res3b = create_big_block(res3a, "3b", 128, 512, workspace=workspace, use_global_stats=use_global_stats, lr_type=lr_type)
    res3c = create_big_block(res3b, "3c", 128, 512, workspace=workspace, use_global_stats=use_global_stats, lr_type=lr_type)
    res3d = create_big_block(res3c, "3d", 128, 512, workspace=workspace, use_global_stats=use_global_stats, lr_type=lr_type)
    res4a = create_big_block(res3d, "4a", 256, 1024, stride=1, identity_map=False, pad=2, dilate=2,
                             workspace=workspace, use_global_stats=use_global_stats, lr_type=lr_type)
    res4b = create_big_block(res4a, "4b", 256, 1024, workspace=workspace, pad=2, dilate=2,
                             use_global_stats=use_global_stats, lr_type=lr_type)
    res4c = create_big_block(res4b, "4c", 256, 1024, workspace=workspace, pad=2, dilate=2,
                             use_global_stats=use_global_stats, lr_type=lr_type)
    res4d = create_big_block(res4c, "4d", 256, 1024, workspace=workspace, pad=2, dilate=2,
                             use_global_stats=use_global_stats, lr_type=lr_type)
    res4e = create_big_block(res4d, "4e", 256, 1024, workspace=workspace, pad=2, dilate=2,
                             use_global_stats=use_global_stats, lr_type=lr_type)
    res4f = create_big_block(res4e, "4f", 256, 1024, workspace=workspace, pad=2, dilate=2,
                             use_global_stats=use_global_stats, lr_type=lr_type)
    res5a = create_big_block(res4f, "5a", 512, 2048, stride=1, identity_map=False, pad=4, dilate=4,
                             workspace=workspace, use_global_stats=use_global_stats, lr_type=lr_type)
    res5b = create_big_block(res5a, "5b", 512, 2048, workspace=workspace, pad=4, dilate=4,
                             use_global_stats=use_global_stats, lr_type=lr_type)
    res5c = create_big_block(res5b, "5c", 512, 2048, workspace=workspace, pad=4, dilate=4,
                             use_global_stats=use_global_stats, lr_type=lr_type)
    new_conv1 = syms.conv(res5c, name="new_conv1", num_filter=512, pad=12, kernel=3, dilate=12,
                          workspace=workspace, lr_type="alex10")
    new_relu1 = syms.relu(new_conv1)
    dp1 = syms.dropout(new_relu1)
    fc = syms.conv(dp1, name="fc", num_filter=512, workspace=workspace, lr_type="alex10")
    fc_relu = syms.relu(fc)

    return fc_relu

def create_classifier(data, class_num, lr_type="alex10", workspace=512):
    fc8_SEC = syms.conv(data, "fc8_SEC", num_filter=class_num, lr_type=lr_type, workspace=workspace)
    return fc8_SEC


def create_training(class_num, outputsize, workspace=512):
    data = mx.symbol.Variable(name="data")
    labels = mx.symbol.Variable(name="labels")
    cues = mx.symbol.Variable(name="cues")
    small_ims = mx.symbol.Variable(name="small_ims")
    fg_w = mx.symbol.Variable(name="fg_w", shape=(1, 1, outputsize*outputsize))
    bg_w = mx.symbol.Variable(name="bg_w", shape=(1, outputsize*outputsize))
    body = create_body(data, workspace=workspace)
    preds = create_classifier(body, class_num=class_num, workspace=workspace, lr_type="alex10")


    probs = SEC_syms.create_softmax(preds)


    crf = mx.symbol.Custom(data=probs, small_ims=small_ims, name='crf', op_type='crf_layer',
                           pos_xy_std=3, pos_w=3, bi_xy_std=80, bi_rgb_std=13, bi_w=10,
                            maxiter=10, scale_factor=12.0, min_prob=0.0001)
    expand_loss = SEC_syms.create_expand_loss(probs, labels, fg_w, bg_w)
    seed_loss = SEC_syms.create_seed_loss(probs, cues)
    constrain_loss = SEC_syms.create_constrain_loss(probs, crf)

    return mx.sym.Group([seed_loss, constrain_loss, expand_loss])


def create_infer(class_num, workspace=512):
    data = mx.symbol.Variable(name="data")
    body = create_body(data, workspace=workspace)
    preds = create_classifier(body, class_num=class_num, workspace=workspace)
    softmax = mx.symbol.softmax(preds, axis=1)
    return softmax
