import SEC_syms
import net_symbols as syms
import mxnet as mx
import crf_layer


def create_convrelu_unit(data, name, num_filter, lr_type="alex", pad=1, kernel=3, stride=1, dilate=1, workspace=512):
    conv = syms.conv(data, name=name, num_filter=num_filter, pad=pad, kernel=kernel, stride=stride, dilate=dilate, workspace=workspace, lr_type=lr_type)
    relu = syms.relu(conv)
    return relu

def create_body(data, lr_type="alex", workspace=512):

    # group1
    g1_1 = create_convrelu_unit(data, "conv1_1", 64, lr_type=lr_type, workspace=workspace)
    g1_2 = create_convrelu_unit(g1_1, "conv1_2", 64, lr_type=lr_type, workspace=workspace)
    pool1 = syms.maxpool(g1_2, 3, 1, 2)

    # group2
    g2_1 = create_convrelu_unit(pool1, "conv2_1", 128, lr_type=lr_type, workspace=workspace)
    g2_2 = create_convrelu_unit(g2_1, "conv2_2", 128, lr_type=lr_type, workspace=workspace)
    pool2 = syms.maxpool(g2_2, 3, 1, 2)

    # group3
    g3_1 = create_convrelu_unit(pool2, "conv3_1", 256, lr_type=lr_type, workspace=workspace)
    g3_2 = create_convrelu_unit(g3_1, "conv3_2", 256, lr_type=lr_type, workspace=workspace)
    g3_3 = create_convrelu_unit(g3_2, "conv3_3", 256, lr_type=lr_type, workspace=workspace)
    pool3 = syms.maxpool(g3_3, 3, 1, 2)

    # group4
    g4_1 = create_convrelu_unit(pool3, "conv4_1", 512, lr_type=lr_type, workspace=workspace)
    g4_2 = create_convrelu_unit(g4_1, "conv4_2", 512, lr_type=lr_type, workspace=workspace)
    g4_3 = create_convrelu_unit(g4_2, "conv4_3", 512, lr_type=lr_type, workspace=workspace)
    pool4 = syms.maxpool(g4_3, 3, 1, 1)

    # group5
    g5_1 = create_convrelu_unit(pool4, "conv5_1", 512, lr_type=lr_type, dilate=2, pad=2, workspace=workspace)
    g5_2 = create_convrelu_unit(g5_1, "conv5_2", 512, lr_type=lr_type, dilate=2, pad=2, workspace=workspace)
    g5_3 = create_convrelu_unit(g5_2, "conv5_3", 512, lr_type=lr_type, dilate=2, pad=2, workspace=workspace)

    pool5 = syms.maxpool(g5_3, 3, 1, 1)
    pool5a = syms.avgpool(pool5, 3, 1, 1)

    # group6
    conv6 = syms.conv(pool5a, "fc6", 1024, kernel=3, lr_type=lr_type, dilate=12, pad=12, workspace=workspace)
    relu6 = syms.relu(conv6)

    # group7
    conv7 = syms.conv(relu6, "fc7", 1024, lr_type=lr_type, workspace=workspace)
    relu7 = syms.relu(conv7)

    return relu7

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


    crf = mx.symbol.Custom(data=preds, small_ims=small_ims, name='crf', op_type='crf_layer',
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

