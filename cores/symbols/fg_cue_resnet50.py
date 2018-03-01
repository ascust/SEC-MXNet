import mxnet as mx
import net_symbols as syms



def create_block(data, name, num_filter, kernel, pad=0, stride=1, dilate=1, workspace=512, use_global_stats=True, lr_type="alex"):
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

def create_body(lr_type="alex", workspace=512, use_global_stats=True):
    data = mx.symbol.Variable(name="data")
    conv1 = syms.conv(data, name="conv1", num_filter=64, pad=3, kernel=7, stride=2, workspace=workspace, lr_type=lr_type)
    bn = syms.bn(conv1, name="bn_conv1", use_global_stats=use_global_stats, lr_type=lr_type)
    relu = syms.relu(bn)
    pool1 = syms.maxpool(relu, kernel=3, stride=2, pad=1)

    res2a = create_big_block(pool1, "2a", 64, 256, identity_map=False, workspace=workspace,
                             use_global_stats=use_global_stats, lr_type=lr_type)
    res2b = create_big_block(res2a, "2b", 64, 256, workspace=workspace, use_global_stats=use_global_stats, lr_type=lr_type)
    res2c = create_big_block(res2b, "2c", 64, 256, workspace=workspace, use_global_stats=use_global_stats)
    res3a = create_big_block(res2c, "3a", 128, 512, stride=2,identity_map=False, workspace=workspace,
                             use_global_stats=use_global_stats, lr_type=lr_type)
    res3b = create_big_block(res3a, "3b", 128, 512, workspace=workspace, use_global_stats=use_global_stats, lr_type=lr_type)
    res3c = create_big_block(res3b, "3c", 128, 512, workspace=workspace, use_global_stats=use_global_stats, lr_type=lr_type)
    res3d = create_big_block(res3c, "3d", 128, 512, workspace=workspace, use_global_stats=use_global_stats, lr_type=lr_type)
    res4a = create_big_block(res3d, "4a", 256, 1024, stride=1, identity_map=False,
                             workspace=workspace, use_global_stats=use_global_stats, lr_type=lr_type)
    res4b = create_big_block(res4a, "4b", 256, 1024, workspace=workspace,
                             use_global_stats=use_global_stats, lr_type=lr_type)
    res4c = create_big_block(res4b, "4c", 256, 1024, workspace=workspace,
                             use_global_stats=use_global_stats, lr_type=lr_type)
    res4d = create_big_block(res4c, "4d", 256, 1024, workspace=workspace,
                             use_global_stats=use_global_stats, lr_type=lr_type)
    res4e = create_big_block(res4d, "4e", 256, 1024, workspace=workspace,
                             use_global_stats=use_global_stats, lr_type=lr_type)
    res4f = create_big_block(res4e, "4f", 256, 1024, workspace=workspace,
                             use_global_stats=use_global_stats, lr_type=lr_type)
    res5a = create_big_block(res4f, "5a", 512, 2048, stride=1, identity_map=False,
                             workspace=workspace, use_global_stats=use_global_stats, lr_type=lr_type)
    res5b = create_big_block(res5a, "5b", 512, 2048, workspace=workspace,
                             use_global_stats=use_global_stats, lr_type=lr_type)
    res5c = create_big_block(res5b, "5c", 512, 2048, workspace=workspace,
                             use_global_stats=use_global_stats, lr_type=lr_type)

    # group6
    conv6 = syms.conv(res5c, "conv6_CAM", 1024, pad=1, kernel=3, lr_type="alex10", workspace=workspace)
    relu6 = syms.relu(conv6)
    relu6_dp = syms.dropout(relu6, 0.5)

    # group7
    conv7 = syms.conv(relu6_dp, "conv7_CAM", 1024, pad=1, kernel=3, lr_type="alex10", workspace=workspace)
    relu7 = syms.relu(conv7)
    relu7_dp = syms.dropout(relu7, 0.5)

    return relu7_dp


def create_classifier(data, class_num, lr_type="alex10", workspace=512):
    CAM_pool = syms.avgpool(data, global_pool=True)
    CAM_fc = syms.conv(CAM_pool, "scores", num_filter=class_num, lr_type=lr_type, workspace=workspace)
    flattened = mx.sym.flatten(CAM_fc)
    return flattened


def create_train(num_class, workspace=512):
    label = mx.sym.Variable("label")
    body = create_body(workspace=workspace)
    new_layers = create_classifier(body, class_num=num_class, workspace=workspace, lr_type="alex10")
    score = mx.sym.LogisticRegressionOutput(new_layers, label=label)
    return score


def create_infer(class_num, workspace=512):
    body = create_body(workspace=workspace)
    new_layers = create_classifier(body, class_num=class_num, workspace=workspace)
    score = mx.sym.Activation(new_layers, act_type="sigmoid")
    return score

def create_CAM(class_num, workspace=512):
    score_weight = mx.sym.Variable("scores_weight", shape=(class_num, 1024, 1, 1))
    score_weight = mx.sym.reshape(score_weight, (class_num, 1024))
    feat = create_body(workspace=workspace)
    feat_swap = mx.sym.swapaxes(feat, 0, 1)
    feat_reshape = mx.sym.reshape(feat_swap, (0, -1))
    score = mx.sym.dot(score_weight, feat_reshape)

    return score


