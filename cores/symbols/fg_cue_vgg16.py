import mxnet as mx
import net_symbols as syms


def create_convrelu_unit(data, name, num_filter, lr_type="alex", pad=1, kernel=3, stride=1, workspace=512):
    conv = syms.conv(data, name=name, num_filter=num_filter, pad=pad, kernel=kernel, stride=stride, workspace=workspace, lr_type=lr_type)
    relu = syms.relu(conv)
    return relu

def create_body(lr_type="alex", workspace=512):
    data = mx.symbol.Variable(name="data")
    # group1
    g1_1 = create_convrelu_unit(data, "conv1_1", 64, lr_type=lr_type, workspace=workspace)
    g1_2 = create_convrelu_unit(g1_1, "conv1_2", 64, lr_type=lr_type, workspace=workspace)
    pool1 = syms.maxpool(g1_2, 3, 1, 2, pooling_convention="valid")

    # group2
    g2_1 = create_convrelu_unit(pool1, "conv2_1", 128, lr_type=lr_type, workspace=workspace)
    g2_2 = create_convrelu_unit(g2_1, "conv2_2", 128, lr_type=lr_type, workspace=workspace)
    pool2 = syms.maxpool(g2_2, 3, 1, 2, pooling_convention="valid")

    # group3
    g3_1 = create_convrelu_unit(pool2, "conv3_1", 256, lr_type=lr_type, workspace=workspace)
    g3_2 = create_convrelu_unit(g3_1, "conv3_2", 256, lr_type=lr_type, workspace=workspace)
    g3_3 = create_convrelu_unit(g3_2, "conv3_3", 256, lr_type=lr_type, workspace=workspace)
    pool3 = syms.maxpool(g3_3, 3, 1, 2, pooling_convention="valid")

    # group4
    g4_1 = create_convrelu_unit(pool3, "conv4_1", 512, lr_type=lr_type, workspace=workspace)
    g4_2 = create_convrelu_unit(g4_1, "conv4_2", 512, lr_type=lr_type, workspace=workspace)
    g4_3 = create_convrelu_unit(g4_2, "conv4_3", 512, lr_type=lr_type, workspace=workspace)


    # group5
    g5_1 = create_convrelu_unit(g4_3, "conv5_1", 512, lr_type=lr_type, workspace=workspace)
    g5_2 = create_convrelu_unit(g5_1, "conv5_2", 512, lr_type=lr_type, workspace=workspace)
    g5_3 = create_convrelu_unit(g5_2, "conv5_3", 512, lr_type=lr_type, workspace=workspace)

    # group6
    conv6 = syms.conv(g5_3, "conv6_CAM", 1024, pad=1, kernel=3, lr_type="alex10", workspace=workspace)
    relu6 = syms.relu(conv6)
    relu6_dp = syms.dropout(relu6, 0.5)

    # group7
    conv7 = syms.conv(relu6_dp, "conv7_CAM", 1024, pad=1, kernel=3, lr_type="alex10", workspace=workspace)
    relu7 = syms.relu(conv7)
    relu7_dp = syms.dropout(relu7, 0.5)

    return relu7_dp

def create_classifier(data, num_class, lr_type="alex10", workspace=512):
    CAM_pool = syms.avgpool(data, global_pool=True)
    CAM_fc = syms.conv(CAM_pool, "scores", num_filter=num_class, lr_type=lr_type, workspace=workspace)
    flattened = mx.sym.flatten(CAM_fc)
    return flattened


def create_train(num_class, workspace=512):
    label = mx.sym.Variable("label")
    body = create_body(workspace=workspace)
    new_layers = create_classifier(body, num_class=num_class, workspace=workspace, lr_type="alex10")
    score = mx.sym.LogisticRegressionOutput(new_layers, label=label)
    return score


def create_infer(num_class, workspace=512):
    body = create_body(workspace=workspace)
    new_layers = create_classifier(body, num_class=num_class, workspace=workspace)
    score = mx.sym.Activation(new_layers, act_type="sigmoid")
    return score

def create_CAM(num_class, workspace=512):
    score_weight = mx.sym.Variable("scores_weight", shape=(num_class, 1024, 1, 1))
    score_weight = mx.sym.reshape(score_weight, (num_class, 1024))
    feat = create_body(workspace=workspace)
    feat_swap = mx.sym.swapaxes(feat, 0, 1)
    feat_reshape = mx.sym.reshape(feat_swap, (0, -1))
    score = mx.sym.dot(score_weight, feat_reshape)

    return score


