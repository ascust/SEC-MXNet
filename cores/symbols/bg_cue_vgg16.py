import mxnet as mx
import net_symbols as syms


def create_convrelu_unit(data, name, num_filter, lr_type="alex", pad=1, kernel=3, stride=1, workspace=512):
    conv = syms.conv(data, name=name, num_filter=num_filter, pad=pad, kernel=kernel, stride=stride, workspace=workspace, lr_type=lr_type)
    relu = syms.relu(conv)
    return relu

def create_part1(data, workspace=512):
    g1_1 = create_convrelu_unit(data, "conv1_1", 64, lr_type="alex", workspace=workspace)
    g1_2 = create_convrelu_unit(g1_1, "conv1_2", 64, lr_type="alex", workspace=workspace)
    pool1 = syms.maxpool(g1_2, 2, 0, 2, pooling_convention='valid')

    # group2
    g2_1 = create_convrelu_unit(pool1, "conv2_1", 128, lr_type="alex", workspace=workspace)
    g2_2 = create_convrelu_unit(g2_1, "conv2_2", 128, lr_type="alex", workspace=workspace)
    pool2 = syms.maxpool(g2_2, 2, 0, 2, pooling_convention='valid')

    # group3
    g3_1 = create_convrelu_unit(pool2, "conv3_1", 256, lr_type="alex", workspace=workspace)
    g3_2 = create_convrelu_unit(g3_1, "conv3_2", 256, lr_type="alex", workspace=workspace)
    g3_3 = create_convrelu_unit(g3_2, "conv3_3", 256, lr_type="alex", workspace=workspace)
    pool3 = syms.maxpool(g3_3, 2, 0, 2, pooling_convention='valid')
    g4_1 = create_convrelu_unit(pool3, "conv4_1", 512, lr_type="alex", workspace=workspace)
    return g4_1

def create_part2(data, num_class, workspace=512):
    # group4
    # g4_1 = create_convrelu_unit(data, "conv4_1", 512, lr_type="alex", workspace=workspace)
    g4_2 = create_convrelu_unit(data, "conv4_2", 512, lr_type="alex", workspace=workspace)
    g4_3 = create_convrelu_unit(g4_2, "conv4_3", 512, lr_type="alex", workspace=workspace)
    pool4 = syms.maxpool(g4_3, 2, 0, 2, pooling_convention='valid')

    # group5
    g5_1 = create_convrelu_unit(pool4, "conv5_1", 512, lr_type="alex", workspace=workspace)
    g5_2 = create_convrelu_unit(g5_1, "conv5_2", 512, lr_type="alex", workspace=workspace)
    g5_3 = create_convrelu_unit(g5_2, "conv5_3", 512, lr_type="alex", workspace=workspace)
    pool5 = syms.maxpool(g5_3, 2, 0, 2, pooling_convention='valid')

    fl = mx.sym.flatten(pool5)
    fc6 = syms.fullyconnected(fl, num_hidden=1024, name="CAMfc6", lr_type="alex10")
    relu6 = syms.relu(fc6)

    # group7
    fc7 = syms.fullyconnected(relu6, num_hidden=1024, name="CAMfc7", lr_type="alex10")
    relu7 = syms.relu(fc7)

    fc8 = syms.fullyconnected(relu7, num_hidden=num_class, name="CAMfc8", lr_type="alex10")
    return fc8



def create_train(num_class, workspace=512):
    label = mx.sym.Variable("label")
    data = mx.sym.Variable("data")

    part1 = create_part1(data, workspace=workspace)
    part2 = create_part2(part1, num_class, workspace=workspace)

    score = mx.sym.LogisticRegressionOutput(part2, label=label)
    return score

