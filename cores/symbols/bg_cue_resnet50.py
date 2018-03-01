import mxnet as mx
import net_symbols as syms


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



def create_part1(data, workspace=512, use_global_stats=True, lr_type="alex"):
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

    return res3b

def create_part2(data, num_class, workspace=512, use_global_stats=True, lr_type="alex"):
    res3c = create_big_block(data, "3c", 128, 512, workspace=workspace, use_global_stats=use_global_stats, lr_type=lr_type)
    res3d = create_big_block(res3c, "3d", 128, 512, workspace=workspace, use_global_stats=use_global_stats, lr_type=lr_type)

    res4a = create_big_block(res3d, "4a", 256, 1024, stride=2, identity_map=False,
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
    res5a = create_big_block(res4f, "5a", 512, 2048, stride=2, identity_map=False,
                             workspace=workspace, use_global_stats=use_global_stats, lr_type=lr_type)
    res5b = create_big_block(res5a, "5b", 512, 2048, workspace=workspace,
                             use_global_stats=use_global_stats, lr_type=lr_type)
    res5c = create_big_block(res5b, "5c", 512, 2048, workspace=workspace,
                             use_global_stats=use_global_stats, lr_type=lr_type)

    adapt = syms.conv(res5c, "adapt", 512, kernel=1, lr_type="alex10", workspace=workspace)
    adapt_relu = syms.relu(adapt)

    fl = mx.sym.flatten(adapt_relu)
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

