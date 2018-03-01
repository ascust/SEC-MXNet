import mxnet as mx
import numpy as np
import os
import logging
import cPickle as pickle

def my_mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)

#load checkpoint
def load_checkpoint(prefix, epoch=None, load_symbol=False):
    symbol = None
    if load_symbol:
        symbol = mx.sym.load('%s-symbol.json' % prefix)
    if epoch is None:
        save_dict = mx.nd.load(prefix)
    else:
        save_dict = mx.nd.load('%s-%d.params' % (prefix, epoch))
    arg_params = {}
    aux_params = {}
    for k, v in save_dict.items():
        tp, name = k.split(':', 1)
        if tp == 'arg':
            arg_params[name] = v
        if tp == 'aux':
            aux_params[name] = v
    return (arg_params, aux_params, symbol)

#sava checkpoint
def save_checkpoint(prefix, epoch, symbol, arg_params, aux_params):
    if symbol is not None:
        symbol.save('%s-symbol.json' % prefix)

    save_dict = {('arg:%s' % k) : v.as_in_context(mx.cpu()) for k, v in arg_params.items()}
    save_dict.update({('aux:%s' % k) : v.as_in_context(mx.cpu()) for k, v in aux_params.items()})
    param_name = '%s-%d.params' % (prefix, epoch)
    mx.nd.save(param_name, save_dict)
    logging.info('Saved checkpoint to \"%s\"', param_name)


def create_SEC_cue(bg_cue_file, fg_cue_file, multi_lable_file, output_cue_file):
    with open(multi_lable_file, 'rb') as f:
        multi_label_dict = pickle.load(f)
    with open(fg_cue_file, 'rb') as f:
        fg_dict = pickle.load(f)
    with open(bg_cue_file, 'rb') as f:
        bg_dict = pickle.load(f)
    new_cue_dict = {}
    for f in multi_label_dict.keys():
        cues = fg_dict[f]
        bg_cues = bg_dict[f]
        new_cues = np.zeros((3, len(bg_cues[0]) + len(cues[0])), dtype=np.uint8)
        new_cues[0] = np.concatenate([np.zeros(bg_cues[0].shape), cues[0] + 1])
        new_cues[1] = np.concatenate([bg_cues[0], cues[1]])
        new_cues[2] = np.concatenate([bg_cues[1], cues[2]])
        new_cue_dict[f + "_cues"] = new_cues
        new_cue_dict[f + "_labels"] = np.array(multi_label_dict[f], dtype=np.uint8) + 1
    with open(output_cue_file, "wb") as f:
        pickle.dump(new_cue_dict, f)
