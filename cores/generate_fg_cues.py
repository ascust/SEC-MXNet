import cores.utils.misc as misc
import mxnet as mx
import time
import datetime
import numpy as np
import cPickle as pickle
from cores.data.ImageFetcher import ImageFetcher
import logging

def generate_fg_cues(ctx, image_root, multilabel_file, rgb_mean, symbol, class_num, model_file,
                     input_size, batch_size, output_size, saliency_th, workspace, cue_file_path):
    ctx = ctx

    net_sym = symbol.create_CAM(class_num, workspace)

    arg_dict, aux_dict, _ = misc.load_checkpoint(model_file)
    mod = mx.mod.Module(net_sym, data_names=["data"], label_names=[], context=ctx)
    mod.bind(data_shapes=[("data", (batch_size, 3, input_size, input_size))], for_training=False,
             grad_req="null")
    mod.set_params(arg_dict, aux_dict)

    fetcher = ImageFetcher(image_root=image_root, label_file=multilabel_file, rgb_mean=rgb_mean, im_size=input_size,
                           data_queue_size=8, batch_size=batch_size, num_class=class_num)

    mask_dict = {}
    count = 0
    batch_num = fetcher.get_batch_num()
    start_time = time.time()
    for batch in fetcher:
        ims = batch[0]
        names = batch[1]
        labels = batch[2]
        mod.forward(mx.io.DataBatch(data=[ims], label=None, pad=None, index=None))
        raw_outputs = mod.get_outputs(merge_multi_context=False)[0]
        outputs = []
        for out in raw_outputs:
            outputs.append(out.asnumpy().reshape(class_num, -1, output_size, output_size).swapaxes(0, 1))
        outputs = np.concatenate(outputs, axis=0)

        for i in range(len(names)):
            cue_map = np.zeros((class_num, output_size, output_size), dtype=np.bool)
            for l in np.where(labels[i]>0)[0]:
                cur_heatmap = outputs[i][l]
                new_cue = cur_heatmap > saliency_th*np.max(cur_heatmap)
                cue_map[l, :, :] = new_cue
            tmp = np.where(cue_map)
            mask_dict[names[i]] = [tmp[0].astype(np.uint8), tmp[1].astype(np.uint8), tmp[2].astype(np.uint8)]
        count += 1
        elapsed_time = (time.time() - start_time)
        eta = int((batch_num - count) * (elapsed_time / count))
        logging.info("processed %d/%d\t eta: %s" % (count, batch_num, str(datetime.timedelta(seconds=eta))))
        # print "processed %d/%d\t eta: %s" % (count, batch_num, str(datetime.timedelta(seconds=eta)))
    pickle.dump(mask_dict, open(cue_file_path, 'wb'))
    logging.info("done! saved to %s" % cue_file_path)
    # print "done! saved to %s" % new_cue_file
