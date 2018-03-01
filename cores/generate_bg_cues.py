from data.ImageFetcher import ImageFetcher
import mxnet as mx
import cores.utils.misc as misc
from scipy.ndimage import median_filter
import numpy as np
import time
import datetime
import cPickle as pickle
import logging

def generate_bg_cues(ctx, image_root, multilabel_file, bg_cue_file_path, rgb_mean, symbol, class_num, model_file,
                     input_size, batch_size, output_size, saliency_th):
    bg_cue_dict = {}
    arg_dict, aux_dict, _ = misc.load_checkpoint(model_file)
    part1_sym = symbol.create_part1(mx.sym.Variable("data"))
    part2_sym = symbol.create_part2(mx.sym.Variable("data"), class_num)

    input_shape = (batch_size, 3, input_size, input_size)
    _, part1_output_shape, _ = part1_sym.infer_shape(data=input_shape)

    mod1 = mx.mod.Module(part1_sym, label_names=[], context=ctx)
    mod1.bind(data_shapes=[("data", input_shape)], for_training=False, grad_req="null")
    mod1.set_params(arg_dict, aux_dict)

    mod2 = mx.mod.Module(part2_sym, label_names=[], context=ctx)
    mod2.bind(data_shapes=[("data", part1_output_shape[0])], for_training=True, grad_req="write", inputs_need_grad=True)
    mod2.set_params(arg_dict, aux_dict)



    fetcher = ImageFetcher(image_root=image_root, label_file=multilabel_file, rgb_mean=rgb_mean, im_size=input_size,
                           data_queue_size=8, batch_size=batch_size, num_class=class_num)
    count = 0
    batch_num = fetcher.get_batch_num()
    start_time = time.time()
    for batch in fetcher:
        ims = batch[0]
        names = batch[1]

        mod1.forward(mx.io.DataBatch(data=[mx.nd.array(ims)], label=None, pad=None, index=None))
        part1_output = mod1.get_outputs()

        mod2.forward(mx.io.DataBatch(data=[part1_output[0]], label=None, pad=None, index=None))
        grad = mx.nd.ones((batch_size, class_num))
        mod2.backward(out_grads=[grad])

        output = mod2.get_input_grads()[0].asnumpy()
        output_max = np.max(np.abs(output), axis=1)
        saliency_maps = median_filter(output_max, (1, 3, 3))
        assert saliency_maps.shape[1] == output_size and saliency_maps.shape[2] == output_size
        for i in range(len(names)):
            thr = np.sort(saliency_maps[i].ravel())[int(saliency_th * output_size ** 2)]
            cue = saliency_maps[i] < thr
            tmp = np.where(cue > 0)
            bg_cue_dict[names[i]] = (tmp[0].astype(np.uint8), tmp[1].astype(np.uint8))
        count += 1
        elapsed_time = (time.time() - start_time)
        eta = int((batch_num - count) * (elapsed_time / count))
        logging.info("processed %d/%d\t eta: %s" % (count, batch_num, str(datetime.timedelta(seconds=eta))))
        # print "processed %d/%d\t eta: %s" % (count, batch_num, str(datetime.timedelta(seconds=eta)))
    pickle.dump(bg_cue_dict, open(bg_cue_file_path, 'wb'))
    # print "done! saved to %s" % bg_cue_file_path
    logging.info("done! saved to %s" % bg_cue_file_path)
