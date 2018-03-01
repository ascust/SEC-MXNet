import argparse
import logging
import mxnet as mx
import os
from cores.data.InferenceDataProducer import InferenceDataProducer
from cores.utils import metrics
import numpy as np
from PIL import Image
from cores.config import conf
from cores.data.data_utils import get_voc_classnames as get_classnames
from cores.data.data_utils import get_cmap
from cores.utils import misc
from cores.utils.CRF import CRF


def main():
    conf.epoch = args.epoch
    conf.gpu = args.gpu
    conf.savescoremap = args.savescoremap
    conf.savemask = args.savemask
    conf.crf = args.crf
    conf.flip = args.flip
    logging.info(conf)

    crf = CRF(pos_xy_std=conf.CRF_POS_XY_STD, pos_w=conf.CRF_POS_W, bi_xy_std=conf.CRF_BI_XY_STD,
              bi_rgb_std=conf.CRF_BI_RGB_STD, bi_w=conf.CRF_BI_W)

    os.environ["MXNET_CUDNN_AUTOTUNE_DEFAULT"]="0"
    os.environ["MXNET_CPU_WORKER_NTHREADS"] = str(conf.CPU_WORKER_NUM)

    model_name = "SEC_%s" % conf.BASE_NET
    exec ("import cores.symbols." + model_name + " as net")
    epoch_str = str(args.epoch)
    output_path = os.path.join(conf.OUTPUT_FOLDER, model_name + "_epoch" + epoch_str)

    misc.my_mkdir(output_path)
    if conf.savescoremap:
        misc.my_mkdir(os.path.join(output_path, "scoremaps"))
    if conf.savemask:
        misc.my_mkdir(os.path.join(output_path, "masks"))


    ctx = mx.gpu(int(args.gpu))
    cmap = get_cmap()

    seg_net = net.create_infer(conf.CLASS_NUM, conf.WORKSPACE)
    seg_net_prefix = os.path.join(conf.SNAPSHOT_FOLDER, model_name)
    arg_dict, aux_dict, _ = misc.load_checkpoint(seg_net_prefix, args.epoch)


    mod = mx.mod.Module(seg_net, data_names=["data"], label_names=[], context=ctx)
    mod.bind(data_shapes=[("data", (1, 3, conf.INPUT_SIZE_SEC, conf.INPUT_SIZE_SEC))],
             for_training=False, grad_req="null")
    initializer = mx.init.Normal()
    initializer.set_verbosity(True)
    mod.init_params(initializer=initializer, arg_params=arg_dict, aux_params=aux_dict, allow_missing=True)

    data_producer = InferenceDataProducer(
        im_root=os.path.join(conf.DATASET_PATH, conf.VOC_VAL_IM_FOLDER),
        mask_root=os.path.join(conf.DATASET_PATH, conf.VOC_VAL_MASK_FOLDER),
        flist_path=os.path.join(conf.DATASET_PATH, conf.VOC_VAL_LIST),
        rgb_mean=conf.MEAN_RGB,
        input_size=conf.INPUT_SIZE_SEC)

    nbatch = 0
    eval_metrics = [metrics.IOU(conf.CLASS_NUM, get_classnames())]
    logging.info("In evaluation...")

    while True:
        data = data_producer.get_data()
        if data is None:
            break
        im = data[0]

        label = data[1].squeeze()
        im_name = data[2]
        ori_im = data[3]


        mod.forward(mx.io.DataBatch(data=[im]))
        score = mx.nd.transpose(mod.get_outputs()[0].copyto(mx.cpu()), [0, 2, 3, 1])
        score = mx.nd.reshape(score, (score.shape[1], score.shape[2], score.shape[3]))
        up_score = mx.nd.transpose(mx.image.imresize(score, label.shape[1], label.shape[0], interp=1), [2, 0, 1])

        if conf.flip:
            flip_im = im[:, :, :, ::-1]
            mod.forward(mx.io.DataBatch(data=[flip_im]))
            flip_score = mx.nd.transpose(mod.get_outputs()[0].copyto(mx.cpu()), [0, 2, 3, 1])
            flip_score = mx.nd.reshape(flip_score, (flip_score.shape[1], flip_score.shape[2], flip_score.shape[3]))
            flip_up_score = mx.nd.transpose(mx.image.imresize(flip_score, label.shape[1], label.shape[0], interp=1), [2, 0, 1])
            up_score += mx.nd.flip(flip_up_score, axis=2)
            up_score /= 2



        if conf.crf:
            final_scoremaps = mx.nd.log(up_score).asnumpy()
            final_scoremaps = crf.inference(ori_im.asnumpy(), final_scoremaps)
        else:
            final_scoremaps = up_score.asnumpy()
        pred_label = final_scoremaps.argmax(0)

        for eval in eval_metrics:
            eval.update(label, pred_label)


        if conf.savemask:
            out_img = np.uint8(pred_label)
            out_img = Image.fromarray(out_img)
            out_img.putpalette(cmap)
            output_name = im_name[:im_name.rfind(".")]
            output_name += ".png"
            out_img.save(os.path.join(output_path, "masks", output_name))
        if conf.savescoremap:
            output_name = im_name[:im_name.rfind(".")]
            np.save(os.path.join(output_path, "scoremaps", output_name), final_scoremaps)

        nbatch += 1
        if nbatch % 10 == 0:
            print "processed %dth batch" % nbatch

    logging.info("Epoch [%d]: " % args.epoch)
    for m in eval_metrics:
        logging.info("[overall] [%s: %.4f]" % (m.get()[0], m.get()[1]))
        if m.get_class_values() is not None:
            scores = "[perclass] ["
            for v in m.get_class_values():
                scores += "%s: %.4f\t" % (v[0], v[1])
            scores += "]"
            logging.info(scores)

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training parameters")
    parser.add_argument("--gpu", default="0",
                        help="Device index.")
    parser.add_argument("--epoch", default=0, type=int,
                        help="snapshot name for evaluation")
    parser.add_argument("--savemask", help="whether save the prediction masks.",
                        action="store_true")
    parser.add_argument("--savescoremap", help="whether save the prediction scoremaps.",
                        action="store_true")
    parser.add_argument("--crf", help="whether use crf for post processing.",
                        action="store_true")
    parser.add_argument("--flip", help="whether use flip.",
                        action="store_true")
    misc.my_mkdir(conf.OUTPUT_FOLDER)
    misc.my_mkdir(conf.LOG_FOLDER)
    args = parser.parse_args()
    logging.basicConfig(filename=os.path.join(conf.LOG_FOLDER, "evaluation_log.log"), level=logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger().addHandler(console)
    main()






