#train multi-label classification for foreground cues for SEC models.
import argparse
import mxnet as mx
import cores.utils.misc as misc
import os
from cores.config import conf
import logging
from cores.train_multi_wrapper import train_multi_wrapper
from cores.generate_fg_cues import generate_fg_cues


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training parameters")
    parser.add_argument("--gpus", default="0",
                        help="Device indices.")

    args = parser.parse_args()
    os.environ["MXNET_CPU_WORKER_NTHREADS"] = str(conf.CPU_WORKER_NUM)
    misc.my_mkdir(conf.LOG_FOLDER)
    misc.my_mkdir(conf.SNAPSHOT_FOLDER)
    misc.my_mkdir(conf.CACHE_PATH)

    log_file_name = os.path.join(conf.LOG_FOLDER, "train_fg_cue_net.log")
    if os.path.exists(log_file_name):
        os.remove(log_file_name)
    logging.basicConfig(filename=log_file_name, level=logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger().addHandler(console)


    im_folder = os.path.join(conf.DATASET_PATH, conf.VOC_TRAIN_IM_FOLDER)
    multi_label_file = os.path.join(conf.DATASET_PATH, conf.VOC_TRAIN_MULTI_FILE)
    cue_file = os.path.join(conf.CACHE_PATH, conf.FG_CUE_FILE)
    snapshot_prefix = os.path.join(conf.SNAPSHOT_FOLDER, "fg_cue_net")
    model_file = os.path.join(conf.SNAPSHOT_FOLDER, "fg_cue_net-1.params")
    init_weight_file = "models/%s.params" % conf.BASE_NET
    class_num = conf.CLASS_NUM - 1  # exclude bg class
    model_name = "fg_cue_%s" % conf.BASE_NET
    output_size = conf.INPUT_SIZE_SEC / conf.DOWN_SAMPLE_SEC
    exec ("import cores.symbols." + model_name + " as net_symbol")
    # check shape
    _, outshape, _ = net_symbol.create_body().infer_shape(data=(1, 3, conf.INPUT_SIZE_SEC, conf.INPUT_SIZE_SEC))
    assert outshape[0][2] == output_size, "output shapes do not match."

    ctx = [mx.gpu(int(i)) for i in args.gpus.split(',')]


    logging.info(conf)
    logging.info("start training fg cues for SEC.")

    train_multi_wrapper(ctx=ctx, symbol=net_symbol, snapshot_prefix=snapshot_prefix,
                        init_weight_file=init_weight_file, im_folder=im_folder,
                        multi_label_file=multi_label_file, class_num=class_num, rgb_mean=conf.MEAN_RGB,
                        epoch_size=conf.EPOCH_SIZE_FG, max_epoch=1, input_size=conf.INPUT_SIZE_SEC,
                        batch_size=conf.BATCH_SIZE_FG, lr=conf.LR_FG, wd=conf.WD,
                        momentum=conf.MOMENTUM, lr_decay=conf.LR_DECAY, workspace=conf.WORKSPACE)

    logging.info("start generating fg cue file for SEC.")

    generate_fg_cues(ctx=ctx, image_root=im_folder, multilabel_file=multi_label_file,
                     rgb_mean=conf.MEAN_RGB, symbol=net_symbol, class_num=class_num,
                     model_file=model_file, input_size=conf.INPUT_SIZE_SEC, batch_size=conf.BATCH_SIZE_FG,
                     output_size=output_size, saliency_th=conf.SALIENCY_TH_FG, workspace=conf.WORKSPACE,
                     cue_file_path=cue_file)




