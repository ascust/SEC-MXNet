from cores.config import conf
import argparse
import mxnet as mx
import cores.utils.misc as misc
import os
import logging
from cores.train_multi_wrapper import train_multi_wrapper
from cores.generate_bg_cues import generate_bg_cues

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training parameters")
    parser.add_argument("--gpus", default="0",
                        help="Device indices.")

    args = parser.parse_args()
    os.environ["MXNET_CPU_WORKER_NTHREADS"] = str(conf.CPU_WORKER_NUM)
    misc.my_mkdir(conf.LOG_FOLDER)
    misc.my_mkdir(conf.SNAPSHOT_FOLDER)
    misc.my_mkdir(conf.CACHE_PATH)

    log_file_name = os.path.join(conf.LOG_FOLDER, "train_bg_cue_net.log")
    if os.path.exists(log_file_name):
        os.remove(log_file_name)
    logging.basicConfig(filename=log_file_name, level=logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger().addHandler(console)

    ctx = [mx.gpu(int(i)) for i in args.gpus.split(',')]


    im_folder = os.path.join(conf.DATASET_PATH, conf.VOC_TRAIN_IM_FOLDER)
    multi_label_file = os.path.join(conf.DATASET_PATH, conf.VOC_TRAIN_MULTI_FILE)
    bg_cue_file_path = os.path.join(conf.CACHE_PATH, conf.BG_CUE_FILE)
    snapshot_prefix = os.path.join(conf.SNAPSHOT_FOLDER, "bg_cue_net")
    model_file = os.path.join(conf.SNAPSHOT_FOLDER, "bg_cue_net-1.params")
    init_weight_file = "models/%s.params" % conf.BASE_NET
    class_num = conf.CLASS_NUM - 1


    model_name = "bg_cue_%s" % conf.BASE_NET
    exec ("import cores.symbols." + model_name + " as net_symbol")
    output_size = conf.INPUT_SIZE_SEC/conf.DOWN_SAMPLE_SEC
    #check shape
    _, outshape, _ = net_symbol.create_part1(mx.sym.Variable("data"))\
                     .infer_shape(data=(1, 3, conf.INPUT_SIZE_SEC, conf.INPUT_SIZE_SEC))
    assert outshape[0][2] == output_size, "output shapes do not match."


    logging.info(conf)


    logging.info("start training bg cues for SEC.")
    train_multi_wrapper(ctx=ctx, symbol=net_symbol, snapshot_prefix=snapshot_prefix,
                        init_weight_file=init_weight_file, im_folder=im_folder,
                        multi_label_file=multi_label_file,
                        class_num=class_num, rgb_mean=conf.MEAN_RGB, lr=conf.LR_BG,
                        epoch_size=conf.EPOCH_SIZE_BG, max_epoch=1, input_size=conf.INPUT_SIZE_SEC,
                        batch_size=conf.BATCH_SIZE_BG, wd=conf.WD, momentum=conf.MOMENTUM,
                        lr_decay=conf.LD_DECAY_BG, workspace=conf.WORKSPACE)

    logging.info("start generating bg cue file for SEC.")
    generate_bg_cues(ctx=ctx, image_root=im_folder,
                     bg_cue_file_path=bg_cue_file_path, multilabel_file=multi_label_file,
                     rgb_mean=conf.MEAN_RGB, input_size=conf.INPUT_SIZE_SEC, batch_size=conf.BATCH_SIZE_BG,
                     output_size=output_size, model_file=model_file, symbol=net_symbol,
                     class_num=class_num, saliency_th=conf.SALIENCY_TH_BG)
