import argparse
from cores.config import conf
from cores.data.SECTrainingIter import SECTrainingIter
import os
import mxnet as mx
import cores.utils.metrics as metrics
import logging
import cores.utils.misc as misc
import numpy as np
import cores.utils.callbacks as callbacks

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training parameters")
    parser.add_argument("--gpus", default="0",
                        help="Device indices.")
    parser.add_argument("--epoch", default=0, type=int,
                        help="Starting epoch.")
    parser.add_argument("--lr", default=-1, type=float,
                        help="Learning rate.")


    args = parser.parse_args()
    misc.my_mkdir(conf.SNAPSHOT_FOLDER)
    os.environ["MXNET_CPU_WORKER_NTHREADS"] = str(conf.CPU_WORKER_NUM)

    ctx = [mx.gpu(int(i)) for i in args.gpus.split(',')]
    log_file_name = os.path.join(conf.LOG_FOLDER, "train_SEC_model.log")
    if os.path.exists(log_file_name) and args.epoch==0:
        os.remove(log_file_name)
    logging.basicConfig(filename=log_file_name, level=logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger().addHandler(console)

    bg_cue_file = os.path.join(conf.CACHE_PATH, conf.BG_CUE_FILE)
    fg_cue_file = os.path.join(conf.CACHE_PATH, conf.FG_CUE_FILE)
    multi_lable_file = os.path.join(conf.DATASET_PATH, conf.VOC_TRAIN_MULTI_FILE)
    output_cue_file = os.path.join(conf.CACHE_PATH, conf.CUE_FILE_INITSEC)


    logging.info("generating cue file")
    misc.create_SEC_cue(bg_cue_file=bg_cue_file, fg_cue_file=fg_cue_file,
                   multi_lable_file=multi_lable_file, output_cue_file=output_cue_file)
    logging.info("cue file generated")
    im_root = os.path.join(conf.DATASET_PATH, conf.VOC_TRAIN_IM_FOLDER)
    model_name = "SEC_%s" % conf.BASE_NET
    exec ("import cores.symbols." + model_name + " as net_symbol")
    model_prefix = os.path.join(conf.SNAPSHOT_FOLDER, "%s" % (model_name))
    init_weight_file = "models/%s.params" % conf.BASE_NET
    output_size = conf.INPUT_SIZE_SEC / conf.DOWN_SAMPLE_SEC


    logging.info(conf)
    logging.info("start training model %s" % (model_name))

    arg_dict = {}
    aux_dict = {}
    seg_net = net_symbol.create_training(class_num=conf.CLASS_NUM, outputsize=output_size, workspace=conf.WORKSPACE)
    if args.epoch == 0:
        if not os.path.exists(init_weight_file):
            logging.warn("No model file found at %s. Start from scratch!" % init_weight_file)
        else:
            arg_dict, aux_dict, _ = misc.load_checkpoint(init_weight_file)
    else:
        arg_dict, aux_dict, _ = misc.load_checkpoint(model_prefix, args.epoch)
    # init weights for expand loss

    arg_dict["fg_w"] = mx.nd.array(
        np.array([conf.Q_FG ** i for i in range(output_size * output_size - 1, -1, -1)])[None, None, :])
    arg_dict["bg_w"] = mx.nd.array(np.array([conf.Q_BG ** i for i in range(output_size * output_size - 1, -1, -1)])[None, :])

    data_iter = SECTrainingIter(
        im_root=im_root,
        cue_file_path=output_cue_file,
        class_num=conf.CLASS_NUM,
        rgb_mean=conf.MEAN_RGB,
        im_size=conf.INPUT_SIZE_SEC,
        shuffle=True,
        label_shrink_scale=1.0/conf.DOWN_SAMPLE_SEC,
        random_flip=True,
        data_queue_size=8,
        epoch_size=conf.EPOCH_SIZE,
        batch_size=conf.BATCH_SIZE,
        round_batch=True
    )

    initializer = mx.initializer.Normal()
    initializer.set_verbosity(True)

    mod = mx.mod.Module(seg_net, context=ctx, data_names=["data", "small_ims"], label_names=["labels", "cues"])

    mod.bind(data_shapes=data_iter.provide_data,
             label_shapes=data_iter.provide_label)
    mod.init_params(initializer=initializer, arg_params=arg_dict, aux_params=aux_dict, allow_missing=(args.epoch == 0))

    opt_params = {"learning_rate": conf.LR,
                  "wd": conf.WD,
                  'momentum': conf.MOMENTUM,
                  'lr_scheduler': mx.lr_scheduler.FactorScheduler(step=conf.LR_DECAY, factor=0.1),
                  'rescale_grad': 1.0 / len(ctx)}

    eval_metrics = [metrics.SEC_seed_loss(), metrics.SEC_constrain_loss(), metrics.SEC_expand_loss()]
    mod.fit(data_iter,
            optimizer="sgd",
            optimizer_params=opt_params,
            num_epoch=conf.MAX_EPOCH + 1,
            epoch_end_callback=callbacks.module_checkpoint(model_prefix),
            batch_end_callback=callbacks.Speedometer(conf.BATCH_SIZE, frequent=10),
            eval_metric=eval_metrics,
            begin_epoch=args.epoch + 1)


