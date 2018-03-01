from data.MultiLabelIter import MultiLabelIter
import logging
import utils.misc as misc
import os
import mxnet as mx
import utils.callbacks as callbacks
import utils.metrics as metrics

def train_multi_wrapper(ctx, symbol, snapshot_prefix, init_weight_file, im_folder, multi_label_file, class_num, rgb_mean,
                epoch_size, max_epoch, input_size, batch_size, lr, wd, momentum, lr_decay, workspace):

    train_symbol = symbol.create_train(class_num, workspace)
    data_iter = MultiLabelIter(image_root=im_folder,
                               label_file=multi_label_file,
                               num_class=class_num,
                               rgb_mean=rgb_mean,
                               epoch_size=epoch_size,
                               im_size=input_size,
                               shuffle=True,
                               random_flip=True,
                               batch_size=batch_size)

    if not os.path.exists(init_weight_file):
        logging.error("no file found for %s", init_weight_file)
        return

    arg_dict, aux_dict, _ = misc.load_checkpoint(init_weight_file)


    initializer = mx.initializer.Normal()
    initializer.set_verbosity(True)


    mod = mx.mod.Module(train_symbol,
                        context=ctx,
                        data_names=["data"],
                        label_names=["label"])

    mod.bind(data_shapes=data_iter.provide_data,
             label_shapes=data_iter.provide_label)
    mod.init_params(initializer=initializer, arg_params=arg_dict, aux_params=aux_dict, allow_missing=True)

    opt_params = {"learning_rate": lr,
                  "wd": wd,
                  'momentum': momentum,
                  'lr_scheduler': mx.lr_scheduler.FactorScheduler(step=lr_decay, factor=0.1),
                  'rescale_grad': 1.0 / batch_size}

    eval_metrics = [metrics.MultiLogisticLoss()]

    mod.fit(data_iter,
            optimizer="sgd",
            optimizer_params=opt_params,
            num_epoch=max_epoch+1,
            epoch_end_callback=callbacks.module_checkpoint(snapshot_prefix, 1),
            batch_end_callback=callbacks.Speedometer(batch_size, frequent=10),
            eval_metric=eval_metrics,
            begin_epoch=1)