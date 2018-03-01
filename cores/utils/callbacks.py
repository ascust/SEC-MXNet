import logging
import time
import misc


#callback func for saving checkpoint
def module_checkpoint(prefix, save_freq=1):
    def _callback(epoch_num, sym=None, arg=None, aux=None):
        if epoch_num%save_freq == 0:
            misc.save_checkpoint(prefix, epoch_num, symbol=sym, arg_params=arg, aux_params=aux)
    return _callback

class Speedometer(object):
    """Calculate and log training speed periodically.

    Parameters
    ----------
    batch_size: int
        batch_size of data
    frequent: int
        How many batches between calculations.
        Defaults to calculating & logging every 50 batches.
    """
    def __init__(self, batch_size, frequent=50):
        self.batch_size = batch_size
        self.frequent = frequent
        self.init = False
        self.tic = 0
        self.last_count = 0

    def __call__(self, param):
        """Callback to Show speed."""
        count = param.nbatch
        if self.last_count > count:
            self.init = False
        self.last_count = count

        if self.init:
            if count % self.frequent == 0:
                speed = self.frequent * self.batch_size / (time.time() - self.tic)
                if param.eval_metric is not None:
                    name_value = param.eval_metric.get_name_value()
                    # param.eval_metric.reset()
                    res_str = ""
                    cur_time = time.strftime("%d/%m/%Y--%H:%M:%S")
                    for name, value in name_value:
                        res_str += "\t%s=%f" % (name, value)

                    logging.info('Epoch[%d] Batch[%d]\tSpeed: %.2f samples/sec\t%s\ttime=%s', param.epoch, count, speed, res_str, cur_time)
                else:
                    logging.info("Iter[%d] Batch[%d]\tSpeed: %.2f samples/sec",
                                 param.epoch, count, speed)
                param.eval_metric.reset()
                self.tic = time.time()
        else:
            self.init = True
            self.tic = time.time()