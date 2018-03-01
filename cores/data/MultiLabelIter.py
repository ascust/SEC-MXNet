from Queue import Queue
from threading import Thread
from mxnet.io import DataIter, DataBatch
import mxnet as mx
import os
import numpy as np
import cPickle as pickle

class MultiLabelIter(DataIter):
    def __init__(self,
                 image_root,
                 label_file,
                 num_class,
                 rgb_mean=(128, 128, 128),
                 im_size=320,
                 shuffle=False,
                 random_flip=False,
                 data_queue_size=8,
                 epoch_size=-1,
                 batch_size=1,
                 round_batch=True):
        super(MultiLabelIter, self).__init__()
        self.rgb_mean = mx.nd.array(rgb_mean, dtype=np.float32, ctx=mx.cpu()).reshape((1, 3, 1, 1))
        self.num_class = num_class
        self.im_size = im_size
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.flist = None
        self.image_root = image_root
        self.data_dict = None
        self._load_flist(label_file)
        self.data_num = self.get_data_num()
        self.iter_count = 0
        self.cursor = 0
        self.reset()
        self.round_batch = round_batch
        self.random_flip = random_flip


        if epoch_size == -1:
            self.epoch_size = int(self.data_num/self.batch_size)
        else:
            self.epoch_size = epoch_size
        
        self.flist_item_queue = Queue(maxsize=1000)
        list_producer = Thread(target=self._produce_flist_item)
        list_producer.daemon = True
        list_producer.start()

        self.data_queue = Queue(maxsize=data_queue_size)

        producer = Thread(target=self._produce_data)
        producer.daemon = True
        producer.start()



    def _produce_flist_item(self):
        while True:
            if self.cursor + self.batch_size < self.data_num:
                sub_list = self.flist[self.cursor:self.cursor+self.batch_size]
                self.cursor += self.batch_size
            else:
                if self.round_batch:
                    sub_list = self.flist[self.cursor:self.data_num]
                    sub_list += self.flist[0:(self.batch_size - len(sub_list))]
                    self.cursor = 0
                    if self.shuffle:
                        np.random.shuffle(self.flist)
                else:
                    if self.shuffle:
                        np.random.shuffle(self.flist)
                    sub_list = self.flist[0:self.batch_size]
                    self.cursor = self.batch_size
            self.flist_item_queue.put(sub_list)

    def _produce_data(self):
        while True:
            images = mx.nd.zeros((self.batch_size, 3, self.im_size, self.im_size), ctx=mx.cpu())
            labels = mx.nd.zeros((self.batch_size, self.num_class), ctx=mx.cpu())
            sub_list = self.flist_item_queue.get()

            batch_images = []
            for ind, item_name in enumerate(sub_list):
                buf = mx.nd.array(
                    np.frombuffer(open(os.path.join(self.image_root, item_name+".jpg"), 'rb').read(), dtype=np.uint8),
                    dtype=np.uint8, ctx=mx.cpu())
                im = mx.image.imdecode(buf)
                if self.random_flip and np.random.rand() > 0.5:
                    im = mx.nd.flip(im, axis=1)
                batch_images.append(im)
                l = np.zeros(self.num_class)
                l[np.array(self.data_dict[item_name], dtype=np.int32)] = 1
                labels[ind][:] = l

            batch_images = [mx.image.imresize(im, self.im_size, self.im_size, interp=1) for im in batch_images]

            for i in range(len(sub_list)):
                images[i][:] = mx.nd.transpose(batch_images[i], (2, 0, 1))

            images -= self.rgb_mean

            self.data_queue.put(DataBatch(data=[images], label=[labels], pad=None, index=None))



    def get_data_num(self):
        return len(self.flist)

    def _load_flist(self, path):
        with open(path, 'rb') as f:
            self.data_dict = pickle.load(f)
            self.flist = []
            for i in self.data_dict.keys():
                self.flist.append(i)



    
    def reset(self):
        self.iter_count = 0

    def iter_next(self):
        return self.iter_count < self.epoch_size

    def next(self):
        if self.iter_next():
            self.iter_count += 1
            return self.data_queue.get()
        else:
            raise StopIteration

    @property
    def provide_data(self):
        """The name and shape of data provided by this iterator"""
        return [("data", (self.batch_size, 3, self.im_size, self.im_size))]

    @property
    def provide_label(self):
        """The name and shape of label provided by this iterator"""
        return [("label", (self.batch_size, self.num_class))]

