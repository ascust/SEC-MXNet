from Queue import Queue
from threading import Thread
from mxnet.io import DataIter, DataBatch
import mxnet as mx
import os
import numpy as np
import cPickle as pickle

class ImageFetcher(DataIter):
    def __init__(self,
                 image_root,
                 label_file,
                 num_class,
                 rgb_mean=(128, 128, 128),
                 im_size=320,
                 data_queue_size=8,
                 batch_size=1):
        super(ImageFetcher, self).__init__()
        self.rgb_mean = mx.nd.array(rgb_mean).reshape((1, 3, 1, 1))
        self.im_size = im_size
        self.batch_size = batch_size

        self.flist = None
        self.num_class = num_class
        self.image_root = image_root
        self.data_dict = None
        self._load_flist(label_file)
        self.data_num = self.get_data_num()
        self.batch_count = 0
        self.max_batch_num = int(np.ceil(float(self.data_num)/self.batch_size))
        self.cursor = 0
        self.reset()

        
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
            if self.cursor<self.data_num:
                if self.cursor + self.batch_size < self.data_num:
                    sub_list = self.flist[self.cursor:self.cursor+self.batch_size]
                    self.cursor += self.batch_size
                else:
                    sub_list = self.flist[self.cursor:self.data_num]
                    self.cursor += self.data_num
                self.flist_item_queue.put(sub_list)
            else:
                return

    def _produce_data(self):
        while True:

            images = mx.nd.zeros((self.batch_size, 3, self.im_size, self.im_size), ctx=mx.cpu())
            labels = mx.nd.zeros((self.batch_size, self.num_class), ctx=mx.cpu())
            sub_list = self.flist_item_queue.get()

            batch_images = []
            for ind, item_name in enumerate(sub_list):
                buf = mx.nd.array(
                    np.frombuffer(open(os.path.join(self.image_root, item_name + ".jpg"), 'rb').read(), dtype=np.uint8),
                    dtype=np.uint8, ctx=mx.cpu())
                im = mx.image.imdecode(buf)
                batch_images.append(im)
                l = np.zeros(self.num_class)
                l[np.array(self.data_dict[item_name], dtype=np.int32)] = 1
                labels[ind][:] = l

            batch_images = [mx.image.imresize(im, self.im_size, self.im_size, interp=1) for im in batch_images]

            for i in range(len(sub_list)):
                images[i][:] = mx.nd.transpose(batch_images[i], (2, 0, 1))

            images -= self.rgb_mean

            self.data_queue.put((images, sub_list, labels))



    def get_batch_num(self):
        return self.max_batch_num

    def get_data_num(self):
        return len(self.flist)

    def _load_flist(self, path):
        with open(path, 'rb') as f:
            self.data_dict = pickle.load(f)
            self.flist = self.data_dict.keys()



    def iter_next(self):
        return self.batch_count < self.max_batch_num

    def next(self):
        if self.iter_next():
            self.batch_count += 1
            return self.data_queue.get()
        else:
            raise StopIteration

