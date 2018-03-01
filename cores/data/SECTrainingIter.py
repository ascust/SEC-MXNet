from Queue import Queue
import cPickle as pickle
from threading import Thread
from mxnet.io import DataIter, DataBatch
import mxnet as mx
import os
import logging
from PIL import Image
import numpy as np
import random

class SECTrainingIter(DataIter):
    def __init__(self,
                 im_root,
                 cue_file_path,
                 class_num,
                 rgb_mean=(128, 128, 128),
                 im_size=320,
                 shuffle=True,
                 label_shrink_scale=1/8.0,
                 random_flip=True,
                 data_queue_size=8,
                 epoch_size=-1,
                 batch_size=1,
                 round_batch=True):
        super(SECTrainingIter, self).__init__()
        self.rgb_mean = mx.nd.array(rgb_mean, ctx=mx.cpu()).reshape((1, 3, 1, 1))
        self.im_size = im_size
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.class_num = class_num
        self.label_shrink_scale = label_shrink_scale
        self.flist = None
        self.cue_dict = None
        self.im_root = im_root
        self.cue_file_path = cue_file_path
        self.load_cue_file(cue_file_path)
        self.round_batch = round_batch

        self.data_num = self.get_data_num()
        self.iter_count = 0
        self.cursor = 0
        self.reset()
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
            mask_dim = int(self.im_size * self.label_shrink_scale)
            images = mx.nd.zeros((self.batch_size, 3, self.im_size, self.im_size), ctx=mx.cpu())
            labels = mx.nd.zeros((self.batch_size, 1, 1, self.class_num), ctx=mx.cpu())
            cues = mx.nd.zeros((self.batch_size, self.class_num, mask_dim, mask_dim), ctx=mx.cpu())
            small_ims = mx.nd.zeros((self.batch_size, mask_dim, mask_dim, 3), ctx=mx.cpu())
            sub_list = self.flist_item_queue.get()

            batch_images = []
            for ind, item_name in enumerate(sub_list):
                buf = mx.nd.array(
                    np.frombuffer(open(os.path.join(self.im_root, item_name + ".jpg"), 'rb').read(), dtype=np.uint8),
                    dtype=np.uint8, ctx=mx.cpu())
                im = mx.image.imdecode(buf)
                tmp = self.cue_dict[item_name + "_cues"]
                cue = np.zeros((self.class_num, mask_dim, mask_dim))
                cue[tmp[0], tmp[1], tmp[2]] = 1

                if self.random_flip and np.random.rand() > 0.5:
                    im = mx.nd.flip(im, axis=1)
                    cue = cue[:, :, ::-1]
                batch_images.append(im)
                label = np.zeros(self.class_num)
                label[self.cue_dict[item_name + "_labels"]] = 1
                labels[ind][:] = label.reshape(1, 1, -1)
                cues[ind][:] = cue

            batch_images = [mx.image.imresize(im, self.im_size, self.im_size, interp=1) for im in batch_images]
            sm_images = [mx.image.imresize(im, mask_dim, mask_dim, interp=1) for im in batch_images]

            for i in range(len(sub_list)):
                images[i][:] = mx.nd.transpose(batch_images[i], (2, 0, 1))
                small_ims[i][:] = sm_images[i]

            images -= self.rgb_mean


            self.data_queue.put(DataBatch(data=[images, small_ims], label=[labels, cues], pad=None, index=None))




    def get_data_num(self):
        return len(self.flist)

    def load_cue_file(self,
                      cue_file_path):
        with open(cue_file_path, "rb") as f:
            self.cue_dict = pickle.load(f)
            self.flist = []
            for key in self.cue_dict.keys():
                if key.endswith("_cues"):
                    self.flist.append(str(key[0:key.rfind("_cues")]))
            self.data_num = len(self.flist)
            if self.shuffle:
                random.shuffle(self.flist)

    def _process_data(self, filename):
        try:
            im = Image.open(os.path.join(self.im_root, filename+".jpg"))
            im = im.convert("RGB")
        except Exception as e:
            logging.error(e)
            return None

        im_arr = np.array(im.resize((self.im_size, self.im_size), Image.BICUBIC))
        mask_dim = int(self.im_size * self.label_shrink_scale)
        label = np.zeros(self.class_num)
        cue = np.zeros((self.class_num, mask_dim, mask_dim))
        sm_im = np.array(im.resize((mask_dim, mask_dim), Image.BICUBIC))
        im_arr = np.array(im_arr, dtype=np.float32)
        sm_im_arr = np.array(sm_im, dtype=np.float32)

        label[self.cue_dict[filename+"_labels"]] = 1
        label = label.reshape(1, 1, -1)
        tmp = self.cue_dict[filename+"_cues"]
        cue[tmp[0], tmp[1], tmp[2]] = 1

        if self.random_flip and np.random.random()>0.5:
            im_arr = im_arr[:,::-1,:]
            sm_im_arr = sm_im_arr[:,::-1,:]
            cue = cue[:, :, ::-1]

        im_arr -= self.rgb_mean
        im_arr = np.transpose(im_arr, [2, 0, 1])

        return (im_arr, label, cue, sm_im_arr)

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
        label_dim = int(self.im_size * self.label_shrink_scale)
        return [("data", (self.batch_size, 3, self.im_size, self.im_size)),
                 ("small_ims", (self.batch_size, label_dim, label_dim, 3))]

        # return [("data", (self.batch_size, 3, self.im_size, self.im_size))]

    @property
    def provide_label(self):
        label_dim = int(self.im_size * self.label_shrink_scale)

        return [("labels", (self.batch_size, 1, 1, self.class_num)),
        ("cues", (self.batch_size, self.class_num, label_dim, label_dim))]
