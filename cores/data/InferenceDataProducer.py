import numpy as np
from threading import Thread
from Queue import Queue
from PIL import Image
import os
import mxnet as mx

class InferenceDataProducer(object):
    def __init__(self,
                 im_root,
                 mask_root,
                 flist_path,
                 rgb_mean=(128, 128, 128),
                 data_queue_size=100,
                 input_size=320):
        self.flist = None
        self.input_size = input_size
        self.im_root = im_root
        self.mask_root = mask_root
        self._load_flist(flist_path)
        self.data_num = self.get_data_num()
        self.avail_data_num = self.data_num
        self.cursor = 0
        self.rgb_mean = mx.nd.array(rgb_mean, dtype=np.float32, ctx=mx.cpu()).reshape((1, 1, 3))

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
            if self.cursor + 1 <= self.data_num:
                file = self.flist[self.cursor]
                self.flist_item_queue.put(file)
                self.cursor += 1
            else:
                return

    def _produce_data(self):
        while True:
            flist_item = self.flist_item_queue.get()
            value = self._process_data(flist_item)
            if value is not None:
                self.data_queue.put(value)
            else:
                raise AssertionError("file error: %s"%flist_item[0])

    def _process_data(self, item):
        buf = mx.nd.array(np.frombuffer(open(item[0], 'rb').read(), dtype=np.uint8), dtype=np.uint8, ctx=mx.cpu())
        orig_im = mx.image.imdecode(buf)

        h, w = orig_im.shape[:2]

        tmp_im = mx.image.imresize(orig_im, self.input_size, self.input_size, interp=1)
        tmp_im = tmp_im.astype(np.float32)
        tmp_im -= self.rgb_mean
        tmp_im = mx.nd.transpose(tmp_im, [2, 0, 1])
        tmp_im = mx.nd.expand_dims(tmp_im, 0)


        if item[1] is None:
            l_arr = np.zeros((h, w), dtype=np.uint8)
        else:
            l = Image.open(item[1])
            l_arr = np.array(l, dtype=np.uint8)

        return (tmp_im, l_arr, item[2], orig_im)

    def get_data(self):
        if self.avail_data_num>0:
            self.avail_data_num -= 1
            data = self.data_queue.get()
            return data
        else:
            return None

    def get_data_num(self):
        return len(self.flist)

    def _load_flist(self,
                   flist_path):
        with open(flist_path) as f:
            lines = f.readlines()
            self.flist = []
            for line in lines:
                if len(line.rstrip()) == 0:
                    continue
                item = self._parse_flist_item(line.rstrip())
                self.flist.append(item)
            self.data_num = len(self.flist)

    def _parse_flist_item(self, line):
        item_name = line
        im_name = item_name+".jpg"
        im_path = os.path.join(self.im_root, im_name)
        l_path = None
        if os.path.exists(os.path.join(self.mask_root, item_name+".png")):
            l_path = os.path.join(self.mask_root, item_name+".png")
        return (im_path, l_path, item_name)





