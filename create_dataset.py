import os
from cores.config import conf
import scipy.io as sio
import numpy as np
import cores.utils.misc as misc
import shutil
from PIL import Image
import cPickle as pickle

#convert SBD data and VOC12 data to our format.
if __name__ == "__main__":

    misc.my_mkdir(conf.DATASET_PATH)
    misc.my_mkdir(os.path.join(conf.DATASET_PATH, conf.VOC_TRAIN_IM_FOLDER))
    misc.my_mkdir(os.path.join(conf.DATASET_PATH, conf.VOC_VAL_IM_FOLDER))
    misc.my_mkdir(os.path.join(conf.DATASET_PATH, conf.VOC_VAL_MASK_FOLDER))

    # process SBD
    sbd_list = []
    with open(os.path.join(conf.SBD_PATH, "train.txt")) as f:
        sbd_list += [i.strip() for i in f.readlines()]
    with open(os.path.join(conf.SBD_PATH, "val.txt")) as f:
        sbd_list += [i.strip() for i in f.readlines()]

    with open(os.path.join(conf.VOCDEVKIT_PATH, "ImageSets", "Segmentation", "train.txt")) as f:
        voc_train_list = [i.strip() for i in f.readlines()]
    with open(os.path.join(conf.VOCDEVKIT_PATH, "ImageSets", "Segmentation", "val.txt")) as f:
        voc_val_list = [i.strip() for i in f.readlines()]

    new_sbd_list = []
    for i in sbd_list:
        if i in voc_train_list or i in voc_val_list:
            continue
        new_sbd_list.append(i)

    train_data_dict = {}
    #for training set, only extract image level labels
    for index, i in enumerate(new_sbd_list):
        mask = sio.loadmat(os.path.join(conf.SBD_PATH, "cls", i+".mat"))['GTcls']['Segmentation'][0][0]
        il = np.unique(mask)
        # 0 is bg, so in multi-label file, the bg is removed. VOC 21 classes become 20 classes.
        image_labels = il[(il!=255)&(il!=0)] - 1
        train_data_dict[i] = image_labels
        shutil.copyfile(os.path.join(conf.SBD_PATH, "img", i+".jpg"),
                        os.path.join(conf.DATASET_PATH, conf.VOC_TRAIN_IM_FOLDER, i+".jpg"))
        print "processed %s in SBD\t%d/%d" % (i, index, len(new_sbd_list))
    for index, i in enumerate(voc_train_list):
        mask = Image.open(os.path.join(conf.VOCDEVKIT_PATH, "SegmentationClass", i+".png"))
        il = np.unique(mask)
        image_labels = il[(il != 255) & (il != 0)] - 1
        train_data_dict[i] = image_labels
        shutil.copyfile(os.path.join(conf.VOCDEVKIT_PATH, "JPEGImages", i+".jpg"),
                        os.path.join(conf.DATASET_PATH, conf.VOC_TRAIN_IM_FOLDER, i+".jpg"))
        print "processed %s in VOC training set\t%d/%d" % (i, index, len(voc_train_list))

    #for val set, save both masks and images
    for index, i in enumerate(voc_val_list):
        shutil.copyfile(os.path.join(conf.VOCDEVKIT_PATH, "JPEGImages", i+".jpg"),
                        os.path.join(conf.DATASET_PATH, conf.VOC_VAL_IM_FOLDER, i+".jpg"))
        shutil.copyfile(os.path.join(conf.VOCDEVKIT_PATH, "SegmentationClass", i+".png"),
                        os.path.join(conf.DATASET_PATH, conf.VOC_VAL_MASK_FOLDER, i+".png"))
        print "processed %s in VOC val set\t%d/%d" % (i, index, len(voc_val_list))

    #save file list and multi-label file
    print "saving files"
    pickle.dump(train_data_dict, open(os.path.join(conf.DATASET_PATH, conf.VOC_TRAIN_MULTI_FILE), "wb"))

    with open(os.path.join(conf.DATASET_PATH, conf.VOC_TRAIN_LIST), "w") as f:
        for i in (new_sbd_list+voc_train_list):
            f.write("%s\n" % i)

    with open(os.path.join(conf.DATASET_PATH, conf.VOC_VAL_LIST), "w") as f:
        for i in voc_val_list:
            f.write("%s\n" % i)
    print "done!"
