from easydict import EasyDict as EDict
conf = EDict()

conf.BASE_NET = "vgg16" #vgg16 or resnet50
conf.CACHE_PATH = "cache"

#for dataset
#SBD_PATH is the one named "dataset", which has "cls", "img", "inst", "train.txt" and "val.txt".
#please download at https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz
#VOCDEVKIT_PATH is the one named "VOC2012", which has "Annotations", "ImageSets", "JPEGImages", "SegmentationClass",
conf.DATASET_PATH = "dataset"
conf.SBD_PATH = "dataset/benchmark_RELEASE/dataset"
conf.VOCDEVKIT_PATH = "dataset/VOCdevkit/VOC2012"
conf.VOC_TRAIN_MULTI_FILE = "voc_multi_file.p"
conf.VOC_TRAIN_IM_FOLDER = "train_images"
conf.VOC_VAL_IM_FOLDER = "val_images"
conf.VOC_VAL_MASK_FOLDER = "val_masks"
conf.VOC_TRAIN_LIST = "train_list.txt"
conf.VOC_VAL_LIST = "val_list.txt"

conf.LOG_FOLDER = "log"
conf.SNAPSHOT_FOLDER = "snapshots"
conf.OUTPUT_FOLDER = "outputs"

conf.CLASS_NUM = 21 # in this case voc dataset.
conf.MEAN_RGB = (123, 117, 104) #RGB not BGR

#training params for FG BG cue networks.
conf.EPOCH_SIZE_FG = 8000
conf.BATCH_SIZE_FG = 15
conf.LR_FG = 1e-3
conf.LR_DECAY_FG = 2000
conf.FG_CUE_FILE = "fg_cue_initsec.p"
conf.SALIENCY_TH_FG = 0.2

conf.LR_BG = 1e-3
conf.EPOCH_SIZE_BG = 8000
conf.BATCH_SIZE_BG = 15
conf.LD_DECAY_BG = 2000
conf.BG_CUE_FILE = "bg_cue_initsec.p"
conf.SALIENCY_TH_BG = 0.1

#training params for SEC models
conf.CUE_FILE_INITSEC = "sec_cue.p"
conf.INPUT_SIZE_SEC = 320
conf.DOWN_SAMPLE_SEC = 8 #network resolution
conf.Q_FG = 0.996
conf.Q_BG = 0.999


conf.LR = 1e-3
conf.LR_DECAY = 2000
conf.MAX_EPOCH = 8
conf.EPOCH_SIZE = 1000
conf.BATCH_SIZE = 15

conf.WD = 5e-4
conf.MOMENTUM = 0.9
conf.WORKSPACE = 512
conf.CPU_WORKER_NUM = 8

conf.CRF_POS_XY_STD = 3
conf.CRF_POS_W = 3
conf.CRF_BI_RGB_STD = 10
conf.CRF_BI_XY_STD = 80
conf.CRF_BI_W = 10


