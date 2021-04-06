# -*- coding: utf-8 -*-
# @Author  : leizehua
import random

import cv2
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import ColorMode
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.coco import load_coco_json
from demo import predictor
TRAIN_JSON = '/home/leizehua/workspace/data/detectron2/datasets/widerface/annotations/wider_face_train_annot_coco_style.json'
TRAIN_PATH = '/home/leizehua/workspace/data/detectron2/datasets/widerface/WIDER_train/images/'
VAL_JSON = '/home/leizehua/workspace/data/detectron2/datasets/widerface/annotations/wider_face_val_annot_coco_style.json'
VAL_PATH = '/home/leizehua/workspace/data/detectron2/datasets/widerface/WIDER_val/images/'

DatasetCatalog.register("widerface_train", lambda: load_coco_json(TRAIN_JSON, TRAIN_PATH, "widerface_train"))
MetadataCatalog.get("widerface_train").set(thing_classes=["face"],
                                           json_file=TRAIN_JSON,
                                           image_root=TRAIN_PATH)

DatasetCatalog.register("widerface_val", lambda: load_coco_json(VAL_JSON, VAL_PATH, "widerface_val"))
MetadataCatalog.get("widerface_val").set(thing_classes=["face"],
                                         json_file=VAL_JSON,
                                         image_root=VAL_PATH)
meta_train = MetadataCatalog.get("widerface_train")
dataset_dicts = load_coco_json(TRAIN_JSON, TRAIN_PATH, "widerface_train")
count = 0
for d in random.sample(dataset_dicts, 10):
    im = cv2.imread(d["file_name"])
    v = Visualizer(im[:, :, ::-1],
                   metadata=meta_train,
                   scale=0.8,
                   instance_mode=ColorMode.IMAGE
                   )
    vis = v.draw_dataset_dict(d)
    count += 1
    cv2.imwrite('/home/leizehua/workspace/code/UniDet/tmp/' + str(count) + '.jpg', vis.get_image()[:, :, ::-1])



