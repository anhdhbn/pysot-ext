import glob
import torch
from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.tracker_builder import build_tracker
import cv2
import os
from pathlib import Path
import random
import json
import numpy as np
from pysot.utils.bbox import center2corner, Center

def get_anno_from_img_path(anno, img_path):
    img_name = os.path.basename(img_path)
    p = Path(img_path)
    p1 = os.path.basename(p.parent)
    p2 = os.path.basename(p.parent.parent)
    k = f"{p2}/{p1}"
    
    for key in anno[k]:
        for key2 in anno[k][key]:
            tmp = f"{key2}.{key}.x.jpg"
            if tmp == img_name:
                image = cv2.imread(img_path)
                a1, b1, a2, b2 = get_bbox(image, anno[k][key][key2])
                a1 = int(a1)
                b1 = int(b1)
                a2 = int(a2)
                b2 = int(b2)
                return [a1, b1, a2, b2]

def filter_zero(meta_data):
    meta_data_new = {}
    for video, tracks in meta_data.items():
        new_tracks = {}
        for trk, frames in tracks.items():
            new_frames = {}
            for frm, bbox in frames.items():
                if not isinstance(bbox, dict):
                    if len(bbox) == 4:
                        x1, y1, x2, y2 = bbox
                        w, h = x2 - x1, y2 - y1
                    else:
                        w, h = bbox
                    if w <= 0 or h <= 0:
                        continue
                new_frames[frm] = bbox
            if len(new_frames) > 0:
                new_tracks[trk] = new_frames
        if len(new_tracks) > 0:
            meta_data_new[video] = new_tracks
    return meta_data_new

def get_bbox(image, shape):
    imh, imw = image.shape[:2]
    if len(shape) == 4:
        w, h = shape[2]-shape[0], shape[3]-shape[1]
    else:
        w, h = shape
    context_amount = 0.5
    exemplar_size = cfg.TRAIN.EXEMPLAR_SIZE
    wc_z = w + context_amount * (w+h)
    hc_z = h + context_amount * (w+h)
    s_z = np.sqrt(wc_z * hc_z)
    scale_z = exemplar_size / s_z
    w = w*scale_z
    h = h*scale_z
    cx, cy = imw//2, imh//2
    bbox = center2corner(Center(cx, cy, w, h))
    return bbox

def test_snapshot(epoch:int, snapshot:str, test_path:str):
    # model
    max_img = 8
    model = ModelBuilder()
    data = torch.load(snapshot,
        map_location=lambda storage, loc: storage.cpu())
    model.load_state_dict(data['state_dict'])
    model.eval().to(torch.device('cpu'))
    tracker = build_tracker(model)
    
    root = cfg.DATASET.COCO.ROOT
    cur_path = os.path.dirname(os.path.realpath(__file__))
    root = os.path.join(cur_path, '../../', root)
    anno_path = os.path.join(root, '../', "val2017.json")
    with open(anno_path, 'r') as f:
        anno = json.load(f)
        anno = filter_zero(anno)
    dataset = os.path.join(root, "val2017")
    folder = random.choice(glob.glob(f"{dataset}/**")) 
    zs = glob.glob(f"{folder}/*.z.jpg")
    xs = glob.glob(f"{folder}/*.x.jpg")

    zs = sorted(zs)
    xs = sorted(xs)

    xs = [(x, get_anno_from_img_path(anno, x)) for x in xs]

    for i in range(len(zs[:max_img])):
        z = cv2.imread(zs[i])
        x_path, bbox = xs[i]
        x = cv2.imread(x_path)
        tracker.init_(z)
        cls, (x1, y1, x2, y2) = tracker.track(x)
        cv2.rectangle(x, (x1, y1), (x2, y2), (255,0,0), 2)
        a1, b1, a2, b2 = bbox
        cv2.rectangle(x, (a1, b1), (a2, b2), (0,0,255), 2)
        cv2.putText(x, 'Acc: ' + cls.astype('str'), (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
        parent_dir = f"{test_path}/{os.path.basename(Path(zs[i]).parent)}"
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir)
        cv2.imwrite(f"{parent_dir}/{os.path.basename(x_path)}", x)
        cv2.imwrite(f"{parent_dir}/{os.path.basename(zs[i])}", z)
