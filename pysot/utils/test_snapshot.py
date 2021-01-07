import glob
import torch
from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.tracker_builder import build_tracker
import cv2
import os
from pathlib import Path
import random

def test_snapshot(epoch:int, snapshot:str, test_path:str):
    # model
    max_img = 2
    model = ModelBuilder()
    data = torch.load(snapshot,
        map_location=lambda storage, loc: storage.cpu())
    model.load_state_dict(data['state_dict'])
    model.eval().to(torch.device('cpu'))
    tracker = build_tracker(model)
    
    root = cfg.DATASET.COCO.ROOT
    cur_path = os.path.dirname(os.path.realpath(__file__))
    root = os.path.join(cur_path, '../../', root)
    dataset = os.path.join(root, "val2017")
    folder = random.choice(glob.glob(f"{dataset}/**")) 
    zs = glob.glob(f"{folder}/*.z.jpg")
    xs = glob.glob(f"{folder}/*.x.jpg")

    zs = sorted(zs)
    xs = sorted(xs)

    for i in range(len(zs[:max_img])):
        z = cv2.imread(zs[i])
        x = cv2.imread(xs[i])
        tracker.init(z)
        cls, (x1, y1, x2, y2) = tracker.track(x)
        img = cv2.imread(xs[i])
        cv2.rectangle(img, (x1, y1), (x2, y2), (255,0,0), 2)
        cv2.putText(img, 'Acc: ' + cls.astype('str'), (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
        parent_dir = f"{test_path}/{os.path.basename(Path(zs[i]).parent)}"
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir)
        cv2.imwrite(f"{parent_dir}/{os.path.basename(xs[i])}", img)
        cv2.imwrite(f"{parent_dir}/{os.path.basename(zs[i])}", z)