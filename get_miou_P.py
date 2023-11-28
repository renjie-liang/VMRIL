import os
import re
import json
import numpy as np
from utils_weak import miou_two_dataset

task, B, T, max_len = "charades", 30, 5, 64
# task, B, T, max_len = "anet",     30, 15, 100
task, B, T, max_len = "tacos", 30, 10, 256

gt_path = './data/{}_gt/train.json'.format(task)
seed_path = './data/{}_P{}_RE{}/train.json'.format(task, B, 0)


for I in range(0, 10):
    if I == 0:
        SUFFIX = "P{}_RE{}".format(B, I)
    else: 
        SUFFIX = "P{}_T{}_RE{}".format(B, T, I)
    
    train_path = './data/{}_{}/train.json'.format(task, SUFFIX)
    path = "./ckpt/{}/model_i3d_{}_{}/model".format(task, max_len, SUFFIX)

    iou = miou_two_dataset(gt_path, train_path)
    iou_ = miou_two_dataset(seed_path, train_path)


    best_path = os.path.join(path, "bestepoch")
    with open(best_path, "r") as f:
        best_epoch = f.readline()
        # print(best_epoch)
    eval_res = os.listdir(path)
    eval_res = [i for i in eval_res if i[-3:]=="txt"]
    eval_res = sorted(eval_res)

    eval_path = os.path.join(path, eval_res[-1])
    with open(eval_path, "r") as f:
        a = f.readlines()

    for i in range(len(a)):
        if a[i].startswith("Epoch {},".format(best_epoch)):
            best_line = a[i+1]
            R357 = re.findall(r"\s\d+\.?\d*", best_line)
            res_print = "{:.4f}\t{:.4f}\t{}".format(iou, iou_, "\t ".join(R357))
            print(res_print)
    # print(eval_res)
    # best_epoch 
    # print(path)
