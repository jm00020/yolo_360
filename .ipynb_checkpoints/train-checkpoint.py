
# coding: utf-8

# In[1]:


from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *

import os
import sys
import time
import datetime
import argparse

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim


# In[2]:


epochs = 30
image_folder = "D:/Downloads/PyTorch-YOLOv3-master (2)/PyTorch-YOLOv3-master/data/sample"
batch_size = 16
model_config_path = "config/yolov3.cfg"
data_config_path = "config/coco.data"
weights_path = "weights/yolov3.weights"
class_path = "D:/Downloads/PyTorch-YOLOv3-master (2)/PyTorch-YOLOv3-master/data/coco.names"
conf_thres = 0.8
nms_thres = 0.4
n_cpu = 10
img_size = 416
checkpoint_interval = 1
checkpoint_dir = "checkpoints"
use_cuda = True

cuda = torch.cuda.is_available() and use_cuda

os.makedirs("output", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

classes = load_classes(class_path)


# In[ ]:


data_config = parse_data_config(data_config_path)
train_path = data_config["train"]

hyperparams = parse_model_config(model_config_path)[0]
learning_rate = float(hyperparams["learning_rate"])
momentum = float(hyperparams["momentum"])
decay = float(hyperparams["decay"])
burn_in = int(hyperparams["burn_in"])

model = Darknet(model_config_path)
model.load_weights(weights_path)
# model.apply(weights_init_normal)

if cuda:
    model = model.cuda()

model.train()

dataloader = torch.utils.data.DataLoader(
    ListDataset(train_path), batch_size, shuffle=False, num_workers=n_cpu
)

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))

for epoch in range(epochs):
    for batch_i, (_, imgs, targets) in enumerate(dataloader):
        imgs = Variable(imgs.type(Tensor))
        targets = Variable(targets.type(Tensor), requires_grad=False)

        optimizer.zero_grad()

        loss = model(imgs, targets)

        loss.backward()
        optimizer.step()

        print(
            "[Epoch %d/%d, Batch %d/%d] [Losses: x %f, y %f, w %f, h %f, conf %f, cls %f, total %f, recall: %.5f, precision: %.5f]"
            % (
                epoch,
                epochs,
                batch_i,
                len(dataloader),
                model.losses["x"],
                model.losses["y"],
                model.losses["w"],
                model.losses["h"],
                model.losses["conf"],
                model.losses["cls"],
                loss.item(),
                model.losses["recall"],
                model.losses["precision"],
            )
        )

        model.seen += imgs.size(0)

    if epoch % checkpoint_interval == 0:
        model.save_weights("%s/%d.weights" % (checkpoint_dir, epoch))

