
# coding: utf-8

# In[1]:


from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *

import os
import sys
import time
import datetime
import argparse
import cv2

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator
import pickle as pkl

from PIL import Image


# In[2]:
parser = argparse.ArgumentParser()

parser.add_argument('--video_path', type=str, default='./test.264', help='path to video')
opt = parser.parse_args()

video_folder = opt.video_path
config_path = "/home/USERNAME/jupyter/config/yolov3.cfg"
weights_path = "/home/USERNAME/jupyter/weights/yolov3.weights"
class_path = "/home/USERNAME/jupyter/data/coco.names"
conf_thres = 0.8
nms_thres = 0.4
batch_size = 1
n_cpu = 8
img_size = 416
use_cuda = True

os.makedirs('output', exist_ok=True)

def buildMap(Ws,Hs,Wd,Hd,R1,R2,Cx,Cy):
    map_x = np.zeros((Hd,int(Wd)),np.float32)
    map_y = np.zeros((Hd,int(Wd)),np.float32)
    for y in range(0,int(Hd-1)):
        for x in range(0,int(Wd-1)):
            r = (float(y)/float(Hd))*(R2-R1)+R1
            theta = (float(x)/float(Wd))*2.0*np.pi
            xS = Cx+r*np.sin(theta)
            yS = Cy+r*np.cos(theta)
            map_x.itemset((y,x),int(xS))
            map_y.itemset((y,x),int(yS))
        
    return map_x, map_y
    
def unwarp(img,xmap,ymap):
    output = cv2.remap(np.asarray(img),xmap,ymap,cv2.INTER_LINEAR)
    output = cv2.flip(output, 0)
    result = Image.fromarray(output)
    return result

def omni_rect(img):
    Cx = 208
    Cy = 208
    R1x = 250
    R1y = 208
    R1 = R1x-Cx
    R2x = 420
    R2y = 208
    R2 = R2x-Cx
    Wd = 2.0*((R2+R1)/2)*np.pi
    Hd = (R2-R1)
    Ws = 416
    Hs = 416

    xmap,ymap = buildMap(Ws,Hs,Wd,Hd,R1,R2,Cx,Cy)

    result = unwarp(img,xmap,ymap)

    return result

# In[3]:


cuda = torch.cuda.is_available() and use_cuda
    
model = Darknet(config_path, img_size)
model.load_weights(weights_path)

if cuda:
    model.cuda()
        
model.eval()
    
classes = load_classes(class_path)
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    
img_detections = []
imgs = []
videofile = video_folder
    
cap = cv2.VideoCapture(videofile)
    
assert cap.isOpened(), 'Cannot capture source'
    
count = 0;

while cap.isOpened():
    ret, frame = cap.read()
    if count == cap.get(cv2.CAP_PROP_POS_FRAMES):
        break
    count = cap.get(cv2.CAP_PROP_POS_FRAMES)
    if (count % 22) == 0:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        img = img.resize((img_size, img_size))
#         img = omni_rect(img)
        imgs.append(img)
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break
        
dataloader = DataLoader(VideoLoader(imgs, img_size), batch_size, shuffle=False,num_workers=n_cpu)

for batch_i, input_imgs in enumerate(dataloader):
    input_imgs = Variable(input_imgs.type(Tensor))
        
    with torch.no_grad():
        detections = model(input_imgs)
        detections = non_max_suppression(detections, 80, conf_thres, nms_thres)
            
    img_detections.extend(detections)

cmap = plt.get_cmap('tab20b')
colors = [cmap(i) for i in np.linspace(0, 1, 20)]

mkdir = 'output/%s' % (videofile[36:len(videofile)-4])
os.makedirs(mkdir, exist_ok=True)
opendir = '%s/result.txt' %(mkdir)
result_file = open(opendir, 'a')

for img_i, (imgs, detections) in enumerate(zip(imgs, img_detections)):

    img = np.array(imgs)
    plt.figure()
    fig, ax = plt.subplots(1)
    ax.imshow(img)

    pad_x = max(img.shape[0] - img.shape[1], 0) * (img_size / max(img.shape))
    pad_y = max(img.shape[1] - img.shape[0], 0) * (img_size / max(img.shape))

    unpad_h = img_size - pad_y
    unpad_w = img_size - pad_x

    
    if detections is not None:
        unique_labels = detections[:, -1].cpu().unique()
        n_cls_preds = len(unique_labels)
        bbox_colors = random.sample(colors, n_cls_preds)
        for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:

            write_buf = 'Label: %s, Conf: %.5f \t' % (classes[int(cls_pred)], cls_conf.item())

            result_file.write(write_buf)

            # if classes[int(cls_pred)] == 'person':
            #     print('event')
            
            box_h = ((y2 - y1) / unpad_h) * img.shape[0]
            box_w = ((x2 - x1) / unpad_w) * img.shape[1]
            y1 = ((y1 - pad_y // 2) / unpad_h) * img.shape[0]
            x1 = ((x1 - pad_x // 2) / unpad_w) * img.shape[1]

            color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
            bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2,
                                    edgecolor=color,
                                    facecolor='none')
            ax.add_patch(bbox)
            plt.text(x1, y1, s=classes[int(cls_pred)], color='white', verticalalignment='top',
                    bbox={'color': color, 'pad': 0})
        result_file.write('\n')
    plt.axis('off')
    plt.gca().xaxis.set_major_locator(NullLocator())
    plt.gca().yaxis.set_major_locator(NullLocator())
    plt.savefig('%s/%d.png' % (mkdir, img_i), bbox_inches='tight', pad_inches=0.0)
    plt.close()
result_file.close()

