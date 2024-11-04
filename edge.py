# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import base64
import os
import sys
import numpy as np
import argparse
import pprint
import pdb
import time
import cv2
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import pickle
import pdb
from torchvision import transforms
from ecci_sdk_cloud import Client
import threading

try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--arch', dest='arch', default='rfcn', choices=['rcnn', 'rfcn', 'couplenet'])
    parser.add_argument('--dataset', dest='dataset',
                        help='training dataset',
                        default='imagenet_vid', type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='cfgs/res16.yml', type=str)
    parser.add_argument('--net', dest='net',
                        help='vgg16, res50, res101, res152',
                        default='res101', type=str)
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--load_dir', dest='load_dir',
                        help='directory to load models', default="models",
                        type=str)
    parser.add_argument('--cuda', dest='cuda',
                        help='whether use CUDA',
                        action='store_true')
    parser.add_argument('--ls', dest='large_scale',
                        help='whether use large imag scale',
                        action='store_true')
    parser.add_argument('--mGPUs', dest='mGPUs',
                        help='whether use multiple GPUs',
                        action='store_true')
    parser.add_argument('--cag', dest='class_agnostic',
                        help='whether perform class_agnostic bbox regression',
                        action='store_true')
    parser.add_argument('--parallel_type', dest='parallel_type',
                        help='which part of model to parallel, 0: all, 1: model before roi pooling',
                        default=0, type=int)
    parser.add_argument('--checksession', dest='checksession',
                        help='checksession to load model',
                        default=1, type=int)
    parser.add_argument('--checkepoch', dest='checkepoch',
                        help='checkepoch to load network',
                        default=100, type=int)
    parser.add_argument('--checkpoint', dest='checkpoint',
                        help='checkpoint to load network',
                        default=1036, type=int)
    parser.add_argument('--bs', dest='batch_size',
                        help='batch_size',
                        default=1, type=int)
    parser.add_argument('--vis', dest='vis',
                        help='visualization mode',
                        action='store_true')
    parser.add_argument('--load_name',
                        default="./models/detect.pth",
                        help='load checkpoint',
                        )
    args = parser.parse_args()
    return args
    # lr = cfg.TRAIN.LEARNING_RATE
    # momentum = cfg.TRAIN.MOMENTUM
    # weight_decay = cfg.TRAIN.WEIGHT_DECAY
if __name__ == '__main__':
    # Initialize ecci sdk and connect to the broker in edge-cloud
    ecci_client = Client()
    mqtt_thread = threading.Thread(target=ecci_client.initialize)
    mqtt_thread.start()
    ecci_client.wait_for_ready()
    print('cloud start --------')

    args = parse_args()
    # if args.arch == 'rcnn':
    #     from model.faster_rcnn.vgg16 import vgg16
    #     from model.faster_rcnn.resnet import resnet
    # elif args.arch == 'rfcn':
    #     from model.rfcn.resnet_atrous import resnet
    # elif args.arch == 'couplenet':
    #     from model.couplenet.resnet_atrous import resnet

    print('Called with args:')
    print(args)
    count = 0

    if torch.cuda.is_available() and not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    while True:
        edge_data = ecci_client.get_sub_data_payload_queue().get()
        frame1 = edge_data["frame1"]
        payload = {"type": "data", "contents": {"frame1": "11"}}
        ecci_client.publish(payload, "edge")
        # msg = pickle.loads(frame1)
        # print(msg.shape)
        img = base64.b64decode(frame1)
        npimg = np.frombuffer(img, dtype=np.uint8)
        source = cv2.imdecode(npimg, 1)
        count = count+1
        print("count",count)
        cv2.imshow("stream", source)
        cv2.waitKey(1)
