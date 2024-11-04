from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import base64
import os
import argparse

import cv2
import torch
import numpy as np
import glob
import pickle
import pathlib
from ecci_sdk_edge import Client
import multiprocessing as mp
import threading, random, socket, socketserver, time, pickle
from datasets.vid_dataset import ImagenetDataset
from datasets.data_preprocessing import group_annotation_by_class

parser = argparse.ArgumentParser(description='tracking demo')
parser.add_argument("--dataset", type=str, default="F:/baidupandownload/kubeedge_meeting/dataset/imagenet2015/ILSVRC/",
                    help="The root directory of the VOC dataset or Open Images dataset.")
parser.add_argument('--config', type=str, help='config file', default="./experiments/siamrpn_alex_dwxcorr_multi/config.yaml")
parser.add_argument('--snapshot', type=str, help='model name', default="./experiments/siamrpn_alex_dwxcorr_multi/track.pth")
parser.add_argument("--label_file", type=str, default='./datasets/vid-model-labels.txt', help="The label file path.")
parser.add_argument("--eval_dir", default="./eval/output/", type=str,
                    help="The directory to store evaluation results.")

args = parser.parse_args()
def main():
# Initialize ecci sdk and connect to the broker in edge-cloud
    ecci_client = Client()
    mqtt_thread = threading.Thread(target=ecci_client.initialize)
    mqtt_thread.start()
    ecci_client.wait_for_ready()
    print('edge start --------')

    # load config
    # cfg.merge_from_file(args.config)
    # cfg.CUDA = torch.cuda.is_available() and cfg.CUDA
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = ImagenetDataset(args.dataset, is_val=True)
    #true_case_stat, all_gb_boxes = group_annotation_by_class(dataset)
    #frame = cv2.
    T1 = time.time()
    T2 = time.time()
    for f in range(len(dataset)):
        frame, first_frame = dataset.get_image(f)
        # cv2.imshow("source", frame)
        # cv2.waitKey(0)
        frame1, frame2, frame3 = np.split(frame, 3)
        print("type", type(frame))
        T3 = time.time()
        print("time1111:", T3 - T2)
        #frame_image = frame.array
        encoded, buffer = cv2.imencode('.jpg', frame)
        print("buffer")
        jpg_as_text = base64.b64encode(buffer)
        print("jpg_as_text")


        # send frame to cloud
        payload = {"type": "data", "contents": {"frame1": jpg_as_text}}
        #print("####################", payload)
        ecci_client.publish(payload, "cloud")
        edge_data = ecci_client.get_sub_data_payload_queue().get()
        frame1 = edge_data["frame1"]
        # payload = {"type": "data", "contents": {"frame2": frame2}}
        # #print("####################", payload)
        # ecci_client.publish(payload, "cloud")
        # payload = {"type": "data", "contents": {"frame3": frame3}}
        # #print("####################", payload)
        # ecci_client.publish(payload, "cloud")

        # get rect from cloud
        # cloud_data = ecci_client.get_sub_data_payload_queue().get()
        # print("ss")
        # #print("###########recieve data from cloud", cloud_data)
        # bbox = cloud_data["bbox"]
        # label = cloud_data["label"]
        # probs = cloud_data["probs"]
        T2 = time.time()
        print("time:", T2-T1)
        T1=T2





if __name__ == '__main__':
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass
    main()