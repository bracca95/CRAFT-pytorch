"""  
Copyright (c) 2019-present NAVER Corp.
MIT License
"""

# -*- coding: utf-8 -*-
import sys
import math
import os
import time
import argparse

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from PIL import Image

import cv2
from skimage import io
import numpy as np
import craft_utils
import imgproc
import file_utils
import json
import zipfile

from craft import CRAFT

from collections import OrderedDict

def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict

def str2bool(v):
    '''return True if the input work is in tuple. False otherwise'''
    return v.lower() in ("yes", "y", "true", "t", "1")

parser = argparse.ArgumentParser(description='CRAFT Text Detection')
parser.add_argument('--debug', default=False, type=str2bool, help='debug mode')
parser.add_argument('--divisor', default=3, type=int, help='sub images (div * div')
parser.add_argument('--exclusion', default="", type=str, help='denied words. Separate by comma')
parser.add_argument('--baw', default=True, type=str2bool, help='already black numbers on white bckgrnd')
parser.add_argument('--trained_model', default='weights/craft_mlt_25k.pth', type=str, help='pretrained model')
parser.add_argument('--text_threshold', default=0.7, type=float, help='text confidence threshold')
parser.add_argument('--low_text', default=0.4, type=float, help='text low-bound score')
parser.add_argument('--link_threshold', default=0.4, type=float, help='link confidence threshold')
parser.add_argument('--cuda', default=True, type=str2bool, help='Use cuda for inference')
parser.add_argument('--canvas_size', default=1280, type=int, help='image size for inference')
parser.add_argument('--mag_ratio', default=1.5, type=float, help='image magnification ratio')
parser.add_argument('--poly', default=False, action='store_true', help='enable polygon type')
parser.add_argument('--show_time', default=False, action='store_true', help='show processing time')
parser.add_argument('--test_folder', default='/data/', type=str, help='folder path to input images')
parser.add_argument('--refine', default=False, action='store_true', help='enable link refiner')
parser.add_argument('--refiner_model', default='weights/craft_refiner_CTW1500.pth', type=str, help='pretrained refiner model')

args = parser.parse_args()


""" For test images in a folder """
image_list, _, _ = file_utils.get_files(args.test_folder)

result_folder = './result/'
if not os.path.isdir(result_folder):
    os.mkdir(result_folder)

excl = [w.strip() for w in args.exclusion.split(',')]

def test_net(net, image, text_threshold, link_threshold, low_text, cuda, poly, refine_net=None):
    t0 = time.time()

    # resize
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, args.canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=args.mag_ratio)
    ratio_h = ratio_w = 1 / target_ratio

    # preprocessing
    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
    x = Variable(x.unsqueeze(0))                # [c, h, w] to [b, c, h, w]
    if cuda:
        x = x.cuda()

    # forward pass
    with torch.no_grad():
        y, feature = net(x)

    # make score and link map
    score_text = y[0,:,:,0].cpu().data.numpy()
    score_link = y[0,:,:,1].cpu().data.numpy()

    # refine link
    if refine_net is not None:
        with torch.no_grad():
            y_refiner = refine_net(y, feature)
        score_link = y_refiner[0,:,:,0].cpu().data.numpy()

    t0 = time.time() - t0
    t1 = time.time()

    # Post-processing
    boxes, polys = craft_utils.getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly)

    # coordinate adjustment
    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
    for k in range(len(polys)):
        if polys[k] is None: polys[k] = boxes[k]

    t1 = time.time() - t1

    # render results (optional)
    render_img = score_text.copy()
    render_img = np.hstack((render_img, score_link))
    ret_score_text = imgproc.cvt2HeatmapImg(render_img)

    if args.show_time : print("\ninfer/postproc time : {:.3f}/{:.3f}".format(t0, t1))

    return boxes, polys, ret_score_text



if __name__ == '__main__':
    # load net
    net = CRAFT()     # initialize

    print('Loading weights from checkpoint (' + args.trained_model + ')')
    if args.cuda:
        net.load_state_dict(copyStateDict(torch.load(args.trained_model)))
    else:
        net.load_state_dict(copyStateDict(torch.load(args.trained_model, map_location='cpu')))

    if args.cuda:
        net = net.cuda()
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = False

    net.eval()

    # LinkRefiner
    refine_net = None
    if args.refine:
        from refinenet import RefineNet
        refine_net = RefineNet()
        print('Loading weights of refiner from checkpoint (' + args.refiner_model + ')')
        if args.cuda:
            refine_net.load_state_dict(copyStateDict(torch.load(args.refiner_model)))
            refine_net = refine_net.cuda()
            refine_net = torch.nn.DataParallel(refine_net)
        else:
            refine_net.load_state_dict(copyStateDict(torch.load(args.refiner_model, map_location='cpu')))

        refine_net.eval()
        args.poly = True

    t = time.time()

    # load data
    for k, image_path in enumerate(image_list):
        print("Test image {:d}/{:d}: {:s}".format(k+1, len(image_list), image_path), end='\r')
        image = imgproc.loadImage(image_path)
        
        # prepare result folder
        filename, ext = os.path.splitext(os.path.basename(image_path))
        res_img_fold = os.path.join(result_folder, filename)
        if not os.path.exists(res_img_fold): os.makedirs(res_img_fold)

        part_h = math.floor(image.shape[0] / args.divisor)
        part_w = math.floor(image.shape[1] / args.divisor)

        # divide the whole image into sub-images
        for r in range(args.divisor):
            for c in range(args.divisor):

                part = image[r*part_h+1:(r+1)*part_h, c*part_w+1:(c+1)*part_w, :]
                bboxes, polys, score_text = test_net(net, part, args.text_threshold, args.link_threshold, args.low_text, args.cuda, args.poly, refine_net)
                
                # draw boxes around detected characters
                for i, box in enumerate(polys):

                    # pts is a collection of array [width, height], the opposite of row, col
                    pts = np.int32(box.reshape((-1, 1, 2)))
                    
                    # recognise text and save box region
                    region = imgproc.cropRegion(part, pts)

                    try:
                        region, text = imgproc.reconTxt(region, excl, args.baw, ktype="median", ksize=15)

                        # check if numbers are black
                        if args.debug:
                            cv2.imwrite(os.path.join(res_img_fold, 
                                f"{filename}_box_{r}{c}{i}{ext}"), region)
                        else:
                            if bool(text.strip()):
                                cv2.imwrite(
                                    os.path.join(res_img_fold, f"{filename}_box_{r}{c}{i}{ext}"), 
                                    region)
                                with open(os.path.join(res_img_fold, f"{filename}_box_{r}{c}{i}.txt"), "w") as f:
                                    f.write(text)
                    except Exception:
                        print("error in image:", image_path)
                        
                    # draw boxes on full image
                    pts = np.add([c*part_w, r*part_h], pts)
                    cv2.polylines(image, [pts], True, color=(0, 0, 255), thickness=2)
        
        # save images
        cv2.imwrite(os.path.join(res_img_fold, f"{filename}{ext}"), image[:, :, ::-1])

    print("elapsed time : {}s".format(time.time() - t))
