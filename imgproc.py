"""  
Copyright (c) 2019-present NAVER Corp.
MIT License
"""

# -*- coding: utf-8 -*-
import sys
import numpy as np
import pytesseract as ocr
import cv2
from skimage import io

def loadImage(img_file):
    img = io.imread(img_file)           # RGB order
    if img.shape[0] == 2: img = img[0]
    if len(img.shape) == 2 : img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if img.shape[2] == 4:   img = img[:,:,:3]
    img = np.array(img)

    return img

def normalizeMeanVariance(in_img, mean=(0.485, 0.456, 0.406), variance=(0.229, 0.224, 0.225)):
    # should be RGB order
    img = in_img.copy().astype(np.float32)

    img -= np.array([mean[0] * 255.0, mean[1] * 255.0, mean[2] * 255.0], dtype=np.float32)
    img /= np.array([variance[0] * 255.0, variance[1] * 255.0, variance[2] * 255.0], dtype=np.float32)
    return img

def denormalizeMeanVariance(in_img, mean=(0.485, 0.456, 0.406), variance=(0.229, 0.224, 0.225)):
    # should be RGB order
    img = in_img.copy()
    img *= variance
    img += mean
    img *= 255.0
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img

def resize_aspect_ratio(img, square_size, interpolation, mag_ratio=1):
    height, width, channel = img.shape

    # magnify image size
    target_size = mag_ratio * max(height, width)

    # set original image size
    if target_size > square_size:
        target_size = square_size
    
    ratio = target_size / max(height, width)    

    target_h, target_w = int(height * ratio), int(width * ratio)
    proc = cv2.resize(img, (target_w, target_h), interpolation = interpolation)


    # make canvas and paste image
    target_h32, target_w32 = target_h, target_w
    if target_h % 32 != 0:
        target_h32 = target_h + (32 - target_h % 32)
    if target_w % 32 != 0:
        target_w32 = target_w + (32 - target_w % 32)
    resized = np.zeros((target_h32, target_w32, channel), dtype=np.float32)
    resized[0:target_h, 0:target_w, :] = proc
    target_h, target_w = target_h32, target_w32

    size_heatmap = (int(target_w/2), int(target_h/2))

    return resized, ratio, size_heatmap

def cvt2HeatmapImg(img):
    img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
    return img


def cropRegion(img, pts, rang=0):
    """crop image

    Args:
    - img: the image to crop
    - pts: points of detection area (shape is mandatory)
    - rang: default=0. Extra space around detection box. Subtract to shrink

    https://stackoverflow.com/a/48301735/7347566
    """

    assert pts.shape[1] == 1 and pts.shape[2] == 2, "shape must be (-1, 1, 2)"

    # Crop a bounding rect around the shape
    rect = cv2.boundingRect(pts)
    x,y,w,h = rect
    cropped = img[y-rang:y+h+rang, x-rang:x+w+rang].copy()

    # make mask of the polygon
    pts = pts - pts.min(axis=0)
    mask = np.zeros(cropped.shape[:2], np.uint8)
    cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)

    # do bit-op
    dst = cv2.bitwise_and(cropped, cropped, mask=mask)

    # add the white background
    bg = np.ones_like(cropped, np.uint8)*255
    cv2.bitwise_not(bg,bg, mask=mask)
    dst2 = bg + dst

    return dst2


def reconTxt(img, exclusion, baw, ktype=None, ksize=None, iterat=1):
    """recognise text from image

    Args:
    - img: image to analyse
    - exclusion: word to exclude from detection
    - baw: bool: true if images are already in a black & white representation
    - ktype: string: "median" or "morpho". None if not willing to use
    - kezie: kernel size for morphological operations. Suggested (2, 1) for morpho; 15 for median
    - iter: kernel iteration (default = 1)
    """

    # check any problems with the image
    try:
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    except cv2.error as err:
        raise Exception from err
        sys.exit(1)

    img_gray, img_bin = cv2.threshold(img_gray, 200, 255, cv2.THRESH_BINARY)

    if baw: img_gray = img_bin.copy()
    else: img_gray = cv2.bitwise_not(img_bin)
    
    if ktype == "morpho":
        assert ksize is not None, "ksize cannot be None. Suggested: (2, 1)"
        
        # define a filter kernel
        kernel = np.ones(kernel, np.uint8)

        # open filter: erosion + dilation
        img_fin = cv2.erode(img_gray, kernel, iterat)
        img_fin = cv2.dilate(img_fin, kernel, iterat)
    
    elif ktype == "median":
        assert ksize is not None, "ksize cannot be None. Suggested: 15"
        img_fin = cv2.medianBlur(img_gray, ksize)
    
    else:
        img_fin = img_gray.copy()


    # detect work
    txt = ocr.image_to_string(img_fin)

    return img_fin, txt