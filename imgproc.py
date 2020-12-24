"""  
Copyright (c) 2019-present NAVER Corp.
MIT License
"""

# -*- coding: utf-8 -*-
import sys
import math
import numpy as np
import pytesseract as ocr
import cv2
import matplotlib.pyplot as plt
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


def getAngle(angle):
    '''arg: angle of the original box. Return the angle to make it horizontal'''
    
    if angle < -45 and angle >= -90: newAngle = 90 - abs(angle)
    else: newAngle = 0 - abs(angle)

    return newAngle


def getCentre(pts):
    '''arg: points of rectangle. Get its centre as tuple'''

    centre_w = math.floor((max(pts[:, :, 0])[0] + min(pts[:, :, 0])[0]) / 2)
    centre_h = math.floor((max(pts[:, :, 1])[0] + min(pts[:, :, 1])[0]) / 2)
    return (centre_w, centre_h)


def cropRegion(img, pts, rang=0):
    """crop image

    Args:
    - img: the image to crop
    - pts: points of detection area (shape is mandatory)
    - rang: default=0. Extra space around detection box. Subtract to shrink

    https://stackoverflow.com/a/48301735/7347566
    """

    assert pts.shape[1] == 1 and pts.shape[2] == 2, "shape must be (-1, 1, 2)"

    # create and draw mask
    mask = np.zeros(img.shape[:-1], np.uint8)
    cv2.fillConvexPoly(mask, pts, 255)

    # get tuple for angle ((min_x, min_y), (max_x, max_y), angle)
    tuple_rect = cv2.minAreaRect(pts)
    angleDEG = getAngle(tuple_rect[-1])
    cog = getCentre(pts)

    # create rotation mat
    transformation = cv2.getRotationMatrix2D(cog, angleDEG, 1)

    # rotated mask holds the object position after rotation
    rotatedMask = cv2.warpAffine(mask, transformation, (img.shape[1], img.shape[0]))
    rotatedInput = cv2.warpAffine(img, transformation, (img.shape[1], img.shape[0]))

    # get non-zero values for mask and use them on the final image
    h, w = np.nonzero(rotatedMask)
    dst = cv2.bitwise_and(rotatedInput, rotatedInput, mask=rotatedMask)
    dst = dst[h[0]:h[-1], w[0]:w[-1], :]

    return dst


def reconTxt(img, reader, exclusion, baw, ktype=None, ksize=None, iterat=1):
    """recognise text from image

    Args:
    - img: image to analyse
    - exclusion: word to exclude from detection
    - baw: bool: true if images are already in a black & white representation
    - ktype: string: "median" or "morpho". None if not willing to use
    - ksize: kernel size for morphological operations. Suggested (2, 1) for morpho; 15 for median
    - iter: kernel iteration (default = 1)
    """

    # check any problems with the image
    try:
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    except cv2.error as err:
        raise Exception from err
        sys.exit(1)

    img_gray, img_bin = cv2.threshold(img_gray, 200, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    if baw: img_gray = img_bin.copy()
    else: img_gray = cv2.bitwise_not(img_bin)

    img_gray = upscale(img_gray, 3)
    
    if ktype == "morpho":
        assert ksize is not None, "ksize cannot be None. Suggested: (5, 5)"

        # define a filter kernel = erosion (enlarge blacks)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        img_fin = cv2.erode(img_gray, kernel, iterat)
    
    elif ktype == "median":
        assert ksize is not None, "ksize cannot be None. Suggested: 15"
        img_fin = cv2.medianBlur(img_gray, ksize)
    
    else:
        img_fin = img_gray.copy()

    # detect word
    #txt = ocr.image_to_string(img_fin)
    text = reader.recognize(img_fin)

    return img_fin, text


def enlarge(img):
    """enlarge

    use matplotlib to save a bigger version of the image. Instead of upscaling,
    the pixel width is enlarged according to matplotlib.pyplot and the output
    is saved to a numpy array

    Args:
    - img: numpy image

    Output:
    - numpy image

    https://stackoverflow.com/a/43363727/7347566
    https://stackoverflow.com/a/7821917/7347566
    """

    fig = plt.figure()

    plt.imshow(img, cmap="gray")
    plt.axis("off")
    
    # draw canvas
    fig.canvas.draw()

    # Now we can save it to a numpy array.
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    plt.clf()
    plt.close()

    return data


def upscale(img, factor):
    """upscale with opencv Gaussian Pyramid

    Args:
    - img: numpy image
    - factor: upscale factor, number of times the upscaling is performed

    Output:
    - numpy image
    """
    
    rows, cols = map(int, img.shape)

    ## UPSCALE LOOP
    for i in range(factor):
        img_hr = cv2.pyrUp(img, dstsize=(2*cols, 2*rows))
        
        # update values
        img = img_hr
        rows, cols = map(int, img.shape)

    return img