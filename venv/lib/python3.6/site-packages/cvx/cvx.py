#Copyright 2016 Erik Perillo <erik.perillo@gmail.com>
#
#This file is part of cvx.
#
#This is free software: you can redistribute it and/or modify
#it under the terms of the GNU General Public License as published by
#the Free Software Foundation, either version 3 of the License, or
#(at your option) any later version.
#
#This is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#GNU General Public License for more details.
#
#You should have received a copy of the GNU General Public License
#along with this. If not, see <http://www.gnu.org/licenses/>. 

"""
CVX - openCV eXtension
This module provides some constantly used CV tasks using OpenCV/numpy.

by Erik Perillo <erik.perillo@gmail.com>
"""

import numpy as np
import cv2

class InvalidDimensions(Exception):
    """
    Exception for unexpected/invalid dimensions.
    """
    pass

class InvalidDataType(Exception):
    """
    Exception for invalid data types.
    """
    pass

def inv(img):
    """
    Inverts image.
    """
    return img.max() - img

def mk_divisable(img, div):
    """
    Reshapes image so as to be divisible by div.
    """
    y_clip = img.shape[0] % div
    x_clip = img.shape[1] % div

    return img[y_clip:, x_clip:]

def pyr_prepare(img, pyr_lvl):
    """
    Reshapes image so as to be down/up sampled and keep original shape.
    """
    return mk_divisable(img, 2**pyr_lvl)

def rep_pyrUp(img, n):
    """
    Performs pyrUp repeated times.
    """
    for __ in range(n):
        img = cv2.pyrUp(img)

    return img

def fill(img, width, height, const=0):
    """
    Fills image with const value in order to get specified dimensions.
    """
    h_diff = height - img.shape[0]
    if h_diff > 0:
        filling = const*np.ones(dtype=img.dtype, shape=(h_diff, img.shape[1]))
        img = np.vstack((img, filling))

    w_diff = width - img.shape[1]
    if w_diff > 0:
        filling = const*np.ones(dtype=img.dtype, shape=(img.shape[0], w_diff))
        img = np.hstack((img, filling))

    return img

def h_append(img1, img2, put_line=False):
    """
    Appends two images horizontally, filling them with zeros to match 
    dimensions if needed.
    """
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    if h2 > h1:
        img1 = fill(img1, w1, h2)
    else:
        img2 = fill(img2, w2, h1)

    if put_line:
        line = img1.max()*np.ones(dtype=img1.dtype, shape=(img1.shape[0], 1))
        return np.hstack((img1, line, img2))
    return np.hstack((img1, img2))

def v_append(img1, img2, put_line=False):
    """
    Appends two images vertically, filling them with zeros to match 
    dimensions if needed.
    """
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    if w2 > w1:
        img1 = fill(img1, w2, h1)
    else:
        img2 = fill(img2, w1, h2)

    if put_line:
        line = img1.max()*np.ones(dtype=img1.dtype, shape=(1, img1.shape[1]))
        return np.vstack((img1, line, img2))
    return np.vstack((img1, img2))

def append(img1, img2, hor=True):
    """
    Appends two images either horizontally or vertically.
    """
    return h_append(img1, img2) if hor else v_append(img1, img2)

def str_dim(img):
    """
    Returns a string with image dimensions (height x width x depth).
    """
    return " x ".join(str(dim) for dim in img.shape)

def str_type(img):
    """
    Returns a string with image type.
    """
    return str(img.dtype)

def str_info(img):
    """
    Returns a string with info for image.
    """
    return "dims: %s | type: %s" % (str_dim(img), str_type(img))

def scale(img, new_min=0.0, new_max=255.0):
    """
    Scales pixels of input image to new interval.
    """
    minn = img.min()
    maxx = img.max()
    sigma = maxx - minn

    if sigma == 0.0:
        return img

    out_img = new_min + ((img - minn)/sigma)*(new_max - new_min)

    return out_img

def display(img, title, to_uint8=True):
    """
    Displays image in viewable format with useful info.
    """
    if to_uint8:
        img = np.array(scale(img, 0.0, 255.0), dtype=np.uint8)
    cv2.imshow("'%s' (%s)" % (title, str_info(img)), img)

def save(img, name, to_uint8=True):
    """
    Saves image in viewable format with useful info.
    """
    if to_uint8:
        img = np.array(scale(img, 0.0, 255.0), dtype=np.uint8)
    cv2.imwrite(name, img)

def resize(img, max_w, max_h, scale=0.75):
    """
    Resizes image to fit dimensions.
    """
    h, w = img.shape[:2]
    while h > max_h or w > max_w:
        h, w = img.shape[:2]
        img = cv2.resize(img, tuple(map(int, (scale*w, scale*h))))

    return img

def main():
    print(__doc__)

if __name__ == "__main__":
    main()
