import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import cv2


def show_im(img):
    plt.imshow(img, cmap="gray")
    plt.show()


def MSE(Y, YH):
    Y = Y.astype(float)
    YH = YH.astype(float)
    return np.square(Y - YH).mean()


def PSNR(original, noisy, peak=100):
    mse = np.mean((original-noisy)**2)
    return 10*np.log10(peak*peak/mse)


# To get a window where center is (x,y) that is of size (N,N)


def get_window(img, x, y, N=25):
    """
    Extracts a small window of input image, around the center (x,y)
    img - input image
    x,y - cordinates of center
    N - size of window (N,N) {should be odd}
    """

    h, w, c = img.shape             # Extracting Image Dimensions

    arm = N//2                      # Arm from center to get window
    window = np.zeros((N, N, c))
    # print((0, x-arm))
    xmin = max(0, x-arm)
    xmax = min(w, x+arm+1)
    ymin = max(0, y-arm)
    ymax = min(h, y+arm+1)

    window[arm - (y-ymin):arm + (ymax-y), arm - (x-xmin)
                  :arm + (xmax-x)] = img[ymin:ymax, xmin:xmax]

    return window

# Provided by Matlab


def matlab_style_gauss2d(shape=(3, 3), sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m, n = [(ss-1.)/2. for ss in shape]
    y, x = np.ogrid[-m:m+1, -n:n+1]
    h = np.exp(-(x*x + y*y)/(2.*sigma*sigma))
    h[h < np.finfo(h.dtype).eps*h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


name = "Image3.png"

img_path = "sp_noise\{}".format(name)
gt_path = "gt\{}".format(name)

img = np.array(ImageOps.grayscale(Image.open(img_path)), dtype=float)
gt = np.array(ImageOps.grayscale(Image.open(gt_path)))

def NL_means(img,h=9,f=4,t=11):
    # neighbourhood size 2f+1
    N = 2*f + 1

    # sliding window size 2t+1
    S = 2*t + 1

    # Filtering Parameter
    sigma_h = h

    pad_img = np.pad(img, t+f)

    h, w = img.shape
    h_pad, w_pad = pad_img.shape

    neigh_mat = np.zeros((h+S-1, w+S-1, N, N))


    for y in range(h+S-1):
        for x in range(w+S-1):
            neigh_mat[y, x] = np.squeeze(get_window(
                pad_img[:, :, np.newaxis], x+f, y+f, 2*f+1))

    output = np.zeros(img.shape)


    prog = tqdm(total=(h-1)*(w-1), position=0, leave=True)

    for Y in range(h):
        for X in range(w):
            x = X + t
            y = Y + t
            a = get_window(np.reshape(
                neigh_mat, (h+S-1, w+S-1, N*N)), x, y, S)
            b = neigh_mat[y, x].flatten()
            c = a-b
            d = c*c
            e = np.sqrt(np.sum(d, axis=2))
            F = np.exp(-e/(sigma_h*sigma_h))
            Z = np.sum(F)
            im_part = np.squeeze(get_window(pad_img[:, :, None], x+f, y+f, S))
            NL = np.sum(F*im_part)
            output[Y, X] = NL/Z
            prog.update(1)
    return output
