import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import cv2

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

def gaussian_smooth(noisy_img,sigma = 2,kernel = 5):
    """
    noisy_img = image that is to be gaussian smoothened
    sigma = sigma for the gaussian kernel
    kernel = size of kernel (5,5) "Should always be odd"
    """
    img = np.array(noisy_img,dtype=float)

    # Padding image for boundaries
    pad_img = np.pad(img, kernel)

    # Making gaussian kernel of given size and sigma
    k_window = matlab_style_gauss2d((kernel,kernel),sigma)

    new_img = np.zeros(img.shape)

    h, w = img.shape
    prog = tqdm(total=h*w, position=0, leave=True)

    # Iterating for each pixel
    for Y in range(h):
        for X in range(w):
            x = X + kernel; y = Y + kernel
            window = np.squeeze(get_window(pad_img[:,:,None],x,y,kernel))
            weights = window*kernel
            pix = np.sum(weights)/np.sum(kernel)
            new_img[Y,X] = pix
            prog.update(1)
    new_img = new_img/new_img.max()
    new_img *= 255
    return new_img

def main():

    file = "Image3.png"

    img_path = os.path.join("sp_noise",file)
    gt_path = os.path.join("gt",file)
    img = np.array(ImageOps.grayscale(Image.open(img_path)), dtype=float)
    gt = np.array(ImageOps.grayscale(Image.open(gt_path))) 

    gaussian_img = gaussian_smooth(img)

    _, axs = plt.subplots(1,3,figsize=(20,20))
    axs[0].imshow(gt,cmap='gray')
    axs[0].title.set_text('Ground Truth')
    axs[1].imshow(img,cmap='gray')
    axs[1].title.set_text('Noisy Image\nMSE={0:.2f}, PSNR={1:.2f}'.format(MSE(gt,img),PSNR(gt,img)))
    axs[2].imshow(gaussian_img,cmap='gray')
    axs[2].title.set_text('Nl Means Denoising\nMSE={0:.2f}, PSNR={1:.2f}'.format(MSE(gt,gaussian_img),PSNR(gt,gaussian_img)))
    plt.show()

if __name__ == "__main__":
    main()