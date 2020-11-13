import cv2
from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm


def add_noise(image, noise_type):
    '''
    Function to add Gassiaun or Salt n Pepper noise to a given image.
    image - image in which noise needs to be added
    noise_type - gaussian or sp(salt n pepper)
    '''

    if noise_type == "gaussian":
        row, col = image.shape
        image *= 255
        mean = 0
        var = 0.1
        sigma = var**0.5
        gauss = np.random.normal(mean, sigma, (row, col))
        gauss = gauss.reshape(row, col)
        gauss /= np.max(gauss)
        noisy = image + 5*gauss
        noisy = noisy/255
        return noisy

    elif noise_type == "sp":
        row, col = image.shape
        s_vs_p = 0.5
        amount = 0.01
        out = np.copy(image)

        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                  for i in image.shape]
        out[tuple(coords)] = 1

        # Pepper mode
        num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                  for i in image.shape]
        out[tuple(coords)] = 0
        return out


def create_noise():
    DATA_DIR = "Data"
    GAUSSIAN_DIR = "gaussian"
    SNP_DIR = "sp_noise"
    GT_DIR = "gt"

    if not os.path.exists(GAUSSIAN_DIR):
        os.mkdir(GAUSSIAN_DIR)
    if not os.path.exists(SNP_DIR):
        os.mkdir(SNP_DIR)
    if not os.path.exists(GT_DIR):
        os.mkdir(GT_DIR)

    for _, _, files in os.walk(DATA_DIR):
        for file in tqdm(files):
            img = np.array(ImageOps.grayscale(
                Image.open(os.path.join(DATA_DIR, file))))
            img = img/255
            gauss = add_noise(img, "gaussian")
            snp = add_noise(img, "sp")

            plt.imsave(os.path.join(GT_DIR, file), img, cmap="gray")
            plt.imsave(os.path.join(GAUSSIAN_DIR, file), gauss, cmap="gray")
            plt.imsave(os.path.join(SNP_DIR, file), snp, cmap="gray")


if __name__ == "__main__":
    create_noise()
