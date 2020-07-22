import cv2
import numpy as np
import matplotlib.pyplot as plt

# define the function of generating gaussian kernel
def gaussian_kernel(kernel_size, sigma):
    kernel = np.zeros((kernel_size, kernel_size), dtype=float)
    kernel_center = kernel_size // 2
    for i in range(kernel_size):
        for j in range(kernel_size):
            dis = (i - kernel_center)**2 + (j - kernel_center)**2
            kernel[i, j] = np.exp(-dis/(2*sigma**2))
    kernel /= 2 * np.pi * sigma**2
    kernel /= kernel.sum()
    return kernel

# define the function of conv2d
def conv2d(input, kernel, padding='valid'):
    img_h, img_w, channel = input.shape
    border = int(kernel.shape[0] / 2)
    if padding == 'valid':
        img_cov = np.zeros([img_h - 2*border, img_w - 2*border, channel], dtype=float)
        for c in range(channel):
            for h in range(border, img_h - border):
                for w in range(border, img_w - border):
                    img_cov[h-border, w-border, c] = np.sum(kernel * input[h-border:h+border+1, w-border:w+border+1, c])
        img_cov = np.clip(img_cov, 0, 255)
        img_cov = img_cov.astype(np.uint8)

    if padding == 'same':
        img_padding = np.zeros([img_h + 2*border, img_w + 2*border, channel], dtype=int)
        img_padding[border:img_h+border, border:img_w+border, :] = input
        img_cov = np.zeros([img_h, img_w, channel], dtype=float)
        for c in range(channel):
            for h in range(border, img_h + border):
                for w in range(border, img_w + border):
                    img_cov[h-border, w-border, c] = np.sum(kernel * img_padding[h-border:h+border+1, w-border:w+border+1, c])
        img_cov = np.clip(img_cov, 0, 255)
        img_cov = img_cov.astype(np.uint8)

    return img_cov

# read input image
img = cv2.imread('C:/Users/ZY03/Desktop/ipynb/cat.jpg')

# gaussian smooth
kernel_size = 5
gaussian_sigma = 2
kernel = gaussian_kernel(kernel_size, gaussian_sigma)
img_cov = conv2d(img, kernel, padding='same')
print(img_cov.shape)

# visualization
fig = plt.figure()
ax = fig.add_subplot(121)
ax.imshow(img)
ax = fig.add_subplot(122)
ax.imshow(img_cov)
plt.show()
