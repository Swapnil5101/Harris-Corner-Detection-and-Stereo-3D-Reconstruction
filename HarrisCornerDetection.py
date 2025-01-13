import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def conv2D(image, kernel):
    """
    Implements 2D convolution of an image with a kernel.
    """
    image_padded = np.pad(image, ((1, 1), (1, 1)), mode='constant')
    output = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            output[i, j] = np.sum(image_padded[i:i+3, j:j+3] * kernel)
    return output

def harris_corner_detection(image, k=0.04):
    """
    Implements Harris corner detection.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)

    # Calculate x and y derivatives
    sobelx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobely = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    Ix = conv2D(gray, sobelx)
    Iy = conv2D(gray, sobely)

    # Calculate Gaussian window
    gaussian = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16.0

    # Calculate Harris matrix
    Ix2 = Ix**2
    Iy2 = Iy**2
    Ixy = Ix * Iy
    Sx2 = conv2D(Ix2, gaussian)
    Sy2 = conv2D(Iy2, gaussian)
    Sxy = conv2D(Ixy, gaussian)

    # Calculate Harris response
    det = Sx2 * Sy2 - Sxy**2
    trace = Sx2 + Sy2
    R = det - k * trace**2

    # Threshold and find corners
    corners = np.zeros_like(R, dtype=np.uint8)
    corners[(R > 0.01 * R.max())] = 255

    return corners

# Folder path containing images
folder_path = '/content/drive/MyDrive/Question 1'

# Loop through images in the folder
for filename in os.listdir(folder_path):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        image_path = os.path.join(folder_path, filename)
        image = cv2.imread(image_path)

        # Detect corners using custom implementation
        custom_corners = harris_corner_detection(image)

        # Detect corners using OpenCV implementation
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        opencv_corners = cv2.cornerHarris(gray, 2, 3, 0.04)

        # Plot images with corners
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        ax[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        ax[0].scatter(np.argwhere(custom_corners == 255)[:, 1], np.argwhere(custom_corners == 255)[:, 0], s=2, c='r')
        ax[0].set_title('Custom Implementation')

        ax[1].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        ax[1].scatter(np.argwhere(opencv_corners > 0.01 * opencv_corners.max())[:, 1], np.argwhere(opencv_corners > 0.01 * opencv_corners.max())[:, 0], s=2, c='r')
        ax[1].set_title('OpenCV Implementation')

        plt.show()