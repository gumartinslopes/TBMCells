import matplotlib.pyplot as plt
import os
import cv2
import numpy as np

def create_folder(folder_path):
    '''Creates a folder if does not exists.'''
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

def visualize(figsize=(15, 15),**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=figsize)
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()

def overlay_visualize(figsize=(15, 15),**images):
    """PLot images in one row with a legend"""
    n = len(images)
    plt.figure(figsize=figsize)
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
        legendas = [
        plt.Line2D([0], [0], color='#ff00ff', lw=2),
        plt.Line2D([0], [0], color='black', lw=2),
        plt.Line2D([0], [0], color='#00ff00', lw=2),
        plt.Line2D([0], [0], color='white', lw=2),
    ]

    # Nomear as legendas
    nomes_legendas = ['True Positive','True Negative', 'False Positive', 'False Negative']

    # Adicionar as legendas ao plot
    plt.legend(legendas, nomes_legendas)

    plt.show()

def change_color(img, color):
    img = cv2.cvtColor(np.float32(img), cv2.COLOR_GRAY2RGB)
    ''' Transforms all the white color in green'''
    # Definir as cores que você quer substituir (branco)
    low = np.array([200, 200, 200])
    high = np.array([255, 255, 255])
    mask = cv2.inRange(np.array(img), low, high)
    painted_img = np.array(img).copy()
    # Substituir as cores brancas pela cor verde
    painted_img[mask != 0] = color
    return painted_img

def overlay_comparison(pred, gt):
    """ Creates an overlapped visualization
    bewtween the ground truth and the obtained prediction"""
    green = np.array([0, 255, 0])
    pred_colored = np.int32(change_color(pred, green))
    pred_arr = np.asarray(pred_colored)
    gt_arr = np.asarray(gt)
    binary_xor = cv2.bitwise_xor(pred_arr, gt_arr)
    return binary_xor

def apply_mask(image, mask):
    """Given an image and it's mask, applies the and operation."""
    return cv2.bitwise_and(np.uint8(image),np.uint8(image), mask = np.uint8(mask))

