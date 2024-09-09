import logging
import os
from types import SimpleNamespace

import cv2
import numpy as np
import scipy

import config

cfg = SimpleNamespace(**vars(config))


def calculate_ncc(image1, image2):
    """Calculates the ncc of two given images"""
    if image1.size == 0 or image2.size == 0:
        return float("-inf")

    if image1.shape != image2.shape:
        raise ValueError("Images must have the same dimensions for NCC calculation.")

    mean1 = np.mean(image1)
    mean2 = np.mean(image2)

    zero_mean1 = image1 - mean1
    zero_mean2 = image2 - mean2

    numerator = np.sum(zero_mean1 * zero_mean2)
    denominator = np.sqrt(np.sum(zero_mean1**2) * np.sum(zero_mean2**2))

    if denominator == 0:
        return float("-inf")

    ncc = numerator / denominator
    return ncc


def load_image():
    """Load images from given folder"""
    base_dir = cfg.base_dir
    image_dir_list = os.listdir(base_dir)
    images = []
    new_dir_list = []
    for image_dir in image_dir_list:
        new_image_dir = image_dir[:-4] + ".jpg"
        image_dir = os.path.join(base_dir, image_dir)

        image = cv2.imread(image_dir, cv2.IMREAD_GRAYSCALE)
        image_dir = image_dir[:-4] + ".jpg"
        images.append(image)
        new_dir_list.append(new_image_dir)
    return images, new_dir_list


def normalize_and_scale(channel):
    """Normalize the channel and ensure it range from 0 -255"""
    min_val = channel.min()
    max_val = channel.max()
    scaled_channel = 255 * (channel - min_val) / (max_val - min_val)
    return scaled_channel


def apply_translation(image, dx=None, dy=None):
    """Apply translation of images"""
    image = scipy.ndimage.shift(image, [dy, dx], mode="nearest")
    return image


def match_image(B, G, R, level):
    """Pyramid match of images"""
    G_dy, G_dx, R_dy, R_dx = 0, 0, 0, 0
    if level < cfg.level:
        new_B = cv2.resize(B, (B.shape[1] // 2, B.shape[0] // 2))
        new_G = cv2.resize(G, (G.shape[1] // 2, G.shape[0] // 2))
        new_R = cv2.resize(R, (R.shape[1] // 2, R.shape[0] // 2))
        new_level = level + 1

        G_dy, G_dx, R_dy, R_dx = match_image(new_B, new_G, new_R, new_level)

        G = apply_translation(G, dx=G_dx * 2, dy=G_dy * 2)
        R = apply_translation(R, dx=R_dx * 2, dy=R_dy * 2)
    if level != 0:
        logging.info("Level:{}".format(level))
        new_G_dy, new_G_dx = find_best_shift(G, B, level)
        new_R_dy, new_R_dx = find_best_shift(R, B, level)
        return (
            G_dy * 2 + new_G_dy,
            G_dx * 2 + new_G_dx,
            R_dy * 2 + new_R_dy,
            R_dx * 2 + new_R_dx,
        )
    else:
        logging.info("Level:{}".format(level))
        if cfg.base_dir == "data_jpg":
            G_dy, G_dx = find_best_shift(G, B, level)
            R_dy, R_dx = find_best_shift(R, B, level)
            G = apply_translation(G, dx=G_dx, dy=G_dy)
            R = apply_translation(R, dx=R_dx, dy=R_dy)
        return B, G, R


def adjust_image_size(image, target):
    target_height, target_width = target.shape
    current_height, current_width = image.shape

    if current_height > target_height:
        image = image[:target_height, :]
        current_height = target_height
    if current_width > target_width:
        image = image[:, :target_width]
        current_width = target_width
    padding_bottom = target_height - current_height
    padding_right = target_width - current_width

    if padding_bottom > 0 or padding_right > 0:
        image = np.pad(image, ((0, padding_bottom), (0, padding_right)), "edge")

    return image


def sobel_scan(image):
    """Sobel operator scan images"""
    pad_image = np.pad(image, 1, "edge")
    pad_image = pad_image.astype(np.int32)
    grad_x = np.zeros((pad_image.shape[0], pad_image.shape[1], 6))
    grad_x[1:, 1:, 0] = -1 * pad_image[:-1, :-1]
    grad_x[:, 1:, 1] = -2 * pad_image[:, :-1]
    grad_x[:-1, 1:, 2] = -1 * pad_image[1:, :-1]
    grad_x[1:, :-1, 3] = 1 * pad_image[:-1, 1:]
    grad_x[:, :-1, 4] = 2 * pad_image[:, 1:]
    grad_x[:-1, :-1, 5] = 1 * pad_image[1:, 1:]

    gradx = np.sum(grad_x, axis=-1)

    grad_y = np.zeros((pad_image.shape[0], pad_image.shape[1], 6))
    grad_y[1:, 1:, 0] = -1 * pad_image[:-1, :-1]
    grad_y[1:, :, 1] = -2 * pad_image[1:, :]
    grad_y[1:, :-1, 2] = -1 * pad_image[:-1, 1:]
    grad_y[:-1, 1:, 3] = 1 * pad_image[1:, :-1]
    grad_y[:-1, :, 4] = 2 * pad_image[:-1, :]
    grad_y[:-1, :-1, 5] = 1 * pad_image[1:, 1:]
    grady = np.sum(grad_y, axis=-1)
    grad = np.sqrt(gradx**2 + grady**2)
    return grad[1:-1, 1:-1]




def find_best_shift(imageA, imageB, level):
    new_A = sobel_scan(imageA)
    new_A = normalize_and_scale(new_A)

    new_B = sobel_scan(imageB)
    new_B = normalize_and_scale(new_B)

    if level == cfg.level:
        marginx = imageB.shape[1]
        marginy = imageB.shape[0]
    else:
        marginx = 1
        marginy = 1
    max_ncc = -1
    best_y, best_x = 0, 0
    for dy in range(-marginy, marginy + 1):
        for dx in range(-marginx, marginx + 1):
            shift_A = apply_translation(new_A, dx=dx, dy=dy)
            ncc = calculate_ncc(
                shift_A,
                new_B,
            )
            if ncc > max_ncc:
                max_ncc = ncc
                best_y, best_x = dy, dx

    return best_y, best_x


def average_channel(image):
    average_rgb = np.mean(image)

    average_r = np.mean(image[:, :, 0])
    average_g = np.mean(image[:, :, 1])
    average_b = np.mean(image[:, :, 2])

    r_coef = average_rgb / average_r
    g_coef = average_rgb / average_g
    b_coef = average_rgb / average_b

    channel_b = image[:, :, 0] * r_coef
    channel_g = image[:, :, 1] * g_coef
    channel_r = image[:, :, 2] * b_coef
    new_image = np.dstack([channel_b, channel_g, channel_r])
    new_image = np.where(new_image > 255, 255, new_image)
    return new_image.astype(np.uint8)


def adaptive_crop(image):
    shape = image.shape
    crop_area = cfg.crop_area
    adjust_width = cfg.adjust_width
    threshold = cfg.cut_threshold
    crop_left = image[:, :crop_area]
    crop_right = image[:, -crop_area:]
    crop_up = image[:crop_area, :]
    crop_down = image[-crop_area:, :]

    scan_left = sobel_scan(crop_left)
    scan_right = sobel_scan(crop_right)
    scan_up = sobel_scan(crop_up)
    scan_down = sobel_scan(crop_down)

    scan_left = np.where(scan_left > 30, 1, 0)
    scan_right = np.where(scan_right > 30, 1, 0)
    scan_up = np.where(scan_up > 30, 1, 0)
    scan_down = np.where(scan_down > 30, 1, 0)

    crop_left, crop_right, crop_up, crop_down = None, None, None, None
    for i in range(0, crop_area // adjust_width, crop_area // adjust_width):
        if crop_left is None:
            left = np.sum(
                scan_left[
                    :,
                    (crop_area // adjust_width - i - 1) * adjust_width : (
                        crop_area // adjust_width - i
                    )
                    * adjust_width,
                ]
            )
            if left > threshold:
                crop_left = crop_area - i * adjust_width
        if crop_right is None:
            right = np.sum(scan_right[:, i * adjust_width : (i + 1) * adjust_width])
            if right > threshold:
                crop_right = shape[1] - (i + 1) * adjust_width
        if crop_up is None:
            up = np.sum(
                scan_up[
                    :,
                    (crop_area // adjust_width - i - 1) * adjust_width : (
                        crop_area // adjust_width - i
                    )
                    * adjust_width,
                ]
            )
            if up > threshold:
                crop_up = crop_area - i * adjust_width
        if crop_down is None:
            down = np.sum(scan_down[i * adjust_width : (i + 1) * adjust_width, :])
            if down > threshold:
                crop_down = shape[0] - (i + 1) * adjust_width

    if crop_left is None:
        crop_left = 0
    if crop_right is None:
        crop_right = shape[1]

    if crop_up is None:
        crop_up = 0
    if crop_down is None:
        crop_down = shape[0]
    return crop_left, crop_right, crop_up, crop_down


def cut_image(B, G, R):
    h, w = B.shape
    cut_h = int(h * cfg.cut_coef)
    cut_w = int(w * cfg.cut_coef)

    B = B[cut_h:-cut_h, cut_w:-cut_w]

    G = G[cut_h:-cut_h, cut_w:-cut_w]
    R = R[cut_h:-cut_h, cut_w:-cut_w]

    return B, G, R
