import json

import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.ndimage import map_coordinates


def blend_image(image1, image2, overlap_ratio, offset, lap_layers=2, direction="w"):
    def pad_image_to_target(image, target_size, offset):
        h, w, c = image.shape
        h_bridge = max(target_size[0] - h, 0)
        w_bridge = max(target_size[1] - w, 0)

        h_pad = h_bridge // 2 + offset[0]
        w_pad = w_bridge // 2 + offset[1]

        pad_width = [(h_pad, h_bridge - h_pad), (w_pad, w_bridge - w_pad), (0, 0)]
        image_pad = np.pad(image, pad_width, mode="constant", constant_values=0.0)
        return image_pad

    h1, w1, c = image1.shape
    h2, w2, c = image2.shape

    target_h = max(h1, h2)
    target_w = max(w1, w2)

    image1 = pad_image_to_target(image1, (target_h, target_w), (0, 0))
    image2 = pad_image_to_target(image2, (target_h, target_w), offset)
    if direction == "w":
        mask_w = int(target_w * overlap_ratio)
        blend_w = 2 * target_w - mask_w
        blend_h = target_h
        blend_image = np.zeros((blend_h, blend_w, 3), dtype=np.float32)

        mask = np.zeros_like((blend_image))
        mask[:, : blend_w // 2] = 1.0
        blend_left = np.zeros_like(blend_image)
        blend_right = np.zeros_like(blend_image)

        blend_left[:, :target_w] = image1
        blend_right[:, -target_w:] = image2
        blend_image = image_blend(blend_left, blend_right, lap_layers, mask)
        return blend_image


def create_laplacian_stack(image, level, mask):
    gaussian_stack, mask = create_gaussian_stack(image, level, mask)
    laplacian_stack = []
    for i in range(level):
        laplacian = gaussian_stack[i] - gaussian_stack[i + 1]
        laplacian_stack.append(laplacian)
    return laplacian_stack, mask, gaussian_stack[-1]


def create_gaussian_stack(image, level, mask):
    gaussian_stack = [image]
    mask_stack = [mask]
    for i in range(1, level + 1):
        blured = cv2.GaussianBlur(image, (2 ** (i + 1) + 1, 2 ** (i + 1) + 1), 0)
        blur_mask = cv2.GaussianBlur(mask, (2 ** (i + 1) + 1, 2 ** (i + 1) + 1), 0)
        mask_stack.append(blur_mask)
        gaussian_stack.append(blured)

    return gaussian_stack, mask_stack


def image_blend(image1, image2, level, mask):
    lap_stack1, mask_stack, bottom1 = create_laplacian_stack(image1, level, mask)
    lap_stack2, _, bottom2 = create_laplacian_stack(image2, level, mask)
    blend_image = [bottom1 * mask_stack[-1] + bottom2 * (1 - mask_stack[-1])]
    for i1, i2, mask in zip(lap_stack1, lap_stack2, mask_stack[1:-1]):
        image = i1 * mask + i2 * (1 - mask)
        blend_image.append(image)
    blend_image = np.array(blend_image)
    blend_image = np.sum(blend_image, axis=0)
    blend_image = np.clip(blend_image, 0, 255)
    blend_image = blend_image.astype(np.uint8)
    return blend_image


def warpImage(im, H):
    h, w = im.shape[:2]
    corners = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    ones = np.ones((4, 1))
    corners_hom = np.hstack([corners, ones])  # 4 x 3
    transformed_corners = (H @ corners_hom.T).T  # 4 x 3
    transformed_corners = transformed_corners[:, :2] / transformed_corners[:, 2:]
    min_x = int(np.floor(np.min(transformed_corners[:, 0])))
    max_x = int(np.ceil(np.max(transformed_corners[:, 0])))
    min_y = int(np.floor(np.min(transformed_corners[:, 1])))
    max_y = int(np.ceil(np.max(transformed_corners[:, 1])))
    x_range = np.arange(min_x, max_x + 1)
    y_range = np.arange(min_y, max_y + 1)
    xv, yv = np.meshgrid(x_range, y_range)
    grid_points = np.stack([xv.flatten(), yv.flatten()], axis=-1)  # N x 2
    ones = np.ones((grid_points.shape[0], 1))
    grid_points_hom = np.hstack([grid_points, ones])  # N x 3
    H_inv = np.linalg.inv(H)
    src_points = (H_inv @ grid_points_hom.T).T  # N x 3
    src_points = src_points[:, :2] / src_points[:, 2:]
    src_x = src_points[:, 0].reshape(yv.shape)
    src_y = src_points[:, 1].reshape(yv.shape)
    out_shape = yv.shape
    if im.ndim == 3:
        imwarped = np.zeros((out_shape[0], out_shape[1], im.shape[2]), dtype=im.dtype)
    else:
        imwarped = np.zeros((out_shape[0], out_shape[1], 2), dtype=im.dtype)
    for i in range(im.shape[2]) if im.ndim == 3 else range(1):
        if im.ndim == 3:
            channel = im[:, :, i]
        else:
            channel = im
        coords = [src_y, src_x]
        channel_warped = map_coordinates(
            channel, coords, order=1, mode="constant", cval=0.0
        )
        imwarped[:, :, i] = channel_warped

    return imwarped[:, :, :3]


def normalize_points(pts):
    mean = np.mean(pts, axis=0)
    std = np.std(pts)
    scale = np.sqrt(2) / std
    T = np.array(
        [[scale, 0, -scale * mean[0]], [0, scale, -scale * mean[1]], [0, 0, 1]]
    )
    pts_hom = np.hstack((pts, np.ones((pts.shape[0], 1))))
    pts_norm = (T @ pts_hom.T).T
    return pts_norm[:, :2], T


def compute_homography(im1_pts, im2_pts):
    assert (
        im1_pts.shape[0] == im2_pts.shape[0]
    ), "Number of points in both arrays must be equal"
    assert (
        im1_pts.shape[0] >= 4
    ), "At least 4 points are required to compute the homography"

    # 对点集进行归一化
    im1_pts_norm, T1 = normalize_points(im1_pts)
    im2_pts_norm, T2 = normalize_points(im2_pts)

    num_points = im1_pts.shape[0]
    A = np.zeros((2 * num_points, 9))

    for i in range(num_points):
        x1, y1 = im1_pts_norm[i]
        x2, y2 = im2_pts_norm[i]
        A[2 * i] = [-x1, -y1, -1, 0, 0, 0, x2 * x1, x2 * y1, x2]
        A[2 * i + 1] = [0, 0, 0, -x1, -y1, -1, y2 * x1, y2 * y1, y2]

    _, _, Vt = np.linalg.svd(A)
    h = Vt[-1]
    H_norm = h.reshape((3, 3))
    H = np.linalg.inv(T2) @ H_norm @ T1
    H /= H[2, 2]

    return H


def read_json(path):
    with open(path, "r") as f:
        json_data = json.load(f)
    im_pts1 = np.array(json_data["im1Points"])
    im_pts2 = np.array(json_data["im2Points"])
    return im_pts1, im_pts2


def read_csv(points1_csv, points2_csv):
    points1 = pd.read_csv(points1_csv, header=None).to_numpy()
    points2 = pd.read_csv(points2_csv, header=None).to_numpy()

    return points1, points2


image1_path = "part4/house0.jpg"
image2_path = "part4/house1.jpg"
# point1_path = "part4/1.csv"
# point2_path = "part4/2.csv"
point_path = "part4/house0_house1.json"
image1 = plt.imread(image1_path)
image2 = plt.imread(image2_path)
offset = (0, 0)
im_pt1, im_pt2 = read_json(point_path)
H = compute_homography(im_pt1, im_pt2)

warp_image1 = warpImage(image1, H)
plt.imsave("left_warp.jpg", warp_image1)

blend_image = blend_image(warp_image1, image2, 0.55, offset, lap_layers=2)
plt.imsave("left_right_blend.jpg", blend_image)
