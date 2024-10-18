import json

import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import map_coordinates


def read_json(path):
    with open(path, "r") as f:
        json_data = json.load(f)
    im_pts1 = np.array(json_data["im1Points"])
    im_pts2 = np.array(json_data["im2Points"])
    return im_pts1, im_pts2


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

    # 反归一化
    H = np.linalg.inv(T2) @ H_norm @ T1
    H /= H[2, 2]

    return H


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


def invert_homography(H):
    return np.linalg.inv(H)


# def warp_image(image, H, output_shape=None):
#     height, width, channels = image.shape

#     # Set output shape if not provided
#     if output_shape is None:
#         # Estimate output shape by transforming the corners of the image
#         corners = np.array(
#             [
#                 [0, 0, 1],
#                 [width - 1, 0, 1],
#                 [width - 1, height - 1, 1],
#                 [0, height - 1, 1],
#             ]
#         ).T
#         transformed_corners = H @ corners
#         transformed_corners /= transformed_corners[2, :]
#         min_x, min_y = np.min(transformed_corners[:2], axis=1)
#         max_x, max_y = np.max(transformed_corners[:2], axis=1)
#         output_shape = (int(np.ceil(max_y - min_y)), int(np.ceil(max_x - min_x)))

#     output_height, output_width = output_shape

#     # Generate all coordinate points for output image
#     indices_y, indices_x = np.indices((output_height, output_width))
#     indices = np.stack(
#         [indices_x.ravel(), indices_y.ravel(), np.ones_like(indices_x).ravel()], axis=0
#     )

#     # Apply inverse homography transformation to map output coordinates back to input coordinates
#     H_inv = invert_homography(H)
#     mapped_coords = H_inv @ indices
#     mapped_coords /= mapped_coords[2, :]  # Normalize homogeneous coordinates

#     # Separate x and y coordinates
#     x_coords = mapped_coords[0].reshape(output_height, output_width)
#     y_coords = mapped_coords[1].reshape(output_height, output_width)

#     # Create output image
#     new_image = np.zeros((output_height, output_width, channels), dtype=image.dtype)

#     # Bilinear interpolation for each channel using map_coordinates
#     for c in range(channels):
#         new_image[:, :, c] = map_coordinates(
#             image[:, :, c], [y_coords, x_coords], order=1, mode="nearest"
#         )

#     return new_image


def warpImage(im, H):
    # 计算输入图像的高度h和宽度w
    h, w = im.shape[:2]

    # 定义输入图像的四个角点坐标（x, y）
    corners = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])

    # 将角点坐标转换为齐次坐标（添加一列1）
    ones = np.ones((4, 1))
    corners_hom = np.hstack([corners, ones])  # 4 x 3

    # 使用H矩阵将角点映射到目标图像坐标
    transformed_corners = (H @ corners_hom.T).T  # 4 x 3

    # 归一化齐次坐标
    transformed_corners = transformed_corners[:, :2] / transformed_corners[:, 2:]

    # 计算输出图像的边界框（最小和最大x、y值）
    min_x = int(np.floor(np.min(transformed_corners[:, 0])))
    max_x = int(np.ceil(np.max(transformed_corners[:, 0])))
    min_y = int(np.floor(np.min(transformed_corners[:, 1])))
    max_y = int(np.ceil(np.max(transformed_corners[:, 1])))

    # 生成输出图像的网格坐标
    x_range = np.arange(min_x, max_x + 1)
    y_range = np.arange(min_y, max_y + 1)
    xv, yv = np.meshgrid(x_range, y_range)

    # 将网格坐标展开并转换为齐次坐标
    grid_points = np.stack([xv.flatten(), yv.flatten()], axis=-1)  # N x 2
    ones = np.ones((grid_points.shape[0], 1))
    grid_points_hom = np.hstack([grid_points, ones])  # N x 3

    # 计算H的逆矩阵
    H_inv = np.linalg.inv(H)

    # 使用H的逆矩阵将输出图像的坐标映射回输入图像坐标
    src_points = (H_inv @ grid_points_hom.T).T  # N x 3

    # 归一化齐次坐标
    src_points = src_points[:, :2] / src_points[:, 2:]

    # 获取输入图像的像素坐标
    src_x = src_points[:, 0].reshape(yv.shape)
    src_y = src_points[:, 1].reshape(yv.shape)

    # 创建一个掩码，标记映射到输入图像范围内的有效点
    valid_mask = (src_x >= 0) & (src_x <= w - 1) & (src_y >= 0) & (src_y <= h - 1)

    # 初始化输出图像
    out_shape = yv.shape
    if im.ndim == 3:
        # 彩色图像
        imwarped = np.zeros(
            (out_shape[0], out_shape[1], im.shape[2] + 1), dtype=im.dtype
        )
    else:
        # 灰度图像
        imwarped = np.zeros((out_shape[0], out_shape[1], 2), dtype=im.dtype)

    # 对每个通道进行插值
    for i in range(im.shape[2]) if im.ndim == 3 else range(1):
        if im.ndim == 3:
            channel = im[:, :, i]
        else:
            channel = im

        # 使用map_coordinates进行插值
        coords = [src_y, src_x]  # 坐标顺序为(y, x)
        channel_warped = map_coordinates(
            channel, coords, order=1, mode="constant", cval=0.0
        )

        # 将插值结果放入输出图像
        imwarped[:, :, i] = channel_warped

    # 生成alpha通道
    alpha = valid_mask.astype(im.dtype)

    return imwarped[:, :, :3], alpha


def pad_image_to_height(image, target_height):
    h, w = image.shape[:2]
    pad_total = target_height - h
    pad_top = pad_total // 2
    pad_bottom = pad_total - pad_top
    if image.ndim == 3:
        # 彩色图像
        padded_image = np.pad(
            image,
            ((pad_top, pad_bottom), (0, 0), (0, 0)),
            mode="constant",
            constant_values=0,
        )
    else:
        # 灰度图像或alpha通道
        padded_image = np.pad(
            image, ((pad_top, pad_bottom), (0, 0)), mode="constant", constant_values=0
        )
    return padded_image


def blend_images_with_overlap_fixed(image1, alpha1, image2, overlap_ratio):
    """
    将两张图像左右拼接，考虑指定的重叠比例，对重叠区域进行平均融合。
    如果两张图像的高度不一致，对较小的图像进行上下均匀填充。
    对于 image2，如果需要，将其在右侧进行填充，以匹配输出图像的宽度。

    参数：
    - image1: 左侧图像，numpy数组，形状为(H1, W1, C)
    - alpha1: 左侧图像的alpha通道，numpy数组，形状为(H1, W1)
    - image2: 右侧图像，numpy数组，形状为(H2, W2, C)
    - overlap_ratio: 重叠比例（0~1之间的浮点数）

    返回：
    - result_image: 融合后的图像，numpy数组，形状为(H, output_width, C)
    """
    h1, w1 = image1.shape[:2]
    h2, w2 = image2.shape[:2]

    # 确定目标高度，为两张图像高度的最大值
    target_height = max(h1, h2)

    # 如果图像高度不一致，对较小的图像进行上下均匀填充
    def pad_image_to_height(image, target_height):
        h, w = image.shape[:2]
        pad_total = target_height - h
        pad_top = pad_total // 2
        pad_bottom = pad_total - pad_top
        if image.ndim == 3:
            # 彩色图像
            padded_image = np.pad(
                image,
                ((pad_top, pad_bottom), (0, 0), (0, 0)),
                mode="constant",
                constant_values=0,
            )
        else:
            # 灰度图像或alpha通道
            padded_image = np.pad(
                image,
                ((pad_top, pad_bottom), (0, 0)),
                mode="constant",
                constant_values=0,
            )
        return padded_image

    if h1 != target_height:
        image1 = pad_image_to_height(image1, target_height)
        alpha1 = pad_image_to_height(alpha1, target_height)
    if h2 != target_height:
        image2 = pad_image_to_height(image2, target_height)
        alpha2 = np.ones((target_height, image2.shape[1]), dtype=np.float32)
    else:
        alpha2 = np.ones((h2, w2), dtype=np.float32)

    # 重新获取填充后的图像尺寸
    h1, w1 = image1.shape[:2]
    h2, w2 = image2.shape[:2]

    # 计算重叠宽度
    overlap_width = int(w1 * overlap_ratio)

    # 计算输出图像的尺寸
    output_width = w1 + w2 - overlap_width
    output_height = target_height

    # 初始化输出图像和权重图像
    channels = image1.shape[2] if image1.ndim == 3 else 1
    result_image = np.zeros((output_height, output_width, channels), dtype=np.float32)
    weight_sum = np.zeros((output_height, output_width), dtype=np.float32)

    # 处理左侧图像（image1）
    x1_start = 0
    x1_end = w1
    result_image[:, x1_start:x1_end] += image1 * alpha1[..., np.newaxis]
    weight_sum[:, x1_start:x1_end] += alpha1

    # 处理右侧图像（image2）
    x2_start = output_width - w2
    x2_end = output_width
    # 检查image2的宽度是否与预期的区域宽度匹配
    expected_width = x2_end - x2_start
    if w2 < expected_width:
        # 需要在右侧进行填充
        pad_width = expected_width - w2
        if image2.ndim == 3:
            image2 = np.pad(
                image2,
                ((0, 0), (0, pad_width), (0, 0)),
                mode="constant",
                constant_values=0,
            )
        else:
            image2 = np.pad(
                image2, ((0, 0), (0, pad_width)), mode="constant", constant_values=0
            )
        alpha2 = np.pad(
            alpha2, ((0, 0), (0, pad_width)), mode="constant", constant_values=0
        )
        w2 = image2.shape[1]  # 更新w2

    result_image[:, x2_start:x2_end] += image2 * alpha2[..., np.newaxis]
    weight_sum[:, x2_start:x2_end] += alpha2

    # 对重叠区域的权重进行调整，使得从左到右权重线性变化
    overlap_start = w1 - overlap_width
    overlap_end = w1
    for x in range(overlap_width):
        alpha = (overlap_width - x) / overlap_width  # 从1逐渐减小到0
        idx1 = overlap_start + x  # image1中的位置
        idx2 = x  # image2中的位置
        result_image[:, idx1] = image1[:, idx1] * alpha + image2[:, idx2] * (1 - alpha)
        weight_sum[:, idx1] = 1  # 已经手动计算，不需要再除以权重

    # 避免除以零
    non_zero_mask = weight_sum > 0
    result_image[non_zero_mask] /= weight_sum[non_zero_mask, np.newaxis]

    # 将结果裁剪到[0, 255]并转换为uint8类型
    result_image = np.clip(result_image, 0, 255).astype(np.uint8)

    return result_image


def pad_image(image, target_height, target_width):
    """
    在图像的上、下、右三个方向进行填充，使其达到目标尺寸。

    参数：
    - image: 输入图像，形状为 (h, w) 或 (h, w, c)
    - target_height: 目标高度
    - target_width: 目标宽度

    返回：
    - padded_image: 填充后的图像，形状为 (target_height, target_width) 或 (target_height, target_width, c)
    """
    h, w = image.shape[:2]
    channels = image.shape[2] if image.ndim == 3 else None

    # 计算需要填充的大小
    pad_top = (target_height - h) // 2
    pad_bottom = target_height - h - pad_top
    pad_left = 0  # 左侧不填充
    pad_right = target_width - w

    # 检查是否需要填充负数（即目标尺寸小于原始尺寸）
    if pad_top < 0 or pad_bottom < 0 or pad_right < 0:
        raise ValueError("目标尺寸小于原始图像尺寸，无法进行填充。")

    # 构建pad_width参数
    if channels is not None:
        # 彩色图像
        pad_width = ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0))
    else:
        # 灰度图像
        pad_width = ((pad_top, pad_bottom), (pad_left, pad_right))

    # 对图像进行填充，填充值为0（黑色）
    padded_image = np.pad(image, pad_width, mode="constant", constant_values=0)

    return padded_image


# Main execution

# Read feature points and images
im_pt1, im_pt2 = read_json("part3/ground_ground_white (3).json")
image1 = plt.imread("part3/ground.jpg")
image2 = plt.imread("part3/ground_white.jpg")
H = compute_homography(im_pt1, im_pt2)
new_image, alpha = warpImage(image1, H)
plt.imsave("test.jpg", new_image.astype(np.uint8))

offset = (new_image.shape[2] * 0.5, 0)
mosaic_image = blend_images_with_overlap_fixed(new_image, alpha, image2, 666 / 888)
plt.imsave("test2.jpg", mosaic_image.astype(np.uint8))

# # Calculate target image size and overlap width
# overlap_percentage = 0.7  # Set to 0.7 or adjust as needed
# height1, width1, channels = new_image.shape
# height2, width2, _ = image2.shape

# target_height = max(height1, height2)
# target_width = width1 + width2 - int(width2 * overlap_percentage)
# target_size = (target_height, target_width, channels)

# # Blend images and save
# blend_image = mosaic_blend(new_image, image2, target_size)
# plt.imsave("mosaic.jpg", blend_image)
