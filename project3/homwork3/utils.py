import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Delaunay
from scipy.ndimage import map_coordinates
from skimage.draw import polygon2mask
import json
from PIL import Image
import cv2


def select_points(image_path):
    img = plt.imread(image_path)
    plt.imshow(img)
    plt.title("Click on the image to select points")

    points = plt.ginput(n=0, timeout=0)
    plt.close()

    return points


def calculate_mid_point(pointA, pointB):
    assert len(pointA) == len(pointB)

    return (np.array(pointA) + np.array(pointB)) // 2


def perform_delaunay_triangulation(points):
    # 计算 Delaunay 三角剖分
    tri = Delaunay(points)
    return tri.simplices


def morph(im1, im2, im1_pts, im2_pts, warp_frac, dissolve_frac):
    intermediate_pts = (1 - warp_frac) * im1_pts + warp_frac * im2_pts
    triangles = perform_delaunay_triangulation(intermediate_pts)

    im1_warped = warpimage(im1, im1_pts, intermediate_pts, triangles)
    im2_warped = warpimage(im2, im2_pts, intermediate_pts, triangles)

    morphed_im = (1 - dissolve_frac) * im1_warped.astype(
        np.float32
    ) + dissolve_frac * im2_warped.astype(np.float32)

    morphed_im = np.clip(morphed_im, 0, 255).astype(np.uint8)

    return morphed_im


def computeAffine(src_tri, dst_tri):
    A = np.vstack([src_tri.T, np.ones((1, src_tri.shape[0]))])
    B = np.vstack([dst_tri.T, np.ones((1, dst_tri.shape[0]))])
    transform = np.linalg.solve(A.T, B.T).T
    return transform


def warpimage(image, src_pts, dst_pts, triangles):
    src_pts = src_pts[:, [1, 0]]
    dst_pts = dst_pts[:, [1, 0]]

    h, w, _ = image.shape
    image_warp = np.zeros_like(image)

    for tri_indices in triangles:
        src_tri = src_pts[tri_indices]
        dst_tri = dst_pts[tri_indices]

        affine_mat = computeAffine(src_tri, dst_tri)
        affine_mat_inv = np.linalg.inv(affine_mat)[:2, :]

        mask = polygon2mask((h, w), dst_tri)
        y_coords, x_coords = np.where(mask)

        ones = np.ones((len(x_coords),))
        dst_coords = np.vstack([y_coords, x_coords, ones])  # [y, x, 1]
        src_coords = np.dot(affine_mat_inv, dst_coords)

        valid_coords = (
            (src_coords[0] >= 0)
            & (src_coords[0] < h)
            & (src_coords[1] >= 0)
            & (src_coords[1] < w)
        )
        y_valid = y_coords[valid_coords]
        x_valid = x_coords[valid_coords]
        valid_src_coords = src_coords[:, valid_coords]

        for ch in range(3):
            image_warp[y_valid, x_valid, ch] = map_coordinates(
                image[:, :, ch],
                [valid_src_coords[0], valid_src_coords[1]],
                order=1,
                mode="nearest",
            )

    return image_warp


def plot_delaunay_triangles(image, points, triangles):
    plt.figure(figsize=(8, 8))
    plt.imshow(image, cmap="gray")
    plt.triplot(points[:, 0], points[:, 1], triangles, color="red")
    plt.plot(points[:, 0], points[:, 1], "o", color="yellow", markersize=3)
    plt.axis("off")
    plt.show()


def syn_mid_face(path1, path2, image1_points, image2_points):
    if (
        path1 is None
        and path2 is None
        and image1_points is None
        and image2_points is None
    ):
        path1 = "src_image\\myself.jpg"
        path2 = "src_image\\picard.jpg"
        image1 = plt.imread(path1)
        image2 = plt.imread(path2)
        with open("myself_picard.json", "r", encoding="utf-8") as file:
            data = json.load(file)
        image1_points = np.array(data["im1Points"])
        image2_points = np.array(data["im2Points"])
    else:
        image1 = plt.imread(path1)
        image2 = plt.imread(path2)
    mid_points = calculate_mid_point(image1_points, image2_points)
    assert len(image1_points) == len(image2_points)
    mid_tri = perform_delaunay_triangulation(mid_points)
    plot_delaunay_triangles(image2, mid_points, mid_tri)

    warp_image1 = warpimage(image1, image1_points, mid_points, mid_tri)
    plt.imshow(warp_image1)
    plt.show()
    warp_image2 = warpimage(image2, image2_points, mid_points, mid_tri)
    plt.imshow(warp_image2)
    plt.show()
    mid_image = warp_image1 // 2 + warp_image2 // 2
    plt.imshow(mid_image)
    plt.show()


def morph_gif(path1, path2, image1_points, image2_points):
    if (
        path1 is None
        and path2 is None
        and image1_points is None
        and image2_points is None
    ):
        path1 = "src_image\\myself.jpg"
        path2 = "src_image\\picard.jpg"
        with open("myself_picard.json", "r", encoding="utf-8") as file:
            data = json.load(file)
        image1_points = np.array(data["im1Points"])
        image2_points = np.array(data["im2Points"])
    else:
        image1 = plt.imread(path1)
        image2 = plt.imread(path2)

    warp_frac = np.linspace(0, 1, 45)
    dissolve_frac = np.linspace(0, 1, 45)

    mid_images = []
    for w, d in zip(dissolve_frac, warp_frac):
        warp_image = morph(image1, image2, image1_points, image2_points, w, d)
        mid_images.append(warp_image)
    first_image = Image.fromarray(mid_images[0])
    frame_images = [Image.fromarray(im) for im in mid_images[1:]]
    first_image.save(
        "morph.gif",
        save_all=True,
        append_images=frame_images,
        duration=1000 // 30,
        loop=0,
    )


def average_face(images, points):
    # point_path_list = [f"{i}a.pts" for i in range(1, 201, 1)]
    # image_path_list = [f"{i}a.jpg" for i in range(1, 201, 1)]
    # points = []
    # images = []
    # for pp, ip in zip(point_path_list, image_path_list):
    #     point_path = "src_image\\frontalshapes_manuallyannotated_46points" + "\\" + pp
    #     image_path = "src_image\\frontalimages_manuallyaligned" + "\\" + ip
    #     image = plt.imread(image_path)
    #     ps = read_pts_file(point_path)
    #     if len(image.shape) == 2:
    #         image = np.stack([image] * 3, axis=-1)
    #     images.append(image)
    #     h, w, _ = images[-1].shape
    #     ps.append([0, 0])  # 左上角
    #     ps.append([0, h - 1])  # 右上角
    #     ps.append([w - 1, 0])  # 左下角
    #     ps.append([w - 1, h - 1])  # 右下角
    #     points.append(np.array(ps))

    points = np.stack(points, axis=0)
    mid_points = np.mean(points, axis=0)
    mid_tri = perform_delaunay_triangulation(mid_points)
    ave_image = np.zeros_like(images[0]).astype(np.float32)
    for point, image in zip(points, images):
        warp_image = warpimage(image, point, mid_points, mid_tri).astype(np.float32)
        plt.imshow(warp_image.astype(np.uint8))
        plt.show()
        ave_image = ave_image + warp_image
    ave_image = ave_image // len(images)
    ave_image = ave_image.astype(np.uint8)
    plt.imshow(ave_image)
    plt.show()


def read_pts_file(file_path):
    points = []
    with open(file_path, "r") as file:
        lines = file.readlines()
        # 忽略前两行和最后一行
        coord_lines = lines[3:-1]
        for line in coord_lines:
            coords = list(map(float, line.strip().split()))
            points.append(coords)
    return points


def get_mid_point():
    point_path_list = [f"{i}a.pts" for i in range(1, 201, 1)]
    image_path_list = [f"{i}a.jpg" for i in range(1, 201, 1)]
    points = []
    images = []
    for pp, ip in zip(point_path_list, image_path_list):
        point_path = "src_image\\frontalshapes_manuallyannotated_46points" + "\\" + pp
        image_path = "src_image\\frontalimages_manuallyaligned" + "\\" + ip
        image = plt.imread(image_path)
        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=-1)

        images.append(image)

        ps = read_pts_file(point_path)
        h, w, _ = images[-1].shape
        ps.append([0, 0])
        ps.append([0, h - 1])
        ps.append([w - 1, 0])
        ps.append([w - 1, h - 1])
        points.append(np.array(ps))

    points = np.stack(points, axis=0)
    mid_points = np.mean(points, axis=0)
    return mid_points


def draw_points_on_image(image_path, points):
    image = Image.open(image_path).convert("RGB")

    img_width, img_height = image.size

    fig, ax = plt.subplots()
    ax.imshow(image)
    ax.axis("off")

    for idx, (x, y) in enumerate(points):
        x = min(max(0, x), img_width - 1)
        y = min(max(0, y), img_height - 1)
        ax.add_patch(plt.Circle((x, y), radius=3, color="red"))
        ax.text(x + 5, y - 10, str(idx + 1), color="blue", fontsize=8, weight="bold")

    plt.show()


def myself2ave(path="myself_ave.jpg", point_path="myself_ave_1a.json"):
    myself_image = plt.imread(path)
    h, w, _ = myself_image.shape

    with open(point_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    myself_points = data["im1Points"]
    myself_points.append([0, 0])
    myself_points.append([0, h - 1])
    myself_points.append([w - 1, 0])
    myself_points.append([w - 1, h - 1])
    myself_points = np.array(myself_points)
    mid_point = get_mid_point()
    mid_tri = perform_delaunay_triangulation(mid_point)
    warp_myself = warpimage(myself_image, myself_points, mid_point, mid_tri)
    plt.imshow(warp_myself)
    plt.show()


def ave2myself():
    myself_image = plt.imread("myself_ave.jpg")
    h, w, _ = myself_image.shape

    with open("myself_ave_1a.json", "r", encoding="utf-8") as file:
        data = json.load(file)
    myself_points = data["im1Points"]
    myself_points.append([0, 0])
    myself_points.append([0, h - 1])
    myself_points.append([w - 1, 0])
    myself_points.append([w - 1, h - 1])
    myself_points = np.array(myself_points)

    my_tri = perform_delaunay_triangulation(myself_points)
    point_path_list = [f"{i}a.pts" for i in range(1, 201, 1)]
    image_path_list = [f"{i}a.jpg" for i in range(1, 201, 1)]
    points = []
    images = []
    for pp, ip in zip(point_path_list, image_path_list):
        point_path = "src_image\\frontalshapes_manuallyannotated_46points" + "\\" + pp
        image_path = "src_image\\frontalimages_manuallyaligned" + "\\" + ip
        image = plt.imread(image_path)
        ps = read_pts_file(point_path)
        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=-1)
        images.append(image)
        h, w, _ = images[-1].shape
        ps.append([0, 0])
        ps.append([0, h - 1])
        ps.append([w - 1, 0])
        ps.append([w - 1, h - 1])
        points.append(np.array(ps))

    points = np.stack(points, axis=0)
    ave_image = np.zeros_like(images[0]).astype(np.float32)
    for point, image in zip(points, images):
        warp_image = warpimage(image, point, myself_points, my_tri).astype(np.float32)

        ave_image = ave_image + warp_image
    ave_image = ave_image // len(images)
    ave_image = ave_image.astype(np.uint8)
    plt.imshow(ave_image)
    plt.show()


def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)
    h, w, _ = plt.imread("src_image\\video\\Archer.jpg").shape
    image1_points = data["im1Points"]
    image1_points.append([0, 0])
    image1_points.append([0, h - 1])
    image1_points.append([w - 1, 0])
    image1_points.append([w - 1, h - 1])
    image1_points = np.array(image1_points)

    image2_points = data["im2Points"]
    image2_points.append([0, 0])
    image2_points.append([0, h - 1])
    image2_points.append([w - 1, 0])
    image2_points.append([w - 1, h - 1])
    image2_points = np.array(image2_points)
    return image1_points, image2_points


# average_face()
def get_video_points():
    path1 = "Archer_Pike.json"
    path2 = "Kirk_picard.json"
    path3 = "Janeway_Sisco.json"
    points = []
    image1_points, image2_points = load_json(path1)
    points.append(image1_points)
    points.append(image2_points)
    image1_points, image2_points = load_json(path2)
    points.append(image1_points)
    points.append(image2_points)
    image1_points, image2_points = load_json(path3)
    points.append(image1_points)
    points.append(image2_points)

    points = np.array(points)
    return points


def morph_image(image1, image2, image1_points, image2_points):
    warp_frac = np.linspace(0, 1, 90) ** 2
    dissolve_frac = np.linspace(0, 1, 90) ** 2
    if image1 is None:
        image1 = np.zeros_like(image2).astype(np.float32)
        image1 = np.where(image1 == 0, 255, 0)
        image1_points = image2_points
    if image2 is None:
        image2 = np.zeros_like(image1).astype(np.float32)
        image2 = np.where(image2 == 0, 255, 0)
        image2_points = image1_points
    mid_images = []
    for w, d in zip(dissolve_frac, warp_frac):
        warp_image = morph(image1, image2, image1_points, image2_points, w, d)
        mid_images.append(warp_image)
    return mid_images


def create_video():
    image_list = [
        "Archer.jpg",
        "Pike.jpg",
        "Kirk.jpg",
        "picard (1).jpg",
        "Janeway.jpg",
        "Sisco.jpg",
    ]
    base_path = "src_image\\video"
    points = get_video_points()
    images = []
    for i in range(len(image_list) + 1):
        if i == 0:
            image2_path = base_path + "\\" + image_list[i]
            image2 = plt.imread(image2_path)
            image2_points = points[i]
            mid_images = morph_image(None, image2, None, image2_points)
            images.extend(mid_images)
        elif i == len(image_list):
            image1_path = base_path + "\\" + image_list[i - 1]
            image1 = plt.imread(image1_path)
            image1_points = points[i - 1]
            mid_images = morph_image(image1, None, image1_points, None)
            images.extend(mid_images)
        else:
            image1_path = base_path + "\\" + image_list[i - 1]
            image2_path = base_path + "\\" + image_list[i]
            image1 = plt.imread(image1_path)
            image2 = plt.imread(image2_path)
            image1_points = points[i - 1]
            image2_points = points[i]

            mid_images = morph_image(image1, image2, image1_points, image2_points)
            images.extend(mid_images)
    base_video_path = "videos"
    image_paths = []
    for i in range(len(images)):
        path = base_video_path + f"\\{i}.jpg"
        plt.imsave(path, images[i])
        plt.close()
        image_paths.append(path)
    images_to_video(image_paths, "Star_Trek.mp4", 10)


def images_to_video(images_list, video_name, fps):
    if not images_list:
        raise ValueError("The image list is empty.")
    frame = cv2.imread(images_list[0])
    height, width, layers = frame.shape

    video = cv2.VideoWriter(
        video_name, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )

    for image_path in images_list:
        frame = cv2.imread(image_path)
        if frame is not None:
            video.write(frame)
        else:
            print(f"Warning: Image {image_path} could not be read.")

    video.release()


def exaggerate_face(exaggeration_factor):
    myself_image = plt.imread("myself_ave.jpg")
    h, w, _ = myself_image.shape

    with open("myself_ave_1a.json", "r", encoding="utf-8") as file:
        data = json.load(file)
    myself_points = data["im1Points"]
    myself_points.append([0, 0])
    myself_points.append([0, h - 1])
    myself_points.append([w - 1, 0])
    myself_points.append([w - 1, h - 1])
    myself_points = np.array(myself_points)
    mid_point = get_mid_point()

    displacement = myself_points - mid_point

    exaggerated_keypoints = myself_points + exaggeration_factor * displacement
    tri = perform_delaunay_triangulation(exaggerated_keypoints)
    exaggerated_face = warpimage(
        myself_image, myself_points, exaggerated_keypoints, tri
    )
    plt.imshow(exaggerated_face.astype(np.uint8))
    plt.show()
