from utils import *
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    # part 1
    image_path = []  # choose your image path
    select_points(image_path)  # annotate points on image
    image = plt.imread(image_path)
    points = []  # Read the points by yourself
    triangles = perform_delaunay_triangulation(points)
    plot_delaunay_triangles(image, points, triangles)
    draw_points_on_image(image_path, points)

    # part 2
    image1_path = None
    image2_path = None
    image1_points_path = None
    image2_points_path = None
    # read points: different data have different types of points, plz read it by yourself
    ###
    # read points
    syn_mid_face(
        image1_path, image2_path, image1_points_path, image2_points_path
    )  # Keep the inputs as None, the function can use default paths to run

    # part 3
    morph_gif(
        path1=None, path2=None, image1_points=None, image2_points=None
    )  # Keep the inputs as None, the function can use default paths to run, path 1 and 2 are image paths

    # part 4

    # Assert that the images is named as {i}a.jpg
    point_path_list = [f"{i}a.pts" for i in range(1, 201, 1)]
    image_path_list = [f"{i}a.jpg" for i in range(1, 201, 1)]
    points = []
    images = []
    for pp, ip in zip(point_path_list, image_path_list):
        point_path = (
            "src_image\\frontalshapes_manuallyannotated_46points" + "\\" + pp
        )  # Change the str to your own folder
        image_path = (
            "src_image\\frontalimages_manuallyaligned" + "\\" + ip
        )  # Change the str to your own folder
        image = plt.imread(image_path)
        ps = read_pts_file(point_path)
        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=-1)
        images.append(image)
        h, w, _ = images[-1].shape
        ps.append([0, 0])  # 左上角
        ps.append([0, h - 1])  # 右上角
        ps.append([w - 1, 0])  # 左下角
        ps.append([w - 1, h - 1])  # 右下角
        points.append(np.array(ps))

    points = np.stack(points, axis=0)
    average_face(points, images)
    ave2myself()
    myself2ave()

    # part 5

    factor = 2  # Multiply factor, larger one will get a bigger distortion
    exaggerate_face(2)
