import logging
import time
from types import SimpleNamespace

import cv2
import numpy as np

import config
from utils import (
    adjust_image_size,
    average_channel,
    cut_image,
    load_image,
    match_image,
)

cfg = SimpleNamespace(**vars(config))
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    datas, image_dirs = load_image()
    for data, image_dir in zip(datas, image_dirs):
        start_time = time.time()
        logging.info(image_dir)
        B = data[0 : data.shape[0] // 3, :]
        G = data[data.shape[0] // 3 : data.shape[0] // 3 * 2, :]
        R = data[data.shape[0] // 3 * 2 : data.shape[0] // 3 * 3, :]
        B, G, R = cut_image(B, G, R)
        B = B.astype(np.uint8)
        G = G.astype(np.uint8)
        R = R.astype(np.uint8)
        if cfg.base_dir == "base_tif":
            B = cv2.GaussianBlur(B, (7, 7), 2.0).astype(np.uint8)
            G = cv2.GaussianBlur(G, (7, 7), 2.0).astype(np.uint8)
            R = cv2.GaussianBlur(R, (7, 7), 2.0).astype(np.uint8)

        if G.shape != B.shape:
            G = adjust_image_size(G, B)
        if R.shape != B.shape:
            R = adjust_image_size(R, B)

        B, G, R = match_image(B, G, R, 0)
        image = np.dstack([B, G, R])
        image = average_channel(image)
        cv2.imwrite(image_dir, image)
        end_time = time.time()
        use_time = end_time - start_time
        logging.info("time_cost:{}".format(use_time))
