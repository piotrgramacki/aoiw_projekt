from typing import Callable, List
import cv2
import os
from tqdm import tqdm
import numpy as np

from src.utils import create_path_if_not_exists
from PIL import Image

def equalize_histogram(image_bgr):
    img_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    img_hsv[:, :, 2] = cv2.equalizeHist(img_hsv[:, :, 2])
    dst_img = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
    return dst_img

def equalize_histogram_pil(pil_image):
    bgr_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    equalized_bgr = equalize_histogram(bgr_image)
    equalized_rgb = cv2.cvtColor(equalized_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(equalized_rgb)


def blur(image_bgr):
    blurred = cv2.GaussianBlur(image_bgr,(3,3),0)
    return blurred


def transform_images(input_path: str, output_path: str, transformations: List[Callable]):
    create_path_if_not_exists(output_path)
    for class_name in tqdm(os.listdir(input_path)):
        class_input_path = os.path.join(input_path, class_name)
        class_output_path = os.path.join(output_path, class_name)
        create_path_if_not_exists(class_output_path)

        for file_name in tqdm(os.listdir(class_input_path)):
            input_file_path =  os.path.join(class_input_path, file_name)
            output_file_path =  os.path.join(class_output_path, file_name)

            src_img = cv2.imread(input_file_path)
            dst_image = src_img
            
            for t in transformations:
                dst_image = t(dst_image)
            
            cv2.imwrite(output_file_path, dst_image)


if __name__ == "__main__": 
    from src.settings import UC_MERCED_DATA_DIRECTORY, RAW_DATA_DIRECTORY
    dest_path = os.path.join(RAW_DATA_DIRECTORY, "uc_merced_blur")
    transform_images(UC_MERCED_DATA_DIRECTORY, dest_path, [blur])
    
    dest_path = os.path.join(RAW_DATA_DIRECTORY, "uc_merced_eq_blur")
    transform_images(UC_MERCED_DATA_DIRECTORY, dest_path, [equalize_histogram, blur])