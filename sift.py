import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image

def extract_sift_keypoints(image_path, n_features=20, contrast_threshold=0.001):
    """
    extract SIFT keypoints

    :param image_path
    :param n_features: num of keypoints
    :param contrast_threshold: filter of low contrast(lower value, more keypoints)
    :return: query_points (Nx2) np.array
    """
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 2. create SIFT
    sift = cv2.SIFT_create(
        nfeatures=n_features,          # num of keypoints
        contrastThreshold=contrast_threshold  # filter of low contrast
    )
    
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    
    # 4. extract query_points
    query_points = np.array([kp.pt for kp in keypoints])  # [[x1,y1], [x2,y2], ...]
    
    return query_points

def map_keypoints_to_padded(orig_keypoints, orig_size, new_size, padding):
    """
    correct keypoints toward pad
    
    :param orig_keypoints: [N, 2] (x, y)
    :param orig_size: (width, height)
    :param new_size: (width, height) - before pad!!
    :param padding: {'top':, 'bottom':, 'left':, 'right':}
    :return: [N, 2]
    """
    orig_w, orig_h = orig_size
    new_w_actual, new_h_actual = new_size  
    
    scale_x = new_w_actual / orig_w
    scale_y = new_h_actual / orig_h
    
    # scale points
    scaled_points = orig_keypoints * np.array([scale_x, scale_y])
    
    scaled_points[:, 0] += padding['left']  # scale axis x
    scaled_points[:, 1] += padding['top']   # scale axis y
    
    return scaled_points

def get_preprocess_params(image_path, mode='pad'):

    img = Image.open(image_path)
    orig_size = img.size  # (width, height)
    target_size = 518
    padding_info = {'top': 0, 'bottom': 0, 'left': 0, 'right': 0}
    new_size_actual = None

    if mode == 'pad':
        width, height = orig_size
        if width >= height:
            # width=518
            new_w_actual = target_size
            new_h_actual = round(height * (new_w_actual / width) / 14) * 14
            padding_info['top'] = (target_size - new_h_actual) // 2
            padding_info['bottom'] = target_size - new_h_actual - padding_info['top']
            new_size_actual = (new_w_actual, new_h_actual)
        else:
            # height=518
            new_h_actual = target_size
            new_w_actual = round(width * (new_h_actual / height) / 14) * 14
            padding_info['left'] = (target_size - new_w_actual) // 2
            padding_info['right'] = target_size - new_w_actual - padding_info['left']
            new_size_actual = (new_w_actual, new_h_actual)
            
    return orig_size, new_size_actual, padding_info

def get_query_points(image_path, n_features=20, contrast_threshold=0.001):
    cv2.setRNGSeed(8)
    np.random.seed(0)
    orig_keypoints = extract_sift_keypoints(image_path, n_features, contrast_threshold)
    
    orig_size, new_size_actual, padding_info = get_preprocess_params(image_path, mode='pad')
    
    mapped_points = map_keypoints_to_padded(
        orig_keypoints, orig_size, new_size_actual, padding_info
    )

    if len(mapped_points) > n_features:
        mapped_points = mapped_points[:n_features]

    query_points = torch.tensor(mapped_points, dtype=torch.float32)

    return query_points
