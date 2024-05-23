import numpy as np
import cv2

def filter_superpixels_by_color_mean(ids, superpixels):
    filtered_ids= []
    filtered_superpixels = []
    red_channel_mean_threshold = 130 
    green_channel_mean_threshold = 90 
    blue_channel_mean_threshold = 80 
    for id, superpixel in zip(ids, superpixels):
        red_channel_mean = sum(np.unique(superpixel[:,:,0])) / len(np.unique(superpixel[:,:,0]))
        green_channel_mean = sum(np.unique(superpixel[:,:,1])) / len(np.unique(superpixel[:,:,1]))
        blue_channel_mean = sum(np.unique(superpixel[:,:,2])) / len(np.unique(superpixel[:,:,2]))
        if red_channel_mean < red_channel_mean_threshold and  green_channel_mean < green_channel_mean_threshold and blue_channel_mean < blue_channel_mean_threshold:
            filtered_superpixels.append(superpixel)
            filtered_ids.append(id)
    return filtered_ids, filtered_superpixels

def get_superpixel_img(label_img:np.array, superpixel_id:int) -> np.array:    
    """
    Given a superpixel label array returns the selected superpixel as an binary array
    """
    # Selecting only the superpixel
    superpixel_img = (label_img * (label_img == superpixel_id))
    superpixel_img[superpixel_img!= 0] = 255
    return superpixel_img

def get_crop_coords(superpixel_img: np.array)-> tuple[int, int, int, int]: 
    # Find the indexes that are different from zero
    indexes = np.argwhere(superpixel_img != 0)

    # Finds the object limit
    (min_linha, min_coluna), (max_row, max_column) = indexes.min(0), indexes.max(0)

    # Computes the crop coords
    margin = 50  # Image margin adjustment
    line_start = max(0, min_linha - margin)
    line_end = min(superpixel_img.shape[0], max_row + margin)
    column_start = max(0, min_coluna - margin)
    column_end = min(superpixel_img.shape[1], max_column + margin)
    return line_start, line_end, column_start, column_end

def get_cropped_original_superpixel_img(original_img, label_img,  superpixel_id,  crop_size=100):
    """
    Finds the object and crops it on the superpixel, original and ground truth images
    """
    superpixel_img = get_superpixel_img(label_img, superpixel_id)
    line_start, line_end, column_start, column_end = get_crop_coords(superpixel_img)
    # Crops the images
    cropped_superpixel_img = superpixel_img[line_start:line_end, column_start:column_end]
    cropped_original_img = original_img[line_start:line_end, column_start:column_end]

    # Resizing the images w = crop_size, h = crop_size
    #cropped_superpixel_img = cropped_superpixel_img[:crop_size,:crop_size,]
    cropped_superpixel_img = cv2.resize(np.float32(cropped_superpixel_img),(100, 100))
    cropped_original_img = cv2.resize(np.float32(cropped_original_img),(100, 100))
    #cropped_original_img = cropped_original_img[:crop_size,:crop_size,]

    return np.float64(cropped_original_img), np.float64(cropped_superpixel_img)

def get_cropped_superpixel_img(original_img, superpixel_img, crop_size=100):
    """
    Finds the object and crops it on the superpixel, original and ground truth images
    """
    line_start, line_end, column_start, column_end = get_crop_coords(superpixel_img)
    # Crops the images
    cropped_superpixel_img = superpixel_img[line_start:line_end, column_start:column_end]
    cropped_original_img = original_img[line_start:line_end, column_start:column_end]

    # Resizing the images w = crop_size, h = crop_size
    #cropped_superpixel_img = cropped_superpixel_img[:crop_size,:crop_size,]
    cropped_superpixel_img = cv2.resize(np.float32(cropped_superpixel_img),(100, 100))
    cropped_original_img = cv2.resize(np.float32(cropped_original_img),(100, 100))
    #cropped_original_img = cropped_original_img[:crop_size,:crop_size,]

    return np.float64(cropped_original_img), np.float64(cropped_superpixel_img)

def get_reconstructed_image(label_img, selected_superpixels):
    """
    Given an selected superpixel id list and the complete label image, creates a
    new one with only the selected ones. 
    """
    reconstructed_image = np.zeros((label_img.shape[0], label_img.shape[1]))
    i = 0
    for positive_superpixel_id in selected_superpixels:
        reconstructed_image = reconstructed_image + get_superpixel_img(label_img, positive_superpixel_id)
        i+=1
    return reconstructed_image