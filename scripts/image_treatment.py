import numpy as np

def get_crop_coords(superpixel_img: np.array)-> tuple[int, int, int, int]: 
    """
    Given an superpixel image, finds it's object coords and computes the 
    crop coordinates
    """
    # Find the indexes that are different from zero
    indexes = np.argwhere(superpixel_img != 0)

    # Finds the object limit
    (min_line, min_column), (max_row, max_column) = indexes.min(0), indexes.max(0)

    # Computes the crop coords
    margin = 50  # Image margin adjustment
    line_start = max(0, min_line - margin)
    line_end = min(superpixel_img.shape[0], max_row + margin)
    column_start = max(0, min_column - margin)
    column_end = min(superpixel_img.shape[1], max_column + margin)
    return line_start, line_end, column_start, column_end

def get_superpixel_img(label_img:np.array, superpixel_id:int) -> np.array:    
    """
    Given a superpixel label array returns the selected superpixel as an binary array
    """
    # Selecting only the superpixel
    superpixel_img = (label_img * (label_img == superpixel_id))
    superpixel_img[superpixel_img!= 0] = 255
    return superpixel_img

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