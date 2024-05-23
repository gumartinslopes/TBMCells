import numpy as np

def calculate_iou(image1: np.array, image2: np.array) -> float:
    """Computes Intersection Over Union."""
    intersection = np.logical_and(image1, image2)
    union = np.logical_or(image1, image2)
    iou = np.sum(intersection) / np.sum(union)
    return iou

def calculate_dice(image1: np.array, image2: np.array) -> float:
    """ Computes the Dice Coheficient metric."""
    intersection = np.sum((image1 & image2))
    sum = abs(np.sum(image1)) + abs(np.sum(image2))
    dice = (2. * intersection) / sum
    return dice

def calculate_precision(tp:int, fp:int) -> float:
    """
    Computes the precision metric:
    True Positives/(True Positives + False Positives)
    """
    return tp/(tp + fp)

def calculate_recall(tp:int, fn:int) -> float:
    """
    Computes the recall metric:
    True Positives/(TruePositives + False Negatives)
    """
    return tp/(tp + fn)

def calculate_f1_score(precision:float, recall:float) -> float:
    """
    Computes the recall metric:
    2*(precision * recall)/(precision + recall)
    """
    return 2 * (precision * recall) / (precision + recall)