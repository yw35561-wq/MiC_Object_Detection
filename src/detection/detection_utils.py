import numpy as np


def non_max_suppression(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list:
    """
    Apply Non-Maximum Suppression to filter overlapping bounding boxes
    :param boxes: Bounding boxes array (n, 4) in xyxy format
    :param scores: Confidence scores array (n,)
    :param iou_threshold: IOU threshold for suppression
    :return: Indices of kept boxes
    """
    indices = np.argsort(scores)[::-1]
    keep = []

    while len(indices) > 0:
        current = indices[0]
        keep.append(current)

        if len(indices) == 1:
            break

        # Calculate IOU between current box and remaining boxes
        iou = compute_iou(boxes[current], boxes[indices[1:]])
        # Keep boxes with IOU below threshold
        indices = indices[1:][iou < iou_threshold]

    return keep


def compute_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """
    Calculate IOU between one box and multiple boxes
    :param box: Single box array (4,) in xyxy format
    :param boxes: Multiple boxes array (n, 4) in xyxy format
    :return: IOU array (n,)
    """
    # Calculate intersection coordinates
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    # Calculate intersection area
    intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

    # Calculate box areas
    area_box = (box[2] - box[0]) * (box[3] - box[1])
    area_boxes = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    # Calculate union area and IOU
    union = area_box + area_boxes - intersection
    return intersection / union