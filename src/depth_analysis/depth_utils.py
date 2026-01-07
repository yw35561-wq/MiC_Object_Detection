import cv2
import numpy as np


def depth_img_read(depth_path: str) -> np.ndarray:
    """
    Read depth image from file (supports 16-bit/32-bit depth formats)
    :param depth_path: Path to depth image file
    :return: Depth image array
    """
    depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if depth_img is None:
        raise ValueError(f"Failed to read depth image from: {depth_path}")
    return depth_img


def preprocess_depth_img(depth_img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Preprocess depth image: denoising and normalization
    :param depth_img: Raw depth image array
    :return: (processed depth image, normalized depth image for visualization)
    """
    # Median filtering for denoising
    depth_img = cv2.medianBlur(depth_img, 3)

    # Normalize to 0-255 for visualization
    depth_normalized = cv2.normalize(
        depth_img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
    )

    return depth_img, depth_normalized