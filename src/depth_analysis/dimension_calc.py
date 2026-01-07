import numpy as np


class DimensionCalculator:
    def __init__(self, depth_scale: float = 0.001):
        """
        Initialize dimension calculator for depth image analysis
        :param depth_scale: Depth scale factor (convert pixel value to meters)
        """
        self.depth_scale = depth_scale

    def calculate_dimensions(self, depth_img: np.ndarray, bbox: list) -> tuple[float, float, float]:
        """
        Calculate physical dimensions of detected object from depth image
        :param depth_img: Depth image array
        :param bbox: Bounding box [x1, y1, x2, y2]
        :return: (width, height, depth) in meters
        """
        x1, y1, x2, y2 = bbox
        # Extract ROI from depth image
        roi_depth = depth_img[y1:y2, x1:x2]
        # Filter out invalid depth values (0 or negative)
        valid_depth = roi_depth[roi_depth > 0]

        if len(valid_depth) == 0:
            return 0.0, 0.0, 0.0

        # Calculate average depth (distance to camera)
        avg_depth = np.mean(valid_depth) * self.depth_scale

        # Convert pixel dimensions to physical dimensions (camera intrinsic parameters)
        fx = 600  # Focal length in x direction
        fy = 600  # Focal length in y direction

        # Calculate physical width and height
        pixel_width = x2 - x1
        real_width = (pixel_width * avg_depth) / fx

        pixel_height = y2 - y1
        real_height = (pixel_height * avg_depth) / fy

        return real_width, real_height, avg_depth