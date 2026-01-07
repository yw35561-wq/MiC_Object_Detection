import numpy as np
import cv2


class RoughnessEstimator:
    def estimate_roughness(self, depth_img: np.ndarray, bbox: list, window_size: int = 5) -> float:
        """
        Estimate surface roughness from depth image texture
        :param depth_img: Depth image array
        :param bbox: Bounding box [x1, y1, x2, y2]
        :param window_size: Sliding window size for local variance calculation
        :return: Roughness value (higher = rougher)
        """
        x1, y1, x2, y2 = bbox
        # Extract ROI from depth image
        roi_depth = depth_img[y1:y2, x1:x2]
        # Filter out invalid depth values
        roi_depth = roi_depth[roi_depth > 0]

        if len(roi_depth) == 0:
            return 0.0

        # Reshape to 2D array
        roi_depth = roi_depth.reshape((y2 - y1, x2 - x1))

        # Calculate local variance (core roughness metric)
        kernel = np.ones((window_size, window_size), np.float32) / (window_size ** 2)
        mean = cv2.filter2D(roi_depth, -1, kernel)
        var = cv2.filter2D((roi_depth - mean) ** 2, -1, kernel)

        # Average variance as roughness value
        roughness = np.mean(var)
        return roughness