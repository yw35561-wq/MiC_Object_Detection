from src.detection import YOLOv12Detector
from src.depth_analysis import (
    DimensionCalculator,
    RoughnessEstimator,
    depth_img_read,
    preprocess_depth_img
)
import cv2
import argparse


def main(args):
    """
    Main function for MiC object detection and depth analysis
    :param args: Command line arguments
    """
    # Initialize core modules
    detector = YOLOv12Detector()
    dim_calc = DimensionCalculator()
    roughness_est = RoughnessEstimator()

    # Perform detection
    img, detections = detector.detect(args.image_path)

    # Process depth image
    depth_img = depth_img_read(args.depth_path)
    depth_img, _ = preprocess_depth_img(depth_img)

    # Print analysis results
    print("=" * 50)
    print("MiC Object Detection & Analysis Results")
    print("=" * 50)

    for i, det in enumerate(detections, 1):
        bbox = det["bbox"]
        cls_name = det["class_name"]
        conf = det["confidence"]

        # Calculate object dimensions
        width, height, depth = dim_calc.calculate_dimensions(depth_img, bbox)
        # Estimate surface roughness
        roughness = roughness_est.estimate_roughness(depth_img, bbox)

        print(f"Object {i}:")
        print(f"  Class: {cls_name} (Confidence: {conf:.2f})")
        print(f"  Dimensions: Width={width:.3f}m, Height={height:.3f}m, Depth={depth:.3f}m")
        print(f"  Surface Roughness: {roughness:.3f}")
        print("-" * 30)

    # Save result image
    img_with_boxes = detector.draw_detections(img, detections)
    cv2.imwrite(args.output_path, img_with_boxes)
    print(f"Result image saved to: {args.output_path}")


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="MiC Object Detection and Depth Analysis")
    parser.add_argument("--image_path", type=str, required=True, help="Path to RGB image file")
    parser.add_argument("--depth_path", type=str, required=True, help="Path to depth image file")
    parser.add_argument("--output_path", type=str, default="result.jpg", help="Path to save result image")

    args = parser.parse_args()
    main(args)