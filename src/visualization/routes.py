from flask import render_template, request, send_from_directory
from src.visualization.app import app
from src.detection import YOLOv12Detector
from src.depth_analysis import (
    DimensionCalculator,
    RoughnessEstimator,
    depth_img_read,
    preprocess_depth_img
)
import cv2
import uuid
import os

# Initialize core modules
detector = YOLOv12Detector()
dim_calc = DimensionCalculator()
roughness_est = RoughnessEstimator()


@app.route('/')
def index():
    """Main page - file upload interface"""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Process uploaded files and perform detection/analysis"""
    # Check if files are uploaded
    if 'image' not in request.files or 'depth' not in request.files:
        return "Please upload both RGB image and depth image", 400

    img_file = request.files['image']
    depth_file = request.files['depth']

    # Check if filenames are empty
    if img_file.filename == '' or depth_file.filename == '':
        return "Empty filename is not allowed", 400

    # Save uploaded files with unique IDs to avoid overwriting
    img_name = f"{uuid.uuid4()}_{img_file.filename}"
    depth_name = f"{uuid.uuid4()}_{depth_file.filename}"

    img_path = os.path.join(app.config["UPLOAD_FOLDER"], img_name)
    depth_path = os.path.join(app.config["UPLOAD_FOLDER"], depth_name)

    img_file.save(img_path)
    depth_file.save(depth_path)

    # Perform object detection
    img, detections = detector.detect(img_path)

    # Process depth image
    depth_img = depth_img_read(depth_path)
    depth_img, depth_normalized = preprocess_depth_img(depth_img)

    # Analyze each detected object
    results = []
    for det in detections:
        bbox = det["bbox"]
        # Calculate dimensions
        width, height, depth = dim_calc.calculate_dimensions(depth_img, bbox)
        # Estimate roughness
        roughness = roughness_est.estimate_roughness(depth_img, bbox)

        results.append({
            "class_name": det["class_name"],
            "confidence": round(det["confidence"], 2),
            "width": round(width, 3),
            "height": round(height, 3),
            "depth": round(depth, 3),
            "roughness": round(roughness, 3)
        })

    # Save detection result image
    img_with_boxes = detector.draw_detections(img, detections)
    result_img_name = f"{uuid.uuid4()}_result.jpg"
    result_img_path = os.path.join(app.config["RESULT_FOLDER"], result_img_name)
    cv2.imwrite(result_img_path, img_with_boxes)

    # Save depth visualization image
    depth_result_name = f"{uuid.uuid4()}_depth.jpg"
    depth_result_path = os.path.join(app.config["RESULT_FOLDER"], depth_result_name)
    cv2.imwrite(depth_result_path, depth_normalized)

    # Render result page
    return render_template(
        'result.html',
        result_img=result_img_name,
        depth_img=depth_result_name,
        results=results
    )


@app.route('/results/<filename>')
def get_result(filename):
    """Serve result images to web page"""
    return send_from_directory(app.config["RESULT_FOLDER"], filename)