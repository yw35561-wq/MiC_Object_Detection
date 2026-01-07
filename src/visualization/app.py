from flask import Flask
from src.utils.path_utils import load_config
import os

# Initialize Flask application
app = Flask(__name__)

# Load web configuration
web_config = load_config("web_config")

# Configure upload and result directories
app.config["UPLOAD_FOLDER"] = web_config["upload_folder"]
app.config["RESULT_FOLDER"] = web_config["result_folder"]

# Create directories if not exist
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs(app.config["RESULT_FOLDER"], exist_ok=True)

# Import routes (must be after app initialization)
from src.visualization import routes

if __name__ == "__main__":
    # Run web server
    app.run(
        host=web_config["host"],
        port=web_config["port"],
        debug=True
    )