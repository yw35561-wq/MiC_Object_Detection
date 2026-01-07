import yaml
import os
from typing import Dict


def load_config(config_name: str) -> Dict:
    """
    Load configuration file from configs/ directory
    :param config_name: Name of config file (dataset/model/web_config)
    :return: Configuration dictionary
    """
    # Get project root path
    root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    config_path = os.path.join(root_path, "configs", f"{config_name}.yaml")

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        raise FileNotFoundError(f"Config file not found: {config_path}")
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML format in config file: {e}")