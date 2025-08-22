import subprocess
import yaml
import logging
import argparse
import os
import time

from typing import Dict, Tuple
from pathlib import Path
from R3DR.utils import dir_utils

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    logging.info("Parsing command line arguments...")
    parser = argparse.ArgumentParser(description="Specify the configuration file for NeRF training.")
    parser.add_argument('--config', type=str, required=True, help="Path to the configuration file")
    return parser.parse_args()


def render_initial_images(config: Dict) -> None:
    
    camera_render_config = config['ns_render_settings']
    render_command = [
        "ns-render", "camera-path",
        "--output_format", camera_render_config['output_format'],
        "--load-config", camera_render_config['load_config'],
        "--camera-path-filename", camera_render_config['camera_path_filename'],
        "--output-path", camera_render_config['output_path']
    ]

    # 명령어 실행
    subprocess.run(render_command, check=True)


def main() -> None:
    start_time = time.time()
    args = parse_arguments()
    config = dir_utils.load_config(args.config)
    dir_utils.validate_config(config)    
    
    # Render the initial images with new camera position
    render_initial_images(config)

    end_time = time.time()
    logging.info(f"Total processing time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.exception("Unexpected error occurred during program execution")