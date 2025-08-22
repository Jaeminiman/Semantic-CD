import argparse
import logging
import os

import time
from pathlib import Path
from typing import Dict, Tuple, Optional
from R3DR.utils import dir_utils
from R3DR.utils import colmap_utils

from R3DR.utils.input_processing_utils import process_input_data
import cv2
import yaml
import shutil

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def parse_arguments() -> argparse.Namespace:
    logging.info("Parsing command line arguments...")
    parser = argparse.ArgumentParser(description="Specify configuration file for NeRF correspondence processing")
    parser.add_argument('--config', type=str, required=True, help="Path to configuration file")
    return parser.parse_args()


# Task 2
def run_incremental_colmap_process(new_image_dir: Path ,config: Dict) -> Path:
    
    init_config = config['init_settings']
    init_workspace_dir = dir_utils.get_workspace_dir(init_config)
    init_image_dir = dir_utils.get_image_dir(init_config)
    
    correspondence_config = config['correspondence_settings']    
    new_workspace_dir = dir_utils.get_workspace_dir(correspondence_config)
    new_ns_processed_dir = new_workspace_dir / "ns_processed"
    
    recon_dir = new_ns_processed_dir / "colmap/sparse/0"

    # Skip
    if not dir_utils.is_dir_empty(new_ns_processed_dir):
        logging.warning("ns_processed directory already exists. Skipping colmap process.")
        return new_ns_processed_dir


    # Task 2.1: Copy COLMAP within workspace        
    logging.info(f"Copy initial COLMAP to new workspace")

    
    dir_utils.copy_colmap_init(
        src_dir1=init_image_dir,
        src_dir2=new_image_dir,
        dest_dir=new_ns_processed_dir ,
        colmap_src= init_workspace_dir/"ns_processed/colmap",
        prefix=correspondence_config['prefix']
    )

    logging.info(f"COLMAP process starts")

    # Task 2.2: COLMAP Feature Extractor
    colmap_utils.run_feature_extractor(new_ns_processed_dir , correspondence_config)

    # Task 2.3: COLMAP Vocab Tree Matcher
    colmap_utils.run_vocab_tree_matcher(new_ns_processed_dir , correspondence_config)

    # Task 2.4: COLMAP Image Registrator
    colmap_utils.run_image_registrator(new_ns_processed_dir, recon_dir)
    
    # Task 2.5: COLMAP Bundle Adjuster
    colmap_utils.run_bundle_adjuster(new_ns_processed_dir, recon_dir)

    return new_ns_processed_dir 


def main() -> None:
    start_time = time.time()
    args = parse_arguments()
    config = dir_utils.load_config(args.config)
    dir_utils.validate_config(config)           
    
    correspondence_config = config["correspondence_settings"]

    workspace_dir = dir_utils.get_workspace_dir(correspondence_config)

    # Task 0. Copy config into workspace
    config_dir = workspace_dir / "config"
    config_dir.mkdir(parents=True, exist_ok=True)  # 디렉토리가 없으면 생성
    shutil.copy2(args.config, config_dir / "config.yaml")  # Copy the config file to the workspace

    # Task 1: Process input video (It would be skipped when the new/raw_images folder exists)
    new_image_dir = process_input_data(correspondence_config)           

    # Task 2: Run COLMAP process for incremental pose estimation of new images 
    new_ns_processed_dir = run_incremental_colmap_process(new_image_dir, config)
    
    # Task 3: Generate transforms.json from COLMAP result
    colmap_utils.generate_transforms(new_ns_processed_dir, correspondence_config)

    end_time = time.time()
    logging.info(f"Total processing time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.exception("Unexpected error occurred during program execution")
