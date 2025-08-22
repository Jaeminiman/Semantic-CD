import subprocess
import argparse
import logging
import os
import time
from pathlib import Path
from typing import Dict, Tuple, Optional, List
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


def ns_process_bool_parser(option: str) -> List[str]:
    
    option_list = {
        'verbose': '--verbose',
        'estimate_affine_shape': '--estimate-affine-shape',
        'domain_size_pooling': "--domain-size-pooling",
        'guided_matching': "--guided-matching"
    }

    # Explicit check for unknown options and raise a clear exception
    if option not in option_list:
        raise ValueError(f"Unknown boolean option: '{option}'. Available options are: {', '.join(option_list.keys())}")
    
    return [option_list[option]]

# Function to parse other options and generate the command line arguments
def ns_process_command_parser(image_dir: Path, ns_processed_dir: Path, nerf_config: Dict) -> List[str]:
    
    ns_process_config = nerf_config.get('ns_process_options', {})
        
    # Check if the 'ns_process_options' key exists in the config
    if not ns_process_config:
        raise ValueError("Missing 'ns_process_options' in nerf_config.")

    # Prepare the base command to run the process
    ns_process_command = [
        "ns-process-data", "images",
        "--data", str(image_dir),
        "--output-dir", str(ns_processed_dir)
    ]

    for k, v in ns_process_config.items():
        for key, value in v.items():
            # If the value is a boolean, add the corresponding boolean flag
            if isinstance(value, bool):
                if(value):
                    ns_process_command.extend(ns_process_bool_parser(key))
            else:                
                new_key = key.replace("_", "-")
                ns_process_command.append(f"--{new_key}={value}")

    return ns_process_command  

def run_nerf_process(image_dir: Path, workspace_dir: Path, config: Dict, is_gps: int) -> None:
    """Run NeRF data processing and training."""
    ns_processed_dir = workspace_dir / "ns_processed"  

    correspondence_config = config["correspondence_settings"]
    nerf_config = config['nerf_settings']

    logging.info("Starting NeRF data processing...")     

    try:        
        

        # Task 1. Data processing with COLMAP
        start_time = time.time()
        if dir_utils.is_dir_empty(ns_processed_dir):            
 
            ns_process_command = ns_process_command_parser(image_dir, ns_processed_dir, nerf_config)
            
            logging.info(f"Running the command: {' '.join(ns_process_command)}")
            subprocess.run(ns_process_command , check=True)
            
        else:
            text = "Output directory is not empty. Data processing with COLMAP is skipped."
            logging.info(f"\033[91m{text}\033[0m" ) # red text    

        
        colmap_utils.generate_transforms(ns_processed_dir, correspondence_config)        

        # Geo Registration(GPS version)
        colmap_model_path = str(nerf_config['geo_registration']['colmap_model_path'])
        if is_gps and dir_utils.is_dir_empty(ns_processed_dir / colmap_model_path) :
            
            logging.info("Geo-resgistration in COLMAP using GPS data...")
            ransac_thr = float(nerf_config['geo_registration']['align_ransac_max_error'])

            colmap_utils.geo_registration(image_dir, ns_processed_dir, colmap_model_path, ransac_thr)

            # Save transforms.json in ecef coordinate -> transforms.json은 그냥 no gps와 맞추기.(정밀도 문제)
            # process_command.extend(["--skip_image_processing"])
            # process_command.extend(["--skip_colmap"])
            # process_command.extend(["--colmap_model_path", colmap_model_path])
            # subprocess.run(process_command, check=True)            

        end_time = time.time()
        logging.info(f"ns-process-data(colmap) time: {end_time - start_time:.2f} seconds")

        # Task 2. Transform the coordinate from t2 local world to t1 local world(world integration)       
        if is_gps:       
            init_config = config['init_settings']
            t1_workspace_dir = dir_utils.get_workspace_dir(init_config)
            t2_workspace_dir = dir_utils.get_workspace_dir(correspondence_config)
            sim3_ba_path = os.path.join(t1_workspace_dir, "ns_processed/colmap/sparse/geo-registration/ecef/sim3_transform.json")
            sim3_bc_path = os.path.join(t2_workspace_dir, "ns_processed/colmap/sparse/geo-registration/ecef/sim3_transform.json")
            input_transforms_path = os.path.join(t2_workspace_dir, "ns_processed/transforms.json")
            output_transforms_path = os.path.join(t2_workspace_dir, "ns_processed/transforms_t1world.json")

            colmap_utils.convert_poses_t2_to_t1_with_gps(sim3_ba_path, sim3_bc_path, input_transforms_path, output_transforms_path)

    except subprocess.CalledProcessError as e:
        logging.error(f"Error occurred during subprocess execution: {e}")
        raise

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

    # Task 1: Process input video (It would be skipped when the new/images folder exists)
    image_dir = process_input_data(correspondence_config)           

    # Task 2: Build workspace and Run COLMAP and NeRF process
    is_gps = correspondence_config['is_gps']    

    run_nerf_process(image_dir, workspace_dir, config, is_gps)

    end_time = time.time()
    logging.info(f"Total processing time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.exception("Unexpected error occurred during program execution")
