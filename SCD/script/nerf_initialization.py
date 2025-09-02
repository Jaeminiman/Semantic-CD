# This script initializes and runs the NeRF (Neural Radiance Fields) process
# It handles video processing, image extraction, and NeRF training

import argparse
import logging
import os
import subprocess
import time
from pathlib import Path
from typing import Dict, Tuple, List

import yaml
import shutil

from SCD.utils.input_processing_utils import process_input_data
from SCD.utils import dir_utils
from SCD.utils import colmap_utils

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    logging.info("Parsing command line arguments...")
    parser = argparse.ArgumentParser(description="Specify the configuration file for NeRF training.")
    parser.add_argument('--config', type=str, required=True, help="Path to the configuration file")
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

def ns_train_command_parser(ns_processed_dir: Path, nerf_output_dir: Path, nerf_config: Dict) -> List[str]:   
    
    ns_train_config = nerf_config.get('ns_train_options', {})
        
    # Check if the 'ns_train_options' key exists in the config
    if not ns_train_config:
        raise ValueError("Missing 'ns_train_options' in nerf_config.")


    ns_train_command = [
        "ns-train", ns_train_config['nerf_model'],
        "--data", str(ns_processed_dir),
        "--output-dir", str(nerf_output_dir), 
    ]

    # parse train options
    for key, value in ns_train_config['train_options'].items():
        ns_train_command.extend([f"--{key}", str(value)])
    
    # parse dataparser options
    dataparser = ns_train_config['dataparser_model']
    ns_train_command.append(dataparser)

    for key, value in ns_train_config['dataparser_options'].items():
            ns_train_command.extend([f"--{key}", str(value)])

    if(dataparser == "colmap"):    
        for key, value in ns_train_config['dataparser_colmap_options'].items():
            ns_train_command.extend([f"--{key}", str(value)])

    return ns_train_command




def run_nerf_process(image_dir: Path, workspace_dir: Path, config: Dict, is_gps: int) -> None:
    """Run NeRF data processing and training."""
    ns_processed_dir = workspace_dir / "ns_processed"
    nerf_output_dir = workspace_dir / "nerf_output"    

    init_config = config['init_settings']
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

        
        colmap_utils.generate_transforms(ns_processed_dir, init_config)        

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

        # Task 2. Training initial Nerf              
        ns_train_command = ns_train_command_parser(ns_processed_dir, nerf_output_dir, nerf_config)
        logging.info(f"Running the command: {' '.join(ns_train_command)}")
        
        subprocess.run(ns_train_command, check=True)        
        
        logging.info("NeRF training completed.")

    except subprocess.CalledProcessError as e:
        logging.error(f"Error occurred during subprocess execution: {e}")
        raise
        

def main() -> None:
    """Main function to orchestrate the NeRF initialization process."""
    start_time = time.time()
    args = parse_arguments()
    config = dir_utils.load_config(args.config)
    dir_utils.validate_config(config)
    
    init_config = config['init_settings']
    nerf_config = config['nerf_settings']
    workspace_dir = dir_utils.get_workspace_dir(init_config)

    # Task 0. Copy config into workspace
    config_dir = workspace_dir / "config"
    config_dir.mkdir(parents=True, exist_ok=True)  # 디렉토리가 없으면 생성
    shutil.copy2(args.config, config_dir / "config.yaml")  # Copy the config file to the workspace

    # Task 1: Process input data
    end_time = time.time()
    image_dir = process_input_data(init_config)       
    
    logging.info(f"video pre-processing time: {end_time - start_time:.2f} seconds")

    # Task 2: Build workspace and Run COLMAP and NeRF process
    is_gps = init_config['is_gps']    

    
    run_nerf_process(image_dir, workspace_dir, config, is_gps)
    
    

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.exception("Unexpected error occurred during program execution")
