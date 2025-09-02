import shutil
from pathlib import Path
from typing import Dict
import os 
import logging
import yaml
from PIL import Image
import cv2

def prepare_processed_images(base_dir_path: str):
    """
    Prepare a 'processed' folder inside the given base image directory.
    If it does not exist, traverse all subfolders under base_dir_path,
    collect image files, and copy them into 'processed'.
    
    Args:
        base_dir_path (str): Path to the base directory (e.g., "../../../../data/.../01_raw_data")
    """
    base_dir = Path(base_dir_path)
    
    processed_dir = base_dir / "processed"    
    
    # Supported image extensions
    image_exts = {".jpg", ".jpeg", ".png"}

    if not processed_dir.exists():
        print(f"'{processed_dir}' does not exist. Creating folder and copying images...")

        # Create processed folder
        processed_dir.mkdir(parents=True, exist_ok=True)

        # Traverse all subdirectories and copy images
        for root, _, files in os.walk(base_dir):
            for file in files:
                if Path(file).suffix.lower() in image_exts:
                    src_path = Path(root) / file
                    dst_path = processed_dir / file

                    # Avoid overwriting files with the same name
                    counter = 1
                    while dst_path.exists():
                        dst_path = processed_dir / f"{Path(file).stem}_{counter}{Path(file).suffix}"
                        counter += 1

                    shutil.copy2(src_path, dst_path)
                    print(f"Copied: {src_path} → {dst_path}")

        print("✅ All images have been copied successfully.")
    else:
        print(f"'{processed_dir}' already exists.")

# Copy colmap data, including subdirectories
def copy_all(src: Path, dest: Path) -> None:
    for item in src.iterdir():
        dest_path = dest / item.name
        if item.is_dir():
            if dest_path.exists():
                shutil.rmtree(dest_path)  # Remove existing directories
            shutil.copytree(item, dest_path)
        else:
            if dest_path.exists():
                os.remove(dest_path)  # Remove existing files
            shutil.copy(item, dest_path)

# copy to input images to workspace
def copy_images(src_dir: Path, dest_dir: Path, downscale_factor: int = 1, prefix: str = "") -> None:
    
    # 이미지 확장자 리스트 (소문자로 설정)
    image_extensions = {".jpg", ".jpeg", ".png"}

    # Create destination directories
    if is_dir_empty(dest_dir):
        dest_dir.mkdir(parents=True, exist_ok=True)
    else:
        logging.warning("Destination directory is not empty")
                            
    # 파일 이름순으로 정렬된 이미지 리스트
    sorted_images = sorted(
        [item for item in src_dir.iterdir() if item.is_file() and item.suffix.lower() in image_extensions],
        key=lambda x: x.name.lower()
    )

     # 이미지 downscale 및 저장
    for frame_cnt, item in enumerate(sorted_images, start=1):
        # 원본 이미지 읽기 (OpenCV는 EXIF 못 읽음)
        img_pil = Image.open(item)
        exif_bytes = img_pil.info.get("exif", None)

        # 이미지 OpenCV로 불러와 다운스케일
        img_cv = cv2.imread(str(item))
        if img_cv is None:
            logging.warning(f"Cannot read image: {item}")
            continue

        new_width = img_cv.shape[1] // downscale_factor
        new_height = img_cv.shape[0] // downscale_factor
        resized_img = cv2.resize(img_cv, (new_width, new_height), interpolation=cv2.INTER_AREA)

        # 저장할 경로 및 이름
        new_name = f"{prefix}frame_{frame_cnt:05}{item.suffix.lower()}"
        dest_path = dest_dir / new_name

        # OpenCV → PIL (to save with EXIF)
        resized_pil = Image.fromarray(cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB))
        
        if exif_bytes:
            resized_pil.save(dest_path, exif=exif_bytes)
        else:
            logging.warning(f"No EXIF metadata found in {item.name}")
            resized_pil.save(dest_path)

# For correspondence search
def downscale_processing(image_dir: Path, factors=(2, 4, 8)) -> None:
    prefix = "new_"

    for factor in factors:
        out_dir = image_dir.parent / f"{prefix}images_{factor}"
        out_dir.mkdir(parents=True, exist_ok=True)

        copy_images(image_dir, out_dir, factor, prefix)        
        print(f"[✓] {prefix}images_{factor} saved to: {out_dir}")

def get_workspace_dir(setting_config: Dict) -> Path:
    workspace_dir = Path(setting_config['base_workspace_dir']) / setting_config['trial_num'] / setting_config['phase']
    return workspace_dir 

def get_image_dir(setting_config: Dict) -> Path:
    if setting_config['input_type']=="video":
        image_dir = Path(setting_config['input_dir']) / f"images_step-{setting_config['image_step']}_frame-start-{setting_config['image_start']}"
    else:
        image_dir = Path(setting_config['input_dir']) / "processed"
    return image_dir

# [correspondence search] copy to colmap workspace stage
def copy_colmap_init(src_dir1, src_dir2, dest_dir, colmap_src, prefix):
    
    # new image's name list
    image_list_file = dest_dir / "image-list.txt" 
    
    # Create destination directories
    dest_images_database_dir = dest_dir / "images" # day 1 + day 2
    dest_images_dir = dest_dir / "new_images" # day 2
    dest_colmap_dir = dest_dir / "colmap" 
    dest_images_dir.mkdir(parents=True, exist_ok=True)
    dest_colmap_dir.mkdir(parents=True, exist_ok=True)
    dest_images_database_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy images from first source directory(initial images)
    for item in src_dir1.iterdir():
        if item.is_file():            
            shutil.copy(item, dest_images_database_dir / item.name)

    # Copy and rename images from second source directory(new images)
    for item in src_dir2.iterdir():
        if item.is_file():
            new_name = f"{prefix}{item.name}"        
            dest_path1 = dest_images_database_dir / new_name
            dest_path2 = dest_images_dir / new_name

            shutil.copy(item, dest_path1)
            shutil.copy(item, dest_path2)            

    # Copy colmap data
    copy_all(colmap_src, dest_colmap_dir)

    # Write image list with newly come prefixed images
    with open(image_list_file, 'w') as f:
        for item in dest_images_dir.iterdir():
            if item.is_file() and item.name.startswith(prefix):
                f.write(f"{item.name}\n")

    downscale_processing(dest_images_dir)


# 5.0  copy to "/workspace/Data/240508_SNU-Duraemidam_ChangeDetection/ChangeDetection_workspace
def copy_colmap_endpoint(src_rendered_images_dir, src_new_images_dir, dest_dir):    
    
    # Create destination directories    
    dest_rendered_images_path = dest_dir / "rendered_images"
    dest_rendered_images_gt_path = dest_dir / "rendered_images_gt"
    dest_new_images_path = dest_dir / "new_images"
    
    dest_rendered_images_path.mkdir(parents=True, exist_ok=True)
    dest_rendered_images_gt_path.mkdir(parents=True, exist_ok=True)
    dest_new_images_path.mkdir(parents=True, exist_ok=True)

    # Source split
    src_rendered_images_path = src_rendered_images_dir/"rgb"
    src_rendered_images_gt_path = src_rendered_images_dir/"gt-rgb"
    
    # Copy data
    copy_all(src_rendered_images_path, dest_rendered_images_path)
    copy_all(src_rendered_images_gt_path, dest_rendered_images_gt_path)  
    copy_all(src_new_images_dir, dest_new_images_path)  

    
def is_dir_empty(dir_path):
    return not os.path.exists(dir_path) or len(os.listdir(dir_path)) == 0


def load_config(config_path: str) -> Dict:
    """Load and return the configuration from a YAML file."""
    logging.info("Loading configuration file...")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def validate_config(config: Dict) -> None:
    """Validate the configuration."""
    required_keys = ['init_settings', 'correspondence_settings', 'nerf_settings']
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required key in configuration: {key}")
            


