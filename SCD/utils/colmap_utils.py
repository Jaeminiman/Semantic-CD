from pathlib import Path
from typing import Optional, Dict, Any, Union, List
import subprocess
import logging
import numpy as np
import torch
import json
from enum import Enum
import exifread
import os 
import shutil

from .colmap_parsing_utils import (
    qvec2rotmat,
    read_cameras_binary,
    read_images_binary,
    read_points3D_binary,
    read_points3D_text,
)

from .transform_utils import (
    load_sim3,
    sim3_matrix,
    inverse_sim3
)

#########################################################################################
# Source from nerfstudio's colmap_utils.py
#########################################################################################
class CameraModel(Enum):
    """Enum for camera types."""

    OPENCV = "OPENCV"
    OPENCV_FISHEYE = "OPENCV_FISHEYE"
    EQUIRECTANGULAR = "EQUIRECTANGULAR"
    PINHOLE = "PINHOLE"
    SIMPLE_PINHOLE = "SIMPLE_PINHOLE"


def parse_colmap_camera_params(camera) -> Dict[str, Any]:
    """
    Parses all currently supported COLMAP cameras into the transforms.json metadata

    Args:
        camera: COLMAP camera
    Returns:
        transforms.json metadata containing camera's intrinsics and distortion parameters

    """
    out: Dict[str, Any] = {
        "w": camera.width,
        "h": camera.height,
    }

    # Parameters match https://github.com/colmap/colmap/blob/dev/src/base/camera_models.h
    camera_params = camera.params
    if camera.model == "SIMPLE_PINHOLE":
        # du = 0
        # dv = 0
        out["fl_x"] = float(camera_params[0])
        out["fl_y"] = float(camera_params[0])
        out["cx"] = float(camera_params[1])
        out["cy"] = float(camera_params[2])
        out["k1"] = 0.0
        out["k2"] = 0.0
        out["p1"] = 0.0
        out["p2"] = 0.0
        camera_model = CameraModel.OPENCV
    elif camera.model == "PINHOLE":
        # f, cx, cy, k

        # du = 0
        # dv = 0
        out["fl_x"] = float(camera_params[0])
        out["fl_y"] = float(camera_params[1])
        out["cx"] = float(camera_params[2])
        out["cy"] = float(camera_params[3])
        out["k1"] = 0.0
        out["k2"] = 0.0
        out["p1"] = 0.0
        out["p2"] = 0.0
        camera_model = CameraModel.OPENCV
    elif camera.model == "SIMPLE_RADIAL":
        # f, cx, cy, k

        # r2 = u**2 + v**2;
        # radial = k * r2
        # du = u * radial
        # dv = u * radial
        out["fl_x"] = float(camera_params[0])
        out["fl_y"] = float(camera_params[0])
        out["cx"] = float(camera_params[1])
        out["cy"] = float(camera_params[2])
        out["k1"] = float(camera_params[3])
        out["k2"] = 0.0
        out["p1"] = 0.0
        out["p2"] = 0.0
        camera_model = CameraModel.OPENCV
    elif camera.model == "RADIAL":
        # f, cx, cy, k1, k2

        # r2 = u**2 + v**2;
        # radial = k1 * r2 + k2 * r2 ** 2
        # du = u * radial
        # dv = v * radial
        out["fl_x"] = float(camera_params[0])
        out["fl_y"] = float(camera_params[0])
        out["cx"] = float(camera_params[1])
        out["cy"] = float(camera_params[2])
        out["k1"] = float(camera_params[3])
        out["k2"] = float(camera_params[4])
        out["p1"] = 0.0
        out["p2"] = 0.0
        camera_model = CameraModel.OPENCV
    elif camera.model == "OPENCV":
        # fx, fy, cx, cy, k1, k2, p1, p2

        # uv = u * v;
        # r2 = u**2 + v**2
        # radial = k1 * r2 + k2 * r2 ** 2
        # du = u * radial + 2 * p1 * u*v + p2 * (r2 + 2 * u**2)
        # dv = v * radial + 2 * p2 * u*v + p1 * (r2 + 2 * v**2)
        out["fl_x"] = float(camera_params[0])
        out["fl_y"] = float(camera_params[1])
        out["cx"] = float(camera_params[2])
        out["cy"] = float(camera_params[3])
        out["k1"] = float(camera_params[4])
        out["k2"] = float(camera_params[5])
        out["p1"] = float(camera_params[6])
        out["p2"] = float(camera_params[7])
        camera_model = CameraModel.OPENCV
    elif camera.model == "OPENCV_FISHEYE":
        # fx, fy, cx, cy, k1, k2, k3, k4

        # r = sqrt(u**2 + v**2)

        # if r > eps:
        #    theta = atan(r)
        #    theta2 = theta ** 2
        #    theta4 = theta2 ** 2
        #    theta6 = theta4 * theta2
        #    theta8 = theta4 ** 2
        #    thetad = theta * (1 + k1 * theta2 + k2 * theta4 + k3 * theta6 + k4 * theta8)
        #    du = u * thetad / r - u;
        #    dv = v * thetad / r - v;
        # else:
        #    du = dv = 0
        out["fl_x"] = float(camera_params[0])
        out["fl_y"] = float(camera_params[1])
        out["cx"] = float(camera_params[2])
        out["cy"] = float(camera_params[3])
        out["k1"] = float(camera_params[4])
        out["k2"] = float(camera_params[5])
        out["k3"] = float(camera_params[6])
        out["k4"] = float(camera_params[7])
        camera_model = CameraModel.OPENCV_FISHEYE
    elif camera.model == "FULL_OPENCV":
        # fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6

        # u2 = u ** 2
        # uv = u * v
        # v2 = v ** 2
        # r2 = u2 + v2
        # r4 = r2 * r2
        # r6 = r4 * r2
        # radial = (1 + k1 * r2 + k2 * r4 + k3 * r6) /
        #          (1 + k4 * r2 + k5 * r4 + k6 * r6)
        # du = u * radial + 2 * p1 * uv + p2 * (r2 + 2 * u2) - u
        # dv = v * radial + 2 * p2 * uv + p1 * (r2 + 2 * v2) - v
        out["fl_x"] = float(camera_params[0])
        out["fl_y"] = float(camera_params[1])
        out["cx"] = float(camera_params[2])
        out["cy"] = float(camera_params[3])
        out["k1"] = float(camera_params[4])
        out["k2"] = float(camera_params[5])
        out["p1"] = float(camera_params[6])
        out["p2"] = float(camera_params[7])
        out["k3"] = float(camera_params[8])
        out["k4"] = float(camera_params[9])
        out["k5"] = float(camera_params[10])
        out["k6"] = float(camera_params[11])
        raise NotImplementedError(f"{camera.model} camera model is not supported yet!")
    elif camera.model == "FOV":
        # fx, fy, cx, cy, omega
        out["fl_x"] = float(camera_params[0])
        out["fl_y"] = float(camera_params[1])
        out["cx"] = float(camera_params[2])
        out["cy"] = float(camera_params[3])
        out["omega"] = float(camera_params[4])
        raise NotImplementedError(f"{camera.model} camera model is not supported yet!")
    elif camera.model == "SIMPLE_RADIAL_FISHEYE":
        # f, cx, cy, k

        # r = sqrt(u ** 2 + v ** 2)
        # if r > eps:
        #     theta = atan(r)
        #     theta2 = theta ** 2
        #     thetad = theta * (1 + k * theta2)
        #     du = u * thetad / r - u;
        #     dv = v * thetad / r - v;
        # else:
        #     du = dv = 0
        out["fl_x"] = float(camera_params[0])
        out["fl_y"] = float(camera_params[0])
        out["cx"] = float(camera_params[1])
        out["cy"] = float(camera_params[2])
        out["k1"] = float(camera_params[3])
        out["k2"] = 0.0
        out["k3"] = 0.0
        out["k4"] = 0.0
        camera_model = CameraModel.OPENCV_FISHEYE
    elif camera.model == "RADIAL_FISHEYE":
        # f, cx, cy, k1, k2

        # r = sqrt(u ** 2 + v ** 2)
        # if r > eps:
        #     theta = atan(r)
        #     theta2 = theta ** 2
        #     theta4 = theta2 ** 2
        #     thetad = theta * (1 + k * theta2)
        #     thetad = theta * (1 + k1 * theta2 + k2 * theta4)
        #     du = u * thetad / r - u;
        #     dv = v * thetad / r - v;
        # else:
        #     du = dv = 0
        out["fl_x"] = float(camera_params[0])
        out["fl_y"] = float(camera_params[0])
        out["cx"] = float(camera_params[1])
        out["cy"] = float(camera_params[2])
        out["k1"] = float(camera_params[3])
        out["k2"] = float(camera_params[4])
        out["k3"] = 0
        out["k4"] = 0
        camera_model = CameraModel.OPENCV_FISHEYE
    else:
        # THIN_PRISM_FISHEYE not supported!
        raise NotImplementedError(f"{camera.model} camera model is not supported yet!")

    out["camera_model"] = camera_model.value
    return out

def create_ply_from_colmap(
    filename: str, recon_dir: Path, output_dir: Path, applied_transform: Union[torch.Tensor, None]
) -> None:
    """Writes a ply file from colmap.

    Args:
        filename: file name for .ply
        recon_dir: Directory to grab colmap points
        output_dir: Directory to output .ply
    """
    if (recon_dir / "points3D.bin").exists():
        colmap_points = read_points3D_binary(recon_dir / "points3D.bin")
    elif (recon_dir / "points3D.txt").exists():
        colmap_points = read_points3D_text(recon_dir / "points3D.txt")
    else:
        raise ValueError(f"Could not find points3D.txt or points3D.bin in {recon_dir}")

    # Load point Positions
    points3D = torch.from_numpy(np.array([p.xyz for p in colmap_points.values()], dtype=np.float32))
    if applied_transform is not None:
        assert applied_transform.shape == (3, 4)
        points3D = torch.einsum("ij,bj->bi", applied_transform[:3, :3], points3D) + applied_transform[:3, 3]

    # Load point colours
    points3D_rgb = torch.from_numpy(np.array([p.rgb for p in colmap_points.values()], dtype=np.uint8))

    # write ply
    with open(output_dir / filename, "w") as f:
        # Header
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points3D)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uint8 red\n")
        f.write("property uint8 green\n")
        f.write("property uint8 blue\n")
        f.write("end_header\n")

        for coord, color in zip(points3D, points3D_rgb):
            x, y, z = coord
            r, g, b = color
            f.write(f"{x:8f} {y:8f} {z:8f} {r} {g} {b}\n")


###############################################################################################################
# Customize nerfstudio's colmap_utils for Change Detection
###############################################################################################################
def rename_best_recon_to_zero_safe(sparse_dir: Path, best_recon_dir: Path):
    target_dir = sparse_dir / "0"

    if best_recon_dir == target_dir:
        logging.info("Best reconstruction is already in '0'. No renaming needed.")
        return

    # 기존 sparse/0이 있으면 삭제
    if target_dir.exists():
        logging.warning(f"'0' folder exists. Deleting it.")
        shutil.rmtree(target_dir)

    # best_recon_dir → sparse/0
    shutil.move(str(best_recon_dir), str(target_dir))
    logging.info(f"Renamed best reconstruction folder to: {target_dir}")

def generate_transforms(
    ns_processed_dir: Path,
    setting_config: Dict
    ) -> None:
    sparse_dir = ns_processed_dir / Path("colmap/sparse")    
    target_dir = ns_processed_dir / Path("colmap/sparse/0")

    image_dir = ns_processed_dir / "images"
    output_dir = ns_processed_dir
    
    prefix = setting_config["prefix"]
    
    ext = setting_config["image_ext"]
    image_rename_map = generate_image_rename_map(image_dir, prefix, ext)

    best_recon_dir = None
    best_matched_frames = -1

    # sparse 하위 폴더 순회
    for recon_subdir in sorted(sparse_dir.iterdir()):
        if not recon_subdir.is_dir():
            continue
        if not (recon_subdir / "cameras.bin").exists():
            continue

        logging.info(f"Checking COLMAP result in: {recon_subdir}")

        # 테스트로 저장 경로를 별도로 설정하여 덮어쓰지 않도록
        matched_frames = colmap_to_json(
            recon_dir=recon_subdir,
            output_dir=output_dir,  # 임시 저장이라도 하려면 복사해둘 필요 있음
            image_id_to_depth_path=None,
            camera_mask_path=None,
            image_rename_map=image_rename_map,
            use_single_camera_mode=True,
            dry_run=True  # 실제 저장하지 않고, frame 개수만 계산한다고 가정
        )

        if matched_frames > best_matched_frames:
            best_matched_frames = matched_frames
            best_recon_dir = recon_subdir
    
    if best_recon_dir:
        text = f"Best reconstruction: {best_recon_dir} with {best_matched_frames} frames"
        logging.info(f"\033[91m{text}\033[0m" ) # red text            

        rename_best_recon_to_zero_safe(sparse_dir, best_recon_dir)

        # 실제 저장
        save_transforms(
            recon_dir=target_dir,
            output_dir=output_dir,
            image_rename_map=image_rename_map
        )        
    else:
        logging.warning("No valid COLMAP results found in any subfolder of sparse/")        
    

def save_transforms(
        recon_dir: Path,
        output_dir: Path,
        image_id_to_depth_path: Optional[Dict[int, Path]] = None,
        camera_mask_path: Optional[Path] = None,
        image_rename_map: Optional[Dict[str, str]] = None,
    ) -> None:
        """Save colmap transforms into the output folder

        Args:
            image_id_to_depth_path: When including sfm-based depth, embed these depth file paths in the exported json
            image_rename_map: Use these image names instead of the names embedded in the COLMAP db
        """
        
        if (recon_dir / "cameras.bin").exists():
            logging.info("Saving results to transforms.json")

            num_matched_frames = colmap_to_json(
                recon_dir= recon_dir ,
                output_dir= output_dir,
                image_id_to_depth_path=image_id_to_depth_path,
                camera_mask_path=camera_mask_path,
                image_rename_map=image_rename_map,
                use_single_camera_mode=True,
            )
            logging.info(f"Colmap matched {num_matched_frames} images")

        else:
            logging.warning(
                "Could not find existing COLMAP results. " "Not generating transforms.json"
            )
        

def colmap_to_json(
    recon_dir: Path,
    output_dir: Path,
    camera_mask_path: Optional[Path] = None,
    image_id_to_depth_path: Optional[Dict[int, Path]] = None,
    image_rename_map: Optional[Dict[str, str]] = None,
    ply_filename="sparse_pc.ply",
    keep_original_world_coordinate: bool = False,
    use_single_camera_mode: bool = True,
    dry_run: bool = False 
) -> int:
    """Converts COLMAP's cameras.bin and images.bin to a JSON file.

    Args:
        recon_dir: Path to the reconstruction directory, e.g. "sparse/0"
        output_dir: Path to the output directory.
        camera_model: Camera model used.
        camera_mask_path: Path to the camera mask.
        image_id_to_depth_path: When including sfm-based depth, embed these depth file paths in the exported json
        image_rename_map: Use these image names instead of the names embedded in the COLMAP db
        keep_original_world_coordinate: If True, no extra transform will be applied to world coordinate.
                    Colmap optimized world often have y direction of the first camera pointing towards down direction,
                    while nerfstudio world set z direction to be up direction for viewer.
        dry_run: If true, do not save transforms.json
    Returns:
        The number of registered images.
    """

    # TODO(1480) use pycolmap
    # recon = pycolmap.Reconstruction(recon_dir)
    # cam_id_to_camera = recon.cameras
    # im_id_to_image = recon.images
    cam_id_to_camera = read_cameras_binary(recon_dir / "cameras.bin")
    im_id_to_image = read_images_binary(recon_dir / "images.bin")
    if set(cam_id_to_camera.keys()) != {1}:
        logging.warning(f"More than one camera is found in {recon_dir}")
        print(cam_id_to_camera)
        use_single_camera_mode = False  # update bool: one camera per frame
        out = {}  # out = {"camera_model": parse_colmap_camera_params(cam_id_to_camera[1])["camera_model"]}
    else:  # one camera for all frames
        out = parse_colmap_camera_params(cam_id_to_camera[1])

    frames = []
    for im_id, im_data in im_id_to_image.items():
        # NB: COLMAP uses Eigen / scalar-first quaternions
        # * https://colmap.github.io/format.html
        # * https://github.com/colmap/colmap/blob/bf3e19140f491c3042bfd85b7192ef7d249808ec/src/base/pose.cc#L75
        # the `rotation_matrix()` handles that format for us.

        # TODO(1480) BEGIN use pycolmap API
        # rotation = im_data.rotation_matrix()
        rotation = qvec2rotmat(im_data.qvec)

        translation = im_data.tvec.reshape(3, 1)
        w2c = np.concatenate([rotation, translation], 1)
        w2c = np.concatenate([w2c, np.array([[0, 0, 0, 1]])], 0)
        c2w = np.linalg.inv(w2c)

        # Convert from COLMAP's camera coordinate system (OpenCV) to ours (OpenGL)
        c2w[0:3, 1:3] *= -1
        if not keep_original_world_coordinate:
            c2w = c2w[np.array([0, 2, 1, 3]), :]
            c2w[2, :] *= -1

        name = im_data.name
        if image_rename_map is not None:
            if name in image_rename_map:
                name = image_rename_map[name]
            else:                
                continue

        name = Path(f"./images/{name}")

        frame = {
            "file_path": name.as_posix(),
            "transform_matrix": c2w.tolist(),
            "colmap_im_id": im_id,
        }
        if camera_mask_path is not None:
            frame["mask_path"] = camera_mask_path.relative_to(camera_mask_path.parent.parent).as_posix()
        if image_id_to_depth_path is not None:
            depth_path = image_id_to_depth_path[im_id]
            frame["depth_file_path"] = str(depth_path.relative_to(depth_path.parent.parent))

        if not use_single_camera_mode:  # add the camera parameters for this frame
            frame.update(parse_colmap_camera_params(cam_id_to_camera[im_data.camera_id]))

        frames.append(frame)

    out["frames"] = frames

    applied_transform = None
    if not keep_original_world_coordinate:
        applied_transform = np.eye(4)[:3, :]
        applied_transform = applied_transform[np.array([0, 2, 1]), :]
        applied_transform[2, :] *= -1
        out["applied_transform"] = applied_transform.tolist()

    # create ply from colmap
    assert ply_filename.endswith(".ply"), f"ply_filename: {ply_filename} does not end with '.ply'"
    create_ply_from_colmap(
        ply_filename,
        recon_dir,
        output_dir,
        torch.from_numpy(applied_transform).float() if applied_transform is not None else None,
    )
    out["ply_file_path"] = ply_filename

    if not dry_run:
        with open(output_dir / "transforms.json", "w", encoding="utf-8") as f:
            json.dump(out, f, indent=4)

    return len(frames)

def generate_image_rename_map(directory: Path, prefix: str, ext: str) -> dict:
    """Generates a mapping for renaming images from a given directory.
    
    Args:
        directory (Path): The directory containing images to be renamed.
        
    Returns:
        dict: A dictionary mapping original file paths to new file names.
    """
    image_rename_map = {}
    print(prefix)
    for file_path in directory.glob(f"{prefix}*{ext}"):
        db_name = file_path.name                
        new_name = db_name
        # new_name = file_path.name.replace(prefix, "")
        image_rename_map[db_name] = new_name    

    return image_rename_map  

###############################################################################################################
def run_command(command: List[str]) -> None:
    try:
        logging.info(f"Command: {' '.join(command)}")

        result = subprocess.run(command, shell=False, check=True, capture_output=True, text=True)

        logging.info(f"Command Output: {result.stdout.strip()}")
        if result.stderr:
            logging.warning(f"Command Error: {result.stderr.strip()}")

    except subprocess.CalledProcessError as e:
        logging.error(f"Command '{' '.join(command)}' failed with return code {e.returncode}")
        logging.error(f"Error Output: {e.stderr.strip()}")
        raise

# Task 2.2: COLMAP Feature Extractor
def run_feature_extractor(workspace_dir: Path, correspondence_config: Dict):
    feature_extractor_command = [
        "colmap", "feature_extractor",
        "--log_level", "2",
        "--database_path", str(workspace_dir / "colmap" / "database.db"),
        "--image_path", str(workspace_dir / "images"),
        "--image_list_path", str(workspace_dir / "image-list.txt"),
    ]

    for key, value in correspondence_config['feature_extractor'].items():
        feature_extractor_command += [f"--{key}", str(value)]

    
    run_command(feature_extractor_command)

# Task 2.3: COLMAP Vocab Tree Matcher
def run_vocab_tree_matcher(workspace_dir: Path, correspondence_config: Dict):
    vocab_tree_matcher_command = [
        "colmap", "vocab_tree_matcher",
        "--log_level", "2",
        "--database_path", str(workspace_dir / "colmap" / "database.db"),
        "--VocabTreeMatching.match_list_path", str(workspace_dir / "image-list.txt"),
    ]

    for key, value in correspondence_config.get('feature_matcher', {}).items():
        vocab_tree_matcher_command += [f"--{key}", str(value)]

    for key, value in correspondence_config.get('vocab_tree_matcher', {}).items():
        vocab_tree_matcher_command += [f"--{key}", str(value)]

    run_command(vocab_tree_matcher_command)

# Task 2.3: COLMAP Spatial Matcher
def run_spatial_matcher(workspace_dir: Path, correspondence_config: Dict):
    vocab_tree_matcher_command = [
        "colmap", "vocab_tree_matcher",
        "--database_path", str(workspace_dir / "colmap" / "database.db"),
        "--VocabTreeMatching.match_list_path", str(workspace_dir / "image-list.txt"),
    ]

    for key, value in correspondence_config.get('feature_matcher', {}).items():
        vocab_tree_matcher_command += [f"--{key}", str(value)]

    for key, value in correspondence_config.get('vocab_tree_matcher', {}).items():
        vocab_tree_matcher_command += [f"--{key}", str(value)]

    run_command(vocab_tree_matcher_command)

# Task 2.3: COLMAP Spatial Matcher
def run_exhaustive_matcher(workspace_dir: Path, correspondence_config: Dict):
    exhaustive_matcher_command = [
        "colmap", "exhaustive_matcher",
        "--log_level", "2",
        "--database_path", str(workspace_dir / "colmap" / "database.db"),
    ]

    for key, value in correspondence_config.get('feature_matcher', {}).items():
        exhaustive_matcher_command += [f"--{key}", str(value)]

    run_command(exhaustive_matcher_command)

# Task 2.4: COLMAP Image Registrator
def run_image_registrator(workspace_dir: Path, recon_dir: Path):
    image_registrator_command = [
        "colmap", "image_registrator",
        "--log_level", "2",
        "--database_path", str(workspace_dir / "colmap" / "database.db"),
        "--input_path", str(recon_dir),
        "--output_path", str(recon_dir),
    ]

    run_command(image_registrator_command)

# Task 2.5: COLMAP Bundle Adjuster
def run_bundle_adjuster(workspace_dir: Path, recon_dir: Path):
    bundle_adjuster_command = [
        "colmap", "bundle_adjuster",
        "--log_level", "2",
        "--input_path", str(recon_dir),
        "--output_path", str(recon_dir),
        "--BundleAdjustment.refine_principal_point", "1"
    ]

    run_command(bundle_adjuster_command)

###########################################################################
# Geo-registration in COLMAP

def extract_gps_from_exif(file_path: Path):
    """Extract GPS coordinates from an image file's EXIF data."""
    with open(str(file_path), 'rb') as f:
        tags = exifread.process_file(f)

    gps_latitude = tags.get('GPS GPSLatitude')
    gps_latitude_ref = tags.get('GPS GPSLatitudeRef')
    gps_longitude = tags.get('GPS GPSLongitude')
    gps_longitude_ref = tags.get('GPS GPSLongitudeRef')
    gps_altitude = tags.get('GPS GPSAltitude')
    
    if gps_latitude and gps_longitude:
        # Convert GPS coordinates to decimal
        lat = convert_to_decimal(gps_latitude, gps_latitude_ref)
        lon = convert_to_decimal(gps_longitude, gps_longitude_ref)
        alt = float(gps_altitude.values[0]) if gps_altitude else 0.0  # Use 0.0 if altitude is not available
        return lat, lon, alt
    else:
        return None, None, None

def convert_to_decimal(gps_coord, ref):
    """Convert GPS coordinates to decimal format."""
    degrees = gps_coord.values[0]
    minutes = gps_coord.values[1]
    seconds = gps_coord.values[2]
    decimal = degrees + (minutes / 60.0) + (seconds / 3600.0)
    if ref in ['S', 'W']:
        decimal = -decimal
    return decimal


def generate_geo_registration_txt(image_dir: Path, output_dir: Path) -> None:
    """Iterate over image files in the folder, rename them sequentially, 
    and write GPS data in ECEF format to a text file.
    Supports JPG and PNG images.
    """
    
    # 확장자 필터: jpg, png
    exts = {".jpg", ".jpeg", ".png"}
    image_files = sorted([f for f in image_dir.iterdir() if f.suffix.lower() in exts])

    with open(str(output_dir / "geo-registration-list.txt"), 'w') as f:
        for i, file_path in enumerate(image_files):
            # Generate new file name (확장자 유지)
            new_file_name = f"frame_{i+1:05d}{file_path.suffix}"
            
            # Extract GPS data
            lat, lon, alt = extract_gps_from_exif(file_path)
            
            if lat is not None and lon is not None:
                # Write only the new file name, not full path
                f.write(f"{new_file_name} {lat:.6f} {lon:.6f} {alt:.6f}\n")
            else:
                print(f"No GPS data found for {file_path.name}")


def geo_registration(image_dir: Path, ns_processed_dir: Path, colmap_model_path: str, ransac_thr: float):
    absolute_default_colmap_path = ns_processed_dir / "colmap/sparse/0"
    absolute_colmap_model_path = ns_processed_dir / colmap_model_path    
    absolute_colmap_model_path.mkdir(parents=True, exist_ok=True)

    

    # Geo-registration-list.txt 생성
    generate_geo_registration_txt(image_dir, absolute_colmap_model_path)

    colmap_command = [
        "colmap", "model_aligner",
        "--log_to_stderr", "1", 
        "--log_level", "0",
        "--input_path", str(absolute_default_colmap_path),
        "--output_path", str(absolute_colmap_model_path),
        "--ref_images_path", str(absolute_colmap_model_path / "geo-registration-list.txt"),
        "--ref_is_gps", "1",
        "--alignment_type", "ecef",
        "--transform_path", str(absolute_colmap_model_path / "sim3_transform.json"),
        "--robust_alignment", "1",
        "--robust_alignment_max_error", str(ransac_thr)
    ]

    run_command(colmap_command)

def convert_poses_t2_to_t1_with_gps(sim3_t1_path, sim3_t2_path, input_colmap2_path, output_colmap1_path):

    # Load Sim(3) transforms (from colmap to ECEF) 
    s_ecef_colmap1, R_ecef_colmap1, t_ecef_colmap1 = load_sim3(sim3_t1_path)
    s_ecef_colmap2, R_ecef_colmap2, t_ecef_colmap2 = load_sim3(sim3_t2_path)

    # Invert A→B to get B→A
    s_colmap1_ecef, R_colmap1_ecef, t_colmap1_ecef = inverse_sim3(s_ecef_colmap1, R_ecef_colmap1, t_ecef_colmap1)
    T_colmap1_ecef = sim3_matrix(s_colmap1_ecef, R_colmap1_ecef, t_colmap1_ecef)
    T_ecef_colmap2 = sim3_matrix(s_ecef_colmap2, R_ecef_colmap2, t_ecef_colmap2)

    # Load input transforms (C-world camera poses)
    with open(input_colmap2_path, 'r') as f:
        transforms = json.load(f)
    
    T_nerf_colmap = np.eye(4)
    T_nerf_colmap[:3, :4] = transforms['applied_transform']    # (4 x 4)

    T_colmap_nerf = np.linalg.inv(T_nerf_colmap)

    new_frames = []
    for frame in transforms["frames"]:
        T_nerf2 = np.array(frame["transform_matrix"])
        T_nerf1 = T_nerf_colmap @ T_colmap1_ecef @ T_ecef_colmap2 @ T_colmap_nerf @ T_nerf2
        frame["transform_matrix"] = T_nerf1.tolist()
        new_frames.append(frame)

    transforms["frames"] = new_frames

    # Save to output path
    with open(output_colmap1_path, 'w') as f:
        json.dump(transforms, f, indent=4)