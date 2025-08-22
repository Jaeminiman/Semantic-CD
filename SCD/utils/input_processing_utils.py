import cv2
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple
import logging
import pysrt
import piexif
from fractions import Fraction
from itertools import chain
from R3DR.utils import dir_utils

def process_video(video_dir: Path, image_dir: Path, image_size: Tuple[int, int], image_start: int, image_step: int, image_ext: str, is_gps: int) -> Path:
    """Process video and extract images."""        
    logging.info("Starting video processing!")

    video_processor = ProcessVideo(video_dir= video_dir, output_dir = image_dir, size = image_size, frame_start = image_start, frame_step = image_step, ext = image_ext, is_gps = is_gps)
    video_processor.process()

    print(image_dir)
    logging.info("Video processing completed!")
    return image_dir

def process_input_data(config: Dict) -> Path:
    """Process the input data based on the type specified in the configuration.
    
    Args:
        config (Dict): Configuration dictionary with input image processing settings. ex) config["initial_settings"] / config["correspondence_settings"]        
        
    Returns:
        Path: Directory containing the processed images.
    """

    input_dir = Path(config['input_dir'])        
    
    image_dir = dir_utils.get_image_dir(config)
        
    logging.info(f"Image Directory: {image_dir}")
    logging.info(f"Input type: {config['input_type']}")
    
    if dir_utils.is_dir_empty(image_dir):
        if config['input_type'] == 'video':                
            logging.info("Converting video to images...")
            image_dir = process_video(
                input_dir, 
                image_dir, 
                tuple(config['image_size']) if config['image_size'] else None, 
                config['image_start'],
                config['image_step'],
                config['image_ext'],
                config['is_gps']
            )
        else:            
            logging.info("Using image directory. Copy from source to workspace")                                  
            dir_utils.copy_images(input_dir, image_dir)            
    else:
        logging.warn(f"{image_dir} is not empty, leading to skip processing input")

    
    
    return image_dir

# 2. GPS 정보를 EXIF로 변환
def deg_to_dms(decimal_coordinate, cardinal_directions):
    """
    This function converts decimal coordinates into the DMS (degrees, minutes and seconds) format.
    It also determines the cardinal direction of the coordinates.

    :param decimal_coordinate: the decimal coordinates, such as 34.0522
    :param cardinal_directions: the locations of the decimal coordinate, such as ["S", "N"] or ["W", "E"]
    :return: degrees, minutes, seconds and compass_direction
    :rtype: int, int, float, string
    """
    if decimal_coordinate < 0:
        compass_direction = cardinal_directions[0]
    elif decimal_coordinate > 0:
        compass_direction = cardinal_directions[1]
    else:
        compass_direction = ""
    degrees = int(abs(decimal_coordinate))
    decimal_minutes = (abs(decimal_coordinate) - degrees) * 60
    minutes = int(decimal_minutes)
    seconds = Fraction((decimal_minutes - minutes) * 60).limit_denominator(100)
    return degrees, minutes, seconds, compass_direction

def dms_to_exif_format(dms_degrees, dms_minutes, dms_seconds):
    """
    This function converts DMS (degrees, minutes and seconds) to values that can
    be used with the EXIF (Exchangeable Image File Format).

    :param dms_degrees: int value for degrees
    :param dms_minutes: int value for minutes
    :param dms_seconds: fractions.Fraction value for seconds
    :return: EXIF values for the provided DMS values
    :rtype: nested tuple
    """
    exif_format = (
        (dms_degrees, 1),
        (dms_minutes, 1),
        (int(dms_seconds.limit_denominator(100).numerator), int(dms_seconds.limit_denominator(100).denominator))
    )
    return exif_format
    


def add_geolocation(image_path, latitude, longitude, altitude):
    """
    This function adds GPS values to an image using the EXIF format.
    This fumction calls the functions deg_to_dms and dms_to_exif_format.

    :param image_path: image to add the GPS data to
    :param latitude: the north–south position coordinate
    :param longitude: the east–west position coordinate
    """
    # converts the latitude and longitude coordinates to DMS
    latitude_dms = deg_to_dms(latitude, ["S", "N"])
    longitude_dms = deg_to_dms(longitude, ["W", "E"])

    # convert the DMS values to EXIF values
    exif_latitude = dms_to_exif_format(latitude_dms[0], latitude_dms[1], latitude_dms[2])
    exif_longitude = dms_to_exif_format(longitude_dms[0], longitude_dms[1], longitude_dms[2])
    exif_altitude = (int(altitude * 1000), 1000)

    try:
        # Load existing EXIF data
        exif_data = piexif.load(image_path)

        # https://exiftool.org/TagNames/GPS.html
        # Create the GPS EXIF data
        coordinates = {
            piexif.GPSIFD.GPSVersionID: (2, 0, 0, 0),
            piexif.GPSIFD.GPSLatitude: exif_latitude,
            piexif.GPSIFD.GPSLatitudeRef: latitude_dms[3],
            piexif.GPSIFD.GPSLongitude: exif_longitude,
            piexif.GPSIFD.GPSLongitudeRef: longitude_dms[3],
            piexif.GPSIFD.GPSAltitude: exif_altitude
        }

        # Update the EXIF data with the GPS information
        exif_data['GPS'] = coordinates

        # Dump the updated EXIF data and insert it into the image
        exif_bytes = piexif.dump(exif_data)
        piexif.insert(exif_bytes, image_path)
    except Exception as e:
        print(f"Error: {str(e)}")


@dataclass
class ProcessVideo:
    video_dir: Path
    output_dir: Path
    size: Optional[Tuple[int, int]]
    frame_start: int
    frame_step: int
    num_downscales: int = 3
    ext: str = '.jpg' # output image extension
    is_gps: int = 0 
    """Number of times to downscale the images. Downscales by 2 each time. For example a value of 3 will downscale the
       images by 2x, 4x, and 8x."""
    
    def _parse_srt(self):
        """Parse subtitles in DJI M3E for getting GPS data"""

        files = sorted(list(self.video_dir.glob("*.SRT")))
        reg = 0
        frame_cnt_residual = 0
        gps_data = {}
        for srt_path in files:        
            subtitles = pysrt.open(srt_path)            
            for sub in subtitles:
                frame_cnt = int(sub.text.split("FrameCnt: ")[1].split(",")[0]) + frame_cnt_residual
                latitude = float(sub.text.split("[latitude: ")[1].split("]")[0])
                longitude = float(sub.text.split("[longitude: ")[1].split("]")[0])
                altitude = float(sub.text.split("abs_alt: ")[1].split("]")[0])
                gps_data[frame_cnt] = {"latitude": latitude, "longitude": longitude, "altitude": altitude}
                reg = frame_cnt
            frame_cnt_residual = reg             
        return gps_data

    def process(self) -> None:
        image_dir = self.output_dir
        image_dir.mkdir(parents=True, exist_ok=True)
        
        # Step1. Parse SRT
        if(self.is_gps):
            gps_data = self._parse_srt()

        # For multiple extensions
        patterns = ["*.mp4", "*.MP4"]

        frame_count = 0
        
        files = sorted(list(chain.from_iterable(self.video_dir.glob(pattern) for pattern in patterns)))
        for video_path in files:
            
            print(str(video_path))
                
            cap = cv2.VideoCapture(str(video_path))
            
       
            if not cap.isOpened():
                print(f"Error opening video file: {video_path}")
                return None
              
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break                

                frame_count += 1

                adjust_frame_count = frame_count - self.frame_start
                if adjust_frame_count <= 0:
                    continue
                if adjust_frame_count % self.frame_step != 0:
                    continue
                
                
                idx = int(adjust_frame_count//self.frame_step) # start from idx 1
                frame_path = image_dir/f"frame_{idx:05}{self.ext}"
                if self.size is not None:
                    resized_frame = cv2.resize(frame, self.size)
                else:
                    resized_frame = frame
                    
                cv2.imwrite(str(frame_path), resized_frame)

                if(self.is_gps):
                    gps_info = gps_data[frame_count]
                    add_geolocation(str(frame_path), gps_info["latitude"], gps_info["longitude"], gps_info["altitude"])
                    
            cap.release() 

                
                
        print(f"Number of split frame: {int(adjust_frame_count//self.frame_step)}")
            
        return