import os
import cv2
import numpy as np
from typing import List, Tuple, Union, Dict, Any
import matplotlib.pyplot as plt
import json
import logging
import time
import yaml
from PIL import Image
import argparse


from segment_anything import SamPredictor, SamAutomaticMaskGenerator, sam_model_registry
from ObjectDetector import YOLODetector, MMDetDetector
from SCD.utils import dir_utils

from image_pair import ImagePair
from object_detection_utils import *


# Configure logging
logging.basicConfig(level=logging.ERROR)

def normalize_vector(vector: np.ndarray) -> np.ndarray:
    """Function to normalize a vector."""
    norm = np.linalg.norm(vector, axis=0, keepdims=True)
    return vector / (norm + 1e-8)  # Prevent division by zero by adding a small value


def cosine_similarity(embed1: np.ndarray, embed2: np.ndarray) -> np.ndarray:
    """Function to calculate cosine similarity between two embeddings."""
    embed1 = embed1.astype(np.float32)
    embed2 = embed2.astype(np.float32)
    
    norm_vector1 = normalize_vector(embed1)
    norm_vector2 = normalize_vector(embed2)
    
    cosine_similarity_map = np.sum(norm_vector1 * norm_vector2, axis=0)
    return cosine_similarity_map


def box_cosine_similarity(initial_embed: np.ndarray, new_embed: np.ndarray, boxes: List[List[float]], width: int, height: int, res_embed: int = 64) -> Tuple[List[float], List[np.ndarray]]:
    """Function to calculate cosine similarity for specific box regions."""
    cos_sim = []
    cos_sim_map = []
    for box in boxes:
        cls, x1, y1, x2, y2 = box
        
        x1, x2 = map(lambda p: int(p / width * res_embed), [x1, x2])
        y1, y2 = map(lambda p: int(p / height * res_embed), [y1, y2])

        x1_pad, y1_pad = map(lambda p: max(p - 1, 0), [x1, y1])
        x2_pad = min(x2 + 1, res_embed - 1)
        y2_pad = min(y2 + 1, res_embed - 1)

        embed1 = initial_embed[..., y1_pad:y2_pad, x1_pad:x2_pad]
        embed2 = new_embed[..., y1_pad:y2_pad, x1_pad:x2_pad]
        
        cosine_similarity_map = cosine_similarity(embed1, embed2)        
        logits = np.mean(cosine_similarity_map)
        
        cos_sim.append(logits)
        cos_sim_map.append(cosine_similarity_map)
    return cos_sim, cos_sim_map


def raw_to_sam_scale(coord: List[int], width: int, height: int, res_sam: int = 1024) -> List[int]:
    """Function to convert raw coordinates to SAM scale."""
    x1, y1, x2, y2 = coord
    x1, x2 = map(lambda p: int(p / width * res_sam), [x1, x2])
    y1, y2 = map(lambda p: int(p / height * res_sam), [y1, y2])
    return [x1, y1, x2, y2]


class Grounded_CD:
    def __init__(self, config: Dict):
        """Initialize the Grounded_CD class with configuration file."""
        self.conf = config
        self._setup()

    def load_config(self, config_path: str) -> dict:
        """Function to load YAML configuration file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def _setup(self) -> None:        
        """Setup function to initialize devices and models."""
        self.device = self.conf['device']
        # Setup SAM model
        self._setup_sam()

        # Setup Object Detection model
        self._setup_object_detection()

    def _setup_sam(self) -> None:
        """Function to set up SAM model."""
        sam_checkpoint = self.conf['sam_checkpoint']
        model_type = self.conf['model_type']        
        self.res_embed = self.conf['res_embed']
        self.res_sam = self.conf['res_sam']        
        self.threshold_cossim = self.conf['threshold_cos_sim']
        
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=self.device)
        self.predictor1 = SamPredictor(sam) 
        self.predictor2 = SamPredictor(sam) 

    def _setup_object_detection(self) -> None:
        """Function to set up object detection model."""
         
        print(f"Setup Object Detection: {self.conf['od_model']} model")

        # Object Detection config
        self.od_model = self.conf['od_model']
        od_conf = self.conf['od_config']
        od_checkpoint = self.conf['od_checkpoint']
        self.od_texts = self.conf['od_target_classes']
        self.od_pred_thr = self.conf['od_pred_score_thr']
        self.od_iou_thr = self.conf['od_iou_thr']
        self.od_save_dir = self.conf['od_save_dir']
        os.makedirs(self.od_save_dir, exist_ok=True)

        if self.od_model == "yolo":

            self.inferencer = YOLODetector(
                checkpoint_path=od_checkpoint, 
                config_path=od_conf,
                device=self.device)

        elif self.od_model == "grounding_dino":
            
            self.inferencer = MMDetDetector(                
                checkpoint_path=od_checkpoint,
                config_path=od_conf,
                device=self.device,    
            )                
        else:
            print("error")
        
        # Input image directories & sorting image paths
        image_dir = self.conf['image_dir']                
                
        self.new_paths = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) 
                                if (f.endswith('.png') or f.endswith('.jpg')) and f.startswith('new_')])
        
        self.initial_paths = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) 
                                    if (f.endswith('.png') or f.endswith('.jpg')) and f.startswith('initial_paired_')])

        self.image_pair_resize_factor = self.conf['image_pair_resize_factor']

        # For saving debut/retirement CD results
        self.output_dir = self.conf['output_dir']
        self.output_dir_debut = os.path.join(self.output_dir, "debut")
        self.output_dir_retirement = os.path.join(self.output_dir, "retirement")        
        self.output_dir_homography = os.path.join(self.output_dir, "homography")   
        self.output_dir_simmap = os.path.join(self.output_dir, "simmap")        
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.output_dir_debut, exist_ok=True)
        os.makedirs(self.output_dir_retirement, exist_ok=True)                        
        os.makedirs(self.output_dir_homography, exist_ok=True)                        
        os.makedirs(self.output_dir_simmap, exist_ok=True)                        

    def process_all(self) -> None:
        """Function to iterate through all image files and process them."""
        for idx, (image1_path, image2_path) in enumerate(zip(self.initial_paths, self.new_paths)):            
            logging.info(f"Processing: {idx}/{len(self.new_paths)}")
            self._process(image1_path, image2_path)

    def _process(self, image1_path: str, image2_path: str) -> None:
        """Function to process an image pair."""
        # Get image pair
        image_pair = ImagePair(image1_path, image2_path)
        image1_rectified, image2_rectified, homography_12 = image_pair.rectify(resize_factor=self.image_pair_resize_factor)

        initial_od_output = self.od_inference(image1_rectified, image1_path, "initial")
        new_od_output = self.od_inference(image2_rectified, image2_path, "new")

        # Resize images for embedding
        image1_sam = cv2.resize(image1_rectified, (self.res_sam, self.res_sam)) 
        image2_sam = cv2.resize(image2_rectified, (self.res_sam, self.res_sam))

        # Set images to predictors to generate embeddings
        self.predictor1.set_image(image1_sam)
        self.predictor2.set_image(image2_sam)
        
        embed1 = self.predictor1.features.squeeze().cpu().numpy()
        embed2 = self.predictor2.features.squeeze().cpu().numpy()

        height, width, _ = image1_rectified.shape
        
        debut_boxes = self._extract_boxes(new_od_output, width, height)
        retirement_boxes = self._extract_boxes(initial_od_output, width, height)

        cossim_debut, cossim_map_debut = box_cosine_similarity(embed1, embed2, debut_boxes, width, height, self.res_embed)
        cossim_retirement, cossim_map_retirement = box_cosine_similarity(embed1, embed2, retirement_boxes, width, height, self.res_embed)

        debut_masks, debut_boxes = self._debut_detection(debut_boxes, cossim_debut, width, height)
        retirement_masks, retirement_boxes = self._retirement_detection(retirement_boxes, cossim_retirement, width, height)

        debut_mask_images = map(lambda masks: cv2.resize(masks[0].astype(np.uint8), (width, height)), debut_masks)
        retirement_mask_images = map(lambda masks: cv2.resize(masks[0].astype(np.uint8), (width, height)), retirement_masks)

        debut_change_mask = np.zeros((height, width), dtype=np.uint8)
        retirement_change_mask = np.zeros((height, width), dtype=np.uint8)

        for mask_image in debut_mask_images:    
            debut_change_mask = np.bitwise_or(debut_change_mask, mask_image)
        for mask_image in retirement_mask_images:
            retirement_change_mask = np.bitwise_or(retirement_change_mask, mask_image)
        
        # Already warped
        # retirement_change_mask_warped = cv2.warpPerspective(retirement_change_mask, homography_12, (width, height), borderMode=cv2.BORDER_CONSTANT, borderValue=0) 

        change_mask = np.bitwise_or(debut_change_mask, retirement_change_mask)

        # Save masks for visualization
        debut_change_mask = (debut_change_mask * 255).astype('uint8')
        retirement_change_mask = (retirement_change_mask * 255).astype('uint8')
        change_mask = (change_mask * 255).astype('uint8')

        # Save images using cv2
        cv2.imwrite(os.path.join(self.output_dir_debut, os.path.basename(image2_path)), debut_change_mask)
        cv2.imwrite(os.path.join(self.output_dir_retirement, os.path.basename(image1_path)), retirement_change_mask)

        filename = os.path.basename(image1_path).replace("initial_paired_", "")
        cv2.imwrite(os.path.join(self.output_dir, filename), change_mask)

        # Save homography_12 for GT rectify
        name_without_ext = os.path.splitext(filename)[0]
        np.save(os.path.join(self.output_dir_homography, f"{name_without_ext}.npy"), homography_12)

        
        cosine_similarity_map = np.array(cosine_similarity(embed1, embed2))
        simmap_path = os.path.join(self.output_dir_simmap, f"{name_without_ext}.jpg")
        plt.imsave(simmap_path, cosine_similarity_map, cmap='hot')


    def _extract_boxes(self, od_output: str, width: int, height: int) -> List[List[float]]:
        """Function to extract bounding boxes from Grounding Dino output."""
        boxes = []
        with open(od_output, 'r') as f:
            lines = f.readlines()
            for line in lines:
                # Extract bounding box information
                cls, conf, x_center_r, y_center_r, box_width_r, box_height_r = map(float, line.split(' '))
                cls = int(cls)

                x1 = int((x_center_r - box_width_r / 2) * width)
                x2 = int((x_center_r + box_width_r / 2) * width)
                y1 = int((y_center_r - box_height_r / 2) * height)
                y2 = int((y_center_r + box_height_r / 2) * height)
                
                boxes.append([cls, x1, y1, x2, y2])
        return boxes

    def _debut_detection(self, debut_boxes: List[List[int]], cossim_debut: List[float], width: int, height: int) -> Tuple[List[np.ndarray], List[List[int]]]:
        """Function to detect newly appearing objects (debut detection)."""
        debut_masks = []
        pop_idx = []
        for idx, cos_sim in enumerate(cossim_debut):
            # Test pharse                
            print(f"[Debut] Cos-Sim: {cos_sim}, box:[{debut_boxes[idx]}]")
            #########################################
            if cos_sim > self.threshold_cossim:                                   
                
                pop_idx.append(idx)
                continue

            cls, x1, y1, x2, y2 = debut_boxes[idx]
            x1, y1, x2, y2 = raw_to_sam_scale([x1, y1, x2, y2], width, height, self.res_sam)
            
            input_box = np.array([x1, y1, x2, y2])

            debut_mask, scores, logits = self.predictor2.predict(    
                box=input_box[None, :],
                multimask_output=True    
            )
            debut_masks.append(debut_mask)    

        for idx in reversed(pop_idx):
            debut_boxes.pop(idx)
        
        return debut_masks, debut_boxes

    def _retirement_detection(self, retirement_boxes: List[List[int]], cossim_retirement: List[float], width: int, height: int) -> Tuple[List[np.ndarray], List[List[int]]]:
        """Function to detect objects that have disappeared (retirement detection)."""
        retirement_masks = []
        pop_idx = []
        for idx, cos_sim in enumerate(cossim_retirement):    
            # Test pharse                
            print(f"[Retirement] Cos-Sim: {cos_sim}, box:[{retirement_boxes[idx]}]")
            #########################################

            if cos_sim > self.threshold_cossim:                

                pop_idx.append(idx)
                continue

            cls, x1, y1, x2, y2 = retirement_boxes[idx]
            x1, y1, x2, y2 = raw_to_sam_scale([x1, y1, x2, y2], width, height, self.res_sam)
            
            input_box = np.array([x1, y1, x2, y2])

            masks_retirement, scores, logits = self.predictor1.predict(    
                box=input_box[None, :],
                multimask_output=True
            )     
            retirement_masks.append(masks_retirement) 

        for idx in reversed(pop_idx):
            retirement_boxes.pop(idx)
        
        return retirement_masks, retirement_boxes

    def od_inference(self, image: np.ndarray, image_path: str, phase: str) -> str:
        """Function to run inference using Grounding Dino."""
        texts = self.od_texts            
        pred_score_thr = self.od_pred_thr
        iou_threshold = self.od_iou_thr
        save_dir = self.od_save_dir
        save_image_path = os.path.join(save_dir, phase, os.path.basename(image_path))        
        os.makedirs(os.path.dirname(save_image_path), exist_ok=True)

        # Use the detector to do inference
        detections = self.inferencer.predict(image, texts=texts, conf_threshold=pred_score_thr)

        labels = np.array([det['class_id'] for det in detections])
        scores = np.array([det['confidence'] for det in detections])
        bboxes = np.array([det['bbox'] for det in detections])

        # Set the IoU threshold for NMS(YOLO는 이미 반영되어 있을 수 있음)
        bboxes, scores, labels = class_wise_nms(bboxes, scores, labels, iou_threshold=iou_threshold)

        # Save the label file (YOLO format)
        save_label_path = os.path.join(save_dir, f"{os.path.splitext(os.path.basename(image_path))[0]}.txt")
        logging.info(save_label_path)

        # visualization of object detection
        image_canvas = image.copy()
        color = [0, 255, 0]

        with open(save_label_path, "w") as label_file:
            for i, box in enumerate(bboxes):
                conf = scores[i]  # Confidence score
                class_id = labels[i]  # Class ID
                
                # Extract bounding box information                    
                x1, y1, x2, y2 = map(int, box)  # Pixel coordinates
                x1_f, y1_f, x2_f, y2_f = box  # Float coordinates

                # Exception 1: Skip false prediction if region is black
                # if np.all(image[y1 + 1, x1 + 1] == 0) or np.all(image[y2 - 1, x2 - 1] == 0):                                  
                #     continue
                
                # Calculate image area
                image_area = image.shape[0] * image.shape[1]

                # Calculate bounding box area
                box_width = x2 - x1
                box_height = y2 - y1
                box_area = box_width * box_height

                # Exception 2: Skip if box area is greater than 90% of the image area
                box_ratio = box_area / image_area                    
                if box_ratio >= 0.9:
                    continue
                
                # YOLO style object detection output
                x_center = (x1_f + x2_f) / 2 / image.shape[1]
                y_center = (y1_f + y2_f) / 2 / image.shape[0]
                width = box_width / image.shape[1]
                height = box_height / image.shape[0]
                                                        
                txt_line = f'{class_id} {conf:.4f} {x_center:.4f} {y_center:.4f} {width:.4f} {height:.4f}\n'
                label_file.write(txt_line)

                # 바운딩 박스 그리기
                cv2.rectangle(image_canvas, (x1, y1), (x2, y2), color, 2)

                # 라벨 그리기
                cv2.putText(image_canvas, f"{class_id} ({scores[i]:2f})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.imwrite(save_image_path, image_canvas)

        return save_label_path

def parse_arguments() -> argparse.Namespace:
    logging.info("Parsing command line arguments...")
    parser = argparse.ArgumentParser(description="Specify configuration file for NeRF correspondence processing")
    parser.add_argument('--config', type=str, required=True, help="Path to configuration file")
    return parser.parse_args()

def main() -> None:
    start_time = time.time()
    try:
        args = parse_arguments()
        config = dir_utils.load_config(args.config)    

        grounded_CD = Grounded_CD(config)
        grounded_CD.process_all()

    except Exception as e:
        logging.exception("Unexpected error occurred during program execution")

    end_time = time.time()
    logging.info(f"Total processing time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.exception("Unexpected error occurred during program execution")