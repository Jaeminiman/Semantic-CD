import cv2
import os
import numpy as np
from typing import Tuple

class ImagePair:
    def __init__(self, image1_path: str, image2_path: str):
        """
        Initialize the ImagePair class with two image paths.

        Args:
            image1_path (str): Path to the first image (initial).
            image2_path (str): Path to the second image (new).
        """
        # Set image paths
        self.image1_path = image1_path
        self.image2_path = image2_path

        # File name alignment restriction        
        basename1 = os.path.splitext(os.path.basename(self.image1_path))[0]
        basename2 = os.path.splitext(os.path.basename(self.image2_path))[0]

        # 접두사 제거
        index1 = basename1.replace("initial_paired_", "")
        index2 = basename2.replace("new_", "")

        # 인덱스 비교
        assert index1 == index2, f"Mismatch: {index1} vs {index2}"
        

        # Initialize image data
        self.image1 = None
        self.image2 = None

        # Get filename
        self.filename = os.path.splitext(os.path.basename(self.image1_path))[0]

        # Load images
        self._load_images()

    def _load_images(self) -> None:
        """
        Load images from provided paths.

        Raises:
            FileNotFoundError: If one or both image paths are invalid.
        """
        if os.path.exists(self.image1_path) and os.path.exists(self.image2_path):
            self.image1 = cv2.imread(self.image1_path)
            self.image2 = cv2.imread(self.image2_path)
        else:
            raise FileNotFoundError("One or both image paths are invalid.")

    def get_image_pair(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the image pair as numpy arrays.

        Returns:
            Tuple[np.ndarray, np.ndarray]: The loaded images as numpy arrays.
        """
        return self.image1, self.image2

    def get_image_shapes(self) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
        """
        Get the shapes of the loaded images.

        Returns:
            Tuple[Tuple[int, int, int], Tuple[int, int, int]]: The shapes of the two images.

        Raises:
            ValueError: If the images are not loaded properly.
        """
        if self.image1 is not None and self.image2 is not None:
            return self.image1.shape, self.image2.shape
        else:
            raise ValueError("Images are not loaded properly.")

    def reload_images(self) -> None:
        """
        Reload images from the given paths.
        """
        self._load_images()

    def save_images(self, output_dir: str) -> None:
        """
        Save the images to the specified output directory.

        Args:
            output_dir (str): Path to the output directory.

        Raises:
            ValueError: If the images are not loaded properly.
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Define output paths
        image1_output_path = os.path.join(output_dir, 'initial', f'{self.filename}.jpg')
        image2_output_path = os.path.join(output_dir, 'new', f'{self.filename}.jpg')

        # Save images
        if self.image1 is not None and self.image2 is not None:
            os.makedirs(os.path.dirname(image1_output_path), exist_ok=True)
            os.makedirs(os.path.dirname(image2_output_path), exist_ok=True)
            cv2.imwrite(image1_output_path, self.image1)
            cv2.imwrite(image2_output_path, self.image2)
        else:
            raise ValueError("Images are not loaded properly.")
    
    def resize_image(self, img, resize_factor=None, max_size=None):
        """
        이미지 다운스케일 함수
        - resize_factor 주어지면 해당 배수로 축소
        - max_size가 주어지면 해당 최대 크기 기준으로 축소
        둘 중 하나만 사용하세요.
        
        반환: (resized_img, scale)
        """
        h, w = img.shape[:2]

        if resize_factor is not None:
            scale = 1.0 / resize_factor
            new_w = int(w * scale)
            new_h = int(h * scale)
            resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            return resized_img, scale

        elif max_size is not None:
            scale = min(max_size / h, max_size / w)
            if scale < 1.0:
                new_w = int(w * scale)
                new_h = int(h * scale)
                resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            else:
                resized_img = img
                scale = 1.0
            return resized_img, scale

        else:
            raise ValueError("Either downscale_factor or max_size must be provided.")

    def rectify(self, resize_factor) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Rectify the initial image to align with the new image using homography.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]:
                The corrected initial image, corrected new image, and the homography matrix.
        """
        # Create SIFT feature detector
        sift = cv2.SIFT_create()

        # Copy the original images
        image1 = self.image1.copy()  # initial
        image2 = self.image2.copy()  # new

        # Convert images to grayscale
        img1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        img2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
                
        img1_resize, scale1 = self.resize_image(img1, resize_factor)
        img2_resize, scale2 = self.resize_image(img2, resize_factor)

        kp1, des1 = sift.detectAndCompute(img1_resize, None)
        kp2, des2 = sift.detectAndCompute(img2_resize, None)

        if des1 is None or des2 is None:
            print(f"❌ No features in {img1_path}")


        matcher = cv2.BFMatcher()
        matches = matcher.knnMatch(des1, des2, k=2)
        good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]

        if len(good_matches) < 10:
            print(f"❌ Not enough good matches: {len(good_matches)}")


        pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])

        # Guided Matching: RANSAC 기반 Fundamental Matrix 추정으로 inlier 필터링
        F, inliers = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, ransacReprojThreshold=3.0, confidence=0.99)


        pts1 = pts1[inliers.ravel() == 1]
        pts2 = pts2[inliers.ravel() == 1]        

        # Homography with RANSAC
        H_12, inliers = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
        if H_12 is None or inliers.sum() < 10:
            print(f"❌ Homography too unstable or insufficient inliers: {inliers.sum()}")

        
        scale_matrix = np.eye(3)
        scale_matrix[0, 0] = 1/scale1
        scale_matrix[1, 1] = 1/scale1

        homography_12 = scale_matrix @ H_12 @ np.linalg.inv(scale_matrix)

    
        # Warp the initial image to align with the new image
        height, width, channel = image2.shape
        corrected_image1 = cv2.warpPerspective(image1, homography_12, (width, height))

        # Create a mask to mark valid pixel areas in the warped image
        mask_warped = np.zeros_like(image1, dtype=np.uint8)
        cv2.warpPerspective(np.ones_like(image1, dtype=np.uint8), homography_12, (width, height), dst=mask_warped, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        # Apply mask to the new image
        corrected_image2 = image2 * mask_warped

        return corrected_image1, corrected_image2, homography_12