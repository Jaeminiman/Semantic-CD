from abc import ABC, abstractmethod
from typing import List, Union, Dict, Any
from ultralytics import YOLO
from mmdet.apis import DetInferencer
import re
import yaml

class BaseObjectDetector(ABC):
    @abstractmethod
    def __init__(self, checkpoint_path: str, config_path: str, device: str = 'cuda'):
        pass
        
    @abstractmethod
    def predict(self, image: Union[str, Any], texts: str, conf_threshold: float = 0.25) -> List[Dict]:
        """Run inference and return a list of detected objects."""
        pass

    @abstractmethod
    def visualize(self, image: Union[str, Any], predictions: List[Dict], save_path: str = None):
        """Optional: visualize or save the detection results on image."""
        pass


class YOLODetector(BaseObjectDetector):
    def __init__(self, checkpoint_path: str, config_path: str, device: str = 'cuda'):
        self.model = YOLO(checkpoint_path)
        self.device = device
        self._config_parse(config_path)

    def predict(self, image: Union[str, Any], texts: str, conf_threshold: float = 0.25) -> List[Dict]:

        # texts-class matching                
        keywords = re.findall(r'\b\w[\w\s\-]*\w\b', texts.lower()) # "." "," 등을 기준으로 나눔
        
        matched_class_ids = [
            idx for idx, name in enumerate(self.class_names)
            if any(keyword in (self.class_dict.get(name.lower()) or '') for keyword in keywords)
        ]
        
        print(f"Key words: {[keyword for keyword in keywords]}")
        print(f"Matched class names: {[self.class_names[i] for i in matched_class_ids]}")
        print(f"Matched group names: {set([self.class_dict.get(self.class_names[i].lower()) for i in matched_class_ids])}")

        results = self.model.predict(
            source=image,
            conf=conf_threshold,
            device=self.device
        )

        # 예측 결과에서 원하는 클래스만 필터링
        detections = []
        for result in results:
            boxes = result.boxes

            if boxes is None or len(boxes) == 0:
                continue
                                               
            for box in boxes.data.cpu().numpy():
                x1, y1, x2, y2, conf, cls = box[:6]
                cls = int(cls)
                
                if cls in matched_class_ids:
                    detections.append({
                        "bbox": [float(x1), float(y1), float(x2), float(y2)],
                        "confidence": float(conf),
                        "class_id": cls
                    }
                    )

                    if cls == 10:
                        print({
                        "bbox": [float(x1), float(y1), float(x2), float(y2)],
                        "confidence": float(conf),                        
                    })

        return detections

    def visualize(self, image: Union[str, Any], predictions: List[Dict], save_path: str = None):
        self.model.predict(source=image, save=True if save_path else False, show=not save_path)

    def _config_parse(self, config_path: str):
        
        with open(config_path, "r") as f:
            data = yaml.safe_load(f)

        self.class_names = data['names']        
        class_dict_raw = data['class_dict']

        # YAML에서 잘못된 형식으로 인식된 경우 (list of dict처럼), dict로 변환
        if isinstance(class_dict_raw, list):
            self.class_dict = {}
            for item in class_dict_raw:
                self.class_dict.update(item)
        else:
            self.class_dict = class_dict_raw  # 이미 dict이면 그대로 사용

        # check
        for i, name in enumerate(self.class_names):
            group = self.class_dict.get(name, "others")
            print(f"{name} -> {group}")


class MMDetDetector(BaseObjectDetector):
    def __init__(self, checkpoint_path: str, config_path: str, device: str = 'cuda'):
        self.inferencer = DetInferencer(model=config_path, weights=checkpoint_path, device=device)

    def predict(self, image: Union[str, Any], conf_threshold: float = 0.25) -> List[Dict]:
        results = self.inferencer(image)
        detections = []
        for det in results['predictions'][0]['instances']:
            if det['score'] >= conf_threshold:
                detections.append({
                    "bbox": det['bbox'],
                    "confidence": det['score'],
                    "class_id": det['label']
                })
        return detections

    def visualize(self, image: Union[str, Any], predictions: List[Dict], save_path: str = None):
        self.inferencer.visualize(image, result=predictions, out_file=save_path)
