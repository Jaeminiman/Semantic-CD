import os
import cv2
import torch
import numpy as np
import shutil
import subprocess

###############################################
# 1. 기본 설정
###############################################
# 원본 이미지가 있는 폴더 (원하는 경로로 변경)
images_folder = "images"  

# pre-trained YOLO 체크포인트 (YOLOv5 모델 checkpoint)
checkpoint_path = "/workspace/Laboratory/04.model/yolo/yolo11x.pt"

# active learning으로 모은 데이터셋을 저장할 폴더 구조 (YOLOv5의 형식)
active_learning_dataset_folder = "active_learning_dataset"
train_images_folder = os.path.join(active_learning_dataset_folder, "images/train")
train_labels_folder = os.path.join(active_learning_dataset_folder, "labels/train")
os.makedirs(train_images_folder, exist_ok=True)
os.makedirs(train_labels_folder, exist_ok=True)

# low-confidence 임계값 (예: 0.5 미만이면 사용자 검토)
low_conf_thresh = 0.5

###############################################
# 2. YOLOv5 모델 로드 (torch.hub 이용)
###############################################
print("모델 로딩 중...")
model = torch.hub.load('ultralytics/yolov5', 'custom', path=checkpoint_path, force_reload=True)
print("모델 로딩 완료.")

###############################################
# 3. 폴더 내 이미지 순회 및 low-confidence 검출에 대해 사용자 입력 받기
###############################################
for img_file in os.listdir(images_folder):
    img_path = os.path.join(images_folder, img_file)
    img = cv2.imread(img_path)
    if img is None:
        continue  # 읽기 실패 시 넘어감

    # 모델 추론 수행
    results = model(img)
    # 결과: 각 행 [x1, y1, x2, y2, confidence, class]
    predictions = results.xyxy[0].cpu().numpy()  

    # 현재 이미지에서 사용자가 수정할 bounding box 저장용 리스트
    corrected_annotations = []

    # low-confidence인 각 검출에 대해 처리
    for i, pred in enumerate(predictions):
        x1, y1, x2, y2, conf, cls = pred
        if conf < low_conf_thresh:
            # 현재 검출 박스 표시 (참고용)
            img_display = img.copy()
            cv2.rectangle(img_display, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            cv2.imshow("Low Confidence Detection", img_display)
            cv2.waitKey(1)  # 간단한 대기

            # 사용자에게 해당 박스를 수정할지 여부 확인
            print(f"이미지: {img_file} - 검출 {i}: (x1={x1:.1f}, y1={y1:.1f}, x2={x2:.1f}, y2={y2:.1f}), confidence={conf:.2f}")
            user_choice = input("이 박스를 수정하시겠습니까? (y/n): ")
            if user_choice.lower() == "y":
                bbox_input = input("수정된 bounding box 좌표를 입력하세요 (형식: x1,y1,x2,y2): ")
                try:
                    coords = list(map(float, bbox_input.split(',')))
                    if len(coords) == 4:
                        label_input = input("클래스 라벨을 입력하세요 (정수값): ")
                        corrected_annotations.append((int(label_input), coords))
                    else:
                        print("입력 형식 오류 - 해당 박스는 건너뜁니다.")
                except Exception as e:
                    print("입력 처리 오류:", e, "- 해당 박스는 건너뜁니다.")
            cv2.destroyWindow("Low Confidence Detection")

    # 수정된 annotation이 하나라도 있다면, 해당 이미지를 active learning 데이터셋에 추가
    if corrected_annotations:
        # 원본 이미지를 지정된 학습 이미지 폴더로 복사
        dest_img_path = os.path.join(train_images_folder, img_file)
        shutil.copy(img_path, dest_img_path)

        # YOLO 형식의 annotation 파일 생성
        # YOLO 형식: 각 줄 -> <class> <x_center> <y_center> <width> <height> (모두 0~1 사이로 정규화)
        height, width, _ = img.shape
        annotation_lines = []
        for label, (x1, y1, x2, y2) in corrected_annotations:
            x_center = (x1 + x2) / 2 / width
            y_center = (y1 + y2) / 2 / height
            box_width = (x2 - x1) / width
            box_height = (y2 - y1) / height
            annotation_lines.append(f"{label} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}")
        
        # 이미지 파일명과 동일한 txt 파일로 저장 (예: image1.jpg -> image1.txt)
        label_filename = os.path.splitext(img_file)[0] + ".txt"
        label_path = os.path.join(train_labels_folder, label_filename)
        with open(label_path, "w") as f:
            f.write("\n".join(annotation_lines))
            
cv2.destroyAllWindows()
print("모든 이미지에 대한 active learning annotation 수집 완료.")

###############################################
# 4. YOLO 학습에 사용할 data.yaml 파일 생성
###############################################
# (참고: 여기서는 간단하게 학습 이미지 폴더를 train과 val 모두로 사용)
data_yaml = f"""
train: {os.path.abspath(train_images_folder)}
val: {os.path.abspath(train_images_folder)}
nc: 1
names: ['class0']
"""
data_yaml_path = os.path.join(active_learning_dataset_folder, "data.yaml")
with open(data_yaml_path, "w") as f:
    f.write(data_yaml)

print(f"data.yaml 파일 생성: {data_yaml_path}")

###############################################
# 5. 수집된 annotation으로 모델 추가학습 (fine-tuning)
###############################################
# YOLOv5의 train.py 스크립트를 호출합니다.
# (train.py 스크립트가 현재 경로 또는 PYTHONPATH에 있어야 합니다.)
train_command = f"python train.py --img 640 --batch 4 --epochs 10 --data {data_yaml_path} --weights {checkpoint_path} --cache"
print("추가학습 시작:", train_command)
subprocess.run(train_command.split())

print("Active learning을 통한 추가학습 완료.")
