# Semantic-CD in Construction Sites 🏗️

**NeRF-based Image Pair Generation for Change Detection + YOLO-SAM Integrated Semantic Change Detection**

---

## 📦 Environment Setup

**Base Environment**

* Ubuntu 22.04
* CUDA 11.8
* Python 3.8
* COLMAP 3.8
* NeRFStudio 1.1.5

---

### 1. Create Conda Environment

```bash
conda create --name semantic-CD -y python=3.8
conda activate semantic-CD
pip install --upgrade pip
```

---

### 2. Install NeRFStudio

```bash
pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
conda install -y -c "nvidia/label/cuda-11.8.0" cuda-toolkit
pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch

cd {code_path}/nerfstudio
pip install --upgrade pip setuptools
pip install -e .
```

---

### 3. Install Semantic-CD

```bash
cd {code_path}

pip install -e .
pip install pysrt piexif exifread
conda install -c conda-forge ffmpeg -y
```

---

### 4. Install COLMAP (Linux)

```bash
apt-get update 

apt-get install -y \
    git \
    cmake \
    ninja-build \
    build-essential \
    libboost-program-options-dev \
    libboost-graph-dev \
    libboost-system-dev \
    libeigen3-dev \
    libfreeimage-dev \
    libmetis-dev \
    libgoogle-glog-dev \
    libgtest-dev \
    libgmock-dev \
    libsqlite3-dev \
    libglew-dev \
    qtbase5-dev \
    libqt5opengl5-dev \
    libcgal-dev \
    libceres-dev \
    libcurl4-openssl-dev \
    libmkl-full-dev \
    libboost-all-dev \
    libflann-dev

apt-get install -y \
    nvidia-cuda-toolkit \
    nvidia-cuda-toolkit-gcc

cd {code_path}/colmap
mkdir build
cd build
cmake .. -GNinja -DBLA_VENDOR=Intel10_64lp -DCMAKE_CUDA_ARCHITECTURES=86
ninja
ninja install

# Verify installation
colmap -h
```

---

## 🚀 Run Command

```bash
SCD-init --config /root/code/Semantic-CD/SCD/NeRF_3D_Reconstruction/config/3D_segmentation_earthwork/nerf_config_01_gwangju.yaml
```

---

## 🧩 Pipeline Overview

### YOLO-SAM Semantic Change Detection Framework

![Pipeline](./YOLO-SAM-cd_framework.png)

Our framework integrates **change detection** and **semantic segmentation**, and consists of the following **three stages**:

---

### **1. Image Rectification**

* Input images from two different timestamps (`t1`, `t2`)
* **Geometric correction** and **dimension alignment** are applied to ensure pixel-wise consistency across time



### **2. Interesting Object Detection with YOLO**

* A fine-tuned **YOLO detector** is applied to detect key objects for site management
* Identifies:

  * **Initial objects** (already existing in `t1`)
  * **New objects** (appeared in `t2`)
* Provides **semantic-level classification** of construction materials, structures, and equipment



### **3. Change Detection with SAM**

* Each aligned image is encoded using the **SAM Encoder** to extract feature embeddings (`e_t1`, `e_t2`)
* A **change map** is generated from feature differences, highlighting potential object changes
* The **Prompt Encoder** and **Mask Decoder** produce fine-grained segmentation masks for the detected changes


---

## 📊 Example Results

### Semantic Change Detection(Classification)

![Semantic Segmentation Example](./831be388-ea44-491c-9cb6-d7bf839bf55c.png)

* **materials** (yellow)
* **structure** (green)
* **equipment** (cyan)

---

### Semantic Change Detection(Appearance/Removal)

![Semantic Change Detection Example](./9349d703-5ecc-430e-859b-6cccd26eb668.png)

* **appearance** (blue)
* **removal** (red)

---

## 📖 References

* [COLMAP Official Documentation](https://colmap.github.io/)
* [NeRFStudio GitHub](https://github.com/nerfstudio-project/nerfstudio)

---

## 📌 TODO

* [ ] Add example dataset
* [ ] Provide detailed training/inference pipeline

