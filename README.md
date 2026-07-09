# 🚀FCAIQNet
The open available code for paper "FCAIQNet: Frequency-Channel Adaptive Interaction and Quad-scale Adaptive Fusion Network for Multimodal UAV Object Detection"
<img width="8458" height="5888" alt="overall" src="https://github.com/user-attachments/assets/fed038a1-2a9a-4adc-a87c-79f293dcb4fe" />

# 📚Dataset
DroneVehicleis a large-scale paired RGB–IR UAV vehicle dataset covering nighttime scenes. Its aerial images contain small, densely distributed and partially occluded vehicles under strong illumination variations, especially at night, which closely matches our RGB–IR UAV small-object fusion detection setting and is suitable for evaluating cross-modal interaction and multi-scale feature fusion. In our experiments, white image borders are removed, each RGB–IR pair is resized to 640×640, and annotations are converted to YOLO horizontal bounding boxes. The processed dataset contains 17,990 training pairs, 1,469 validation pairs.

LLVIP is a paired visible–infrared dataset for low-light pedestrian detection and image fusion. Nevertheless, its strictly aligned RGB–IR image pairs and low-light scenes make it well suited for evaluating the generalization ability of multimodal fusion detection beyond aerial scenarios. It contains aligned 14,665 training pairs and 3,666 test pairs, which are resized to 640×640 for input.

OGSOD-1.0 is a SAR–optical remote-sensing object detection dataset built for optical-guided SAR detection. We use this dataset to verify whether the proposed multimodal interaction and fusion strategy can generalize from RGB–IR fusion to broader cross-modal remote-sensing detection. It contains 14,665 training pairs and 3,666 test pairs, which are resized to 512×512 for input. 

﻿DroneVehicleis：
 
LLVIP：

OGSOD-1.0：

# 🔣Pretrained model

# 🔧Requirement
conda create -n FCAIQNet python=3.10
conda activate FCAIQNet
pip install -U pip setuptools wheel
pip install -r requirements.txt







