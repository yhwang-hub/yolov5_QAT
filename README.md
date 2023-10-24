# Description
![Language](https://img.shields.io/badge/language-c++-brightgreen)
![Language](https://img.shields.io/badge/CUDA-11.1-brightgreen) 
![Language](https://img.shields.io/badge/TensorRT-8.5.1.7-brightgreen)
![Language](https://img.shields.io/badge/OpenCV-4.5.5-brightgreen) 
![Language](https://img.shields.io/badge/ubuntu-16.04-brightorigin)

This is a repository for QAT finetune on yolov5 using [TensorRT's pytorch_quantization tool](https://github.com/NVIDIA/TensorRT/tree/main/tools/pytorch-quantization)

# How To QAT Training

## 1.Setup
Suggest to use docker environment.

Download docker image：
```
docker pull longxiaowyh/yolov5:v2.0
```
Create docker container：
```
nvidia-docker run -itu root:root --name yolov5 --gpus all -v /your_path:/target_path -v /tmp/.X11-unix/:/tmp/.X11-unix/ -e DISPLAY=unix$DISPLAY -e GDK_SCALE -e GDK_DPI_SCALE  -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=compute,utility --shm-size=64g yolov5:v2.0 /bin/bash
```

1.Clone and apply patch

```
git clone git@github.com:yhwang-hub/yolov7_quantization.git
```
2.Install dependencies
```
pip install pytorch-quantization --extra-index-url https://pypi.ngc.nvidia.com
```
3.Prepare coco dataset
```
.
├── annotations
│   ├── captions_train2017.json
│   ├── captions_val2017.json
│   ├── instances_train2017.json
│   ├── instances_val2017.json
│   ├── person_keypoints_train2017.json
│   └── person_keypoints_val2017.json
├── coco -> coco
├── coco128
│   ├── images
│   ├── labels
│   ├── LICENSE
│   └── README.txt
├── images
│   ├── train2017
│   └── val2017
├── labels
│   ├── train2017
│   ├── train2017.cache
│   └── val2017
├── train2017.cache
├── train2017.txt
├── val2017.cache
└── val2017.txt
```

## 2.Start PTQ
### 2.1 Start sensitive layer analysis
```
python ptq.py --weights ./weights/yolov5s.pt --cocodir /home/wyh/disk/coco/ --batch_size 5 --save_ptq True --eval_origin --eval_ptq --sensitive True
```
Modify the ignore_layers parameter in ptq.py as follows
```
parser.add_argument("--ignore_layers", type=str, default="model\.24\.m\.(.*)", help="regx")
```
### 2.2 Start PTQ
```
python ptq.py --weights ./weights/yolov5s.pt --cocodir /home/wyh/disk/coco/ --batch_size 5 --save_ptq True --eval_origin --eval_ptq --sensitive False
```

## 3.Start QAT Training
```
python qat.py --weights ./weights/yolov5s.pt --cocodir /home/wyh/disk/coco/ --batch_size 5 --save_ptq True --save_qat True --eval_origin --eval_ptq --eval_qat
```
This script includes steps below:

- Insert Q&DQ nodes to get fake-quant pytorch model
  [Pytorch quntization tool ](https://github.com/NVIDIA/TensorRT/tree/main/tools/pytorch-quantization)provides automatic insertion of QDQ function. But for yolov7 model, it can not get the same performance as PTQ, because in Explicit mode(QAT mode), TensorRT will henceforth refer Q/DQ nodes' placement to restrict the precision of the model. Some of the automatic added Q&DQ nodes can not be fused with other layers which will cause some extra useless precision convertion. In our script, We find Some rules and restrictions for yolov7, QDQ nodes are automatically analyzed and configured in a rule-based manner, ensuring that they are optimal under TensorRT. Ensuring that all nodes are running INT8(confirmed with tool:[trt-engine-explorer](https://github.com/NVIDIA/TensorRT/tree/main/tools/experimental/trt-engine-explorer), see [scripts/draw-engine.py](https://github.com/NVIDIA-AI-IOT/yolo_deepstream/blob/main/yolov7_qat/scripts/draw-engine.py)). for details of this part, please refer quantization/rules.py, About the guidance of Q&DQ insert, please refer [Guidance_of_QAT_performance_optimization](https://github.com/NVIDIA-AI-IOT/yolo_deepstream/blob/main/yolov7_qat/doc/Guidance_of_QAT_performance_optimization.md)

- PTQ calibration
  After inserting Q&DQ nodes, we recommend to run PTQ-Calibration first. Per experiments, Histogram(MSE) is the best PTQ calibration method for yolov7. Note: if you are satisfied with PTQ result, you could also skip QAT.

- QAT training
  After QAT, need to finetune traning our model. after getting the accuracy we are satisfied, Saving the weights to files