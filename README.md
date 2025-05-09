# Project Introduction
MRS-YOLO: High-Precision Remote Sensing Object Detection Method Based on Multi-Dimensional Feature Interaction Enhancement and Dynamic Fusion

# Installation
```
conda create -n mrs python=3.9  
conda activate mrs  
pip install -r requirements.txt  
pip install -e .
```
# Demo
Run the following command to start the project:  
```
python app.py
```
# Catalog Structure Description
```
MRS-YOLO/  
├── ultralytics/  
│   ├── assets/        # Resource files and model assets  
│   ├── cfg/           # Configuration files  
│   ├── data/          # Dataset and dataloader definitions  
│   ├── engine/        # Training, validation, and prediction pipelines  
│   ├── hub/           # Model downloading and registry logic  
│   ├── models/        # Model architecture definitions 
│   ├── nn/            # Custom layers and loss functions  
│   ├── solutions/     # End-to-end solution pipelines  
│   ├── trackers/      # Tracking-related modules  
│   ├── utils/         # Utility functions and evaluation metrics  
│   └── __init__.py    # Package initializer  
│  
├── datasets/ # Data preparation and preprocessing scripts  
│  
├── runs/ # Training logs and saved model checkpoints  
│  
├── app.py # Entry point for training  
├── requirements.txt # Python dependencies  
└── README.md # Project introduction and usage  
```
# Data Preparation
```
datasets/  
└── DOTA/  
    ├── images/  
    │   ├── train/  
    │   ├── val/  
    │   └── test/  
    ├── labels/  
    │   ├── train/  
    │   ├── val/  
    │   └── test/  
    └── data.yaml   # Dataset configuration file for DOTA (class names, paths, etc.)
```
# Validation
```
yolo val model=path/weight.pth data=path/data.yaml batch=1
```
# Or
```
from ultralytics import YOLOv10

model = YOLOv10('your-model.pt')

model.val(data='coco.yaml', batch=256)
```
# Training
```
yolo detect train data=path/data.yaml model=mrs-yolo.yaml epochs=500 batch=256 imgsz=640 device=0,1,2,3,4,5,6,7
```
# Or
```
from ultralytics import YOLOv10

model = YOLOv10()


model.train(data='path/data.yaml', epochs=500, batch=256, imgsz=640)
```
# Prediction
```
yolo predict model=path/your-model.pth source=path/data.yaml
```
# Or
```
from ultralytics import YOLOv10

model = YOLOv10.from_pretrained('path/your-model.pth')

model.predict(source='path/data.yaml',)
```

