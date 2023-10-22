import random
import os
import matplotlib.pyplot as plt
import pandas as pd
from transformers import DetrImageProcessor, AutoImageProcessor
import torch
import supervision as sv
import cv2


# Variables
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# MODEL_NAME = "facebook/detr-resnet-50"
MODEL_NAME = "facebook/deformable-detr-detic"

MODEL_PATH = 'Checkpoints/detr'
# image_processor = DetrImageProcessor.from_pretrained(MODEL_NAME)
image_processor = AutoImageProcessor.from_pretrained(MODEL_NAME)

def show_test_img(DATASET):
    categories = DATASET.coco.cats
    id2label = {k: v['name'] for k,v in categories.items()}
    image_ids = DATASET.coco.getImgIds()
    image_id = random.choice(image_ids)
    box_annotator = sv.BoxAnnotator()
    image = DATASET.coco.loadImgs(image_id)[0]
    annotations = DATASET.coco.imgToAnns[image_id]
    image_path = os.path.join(DATASET.root, image['file_name'])
    image = cv2.imread(image_path)
    detections = sv.Detections.from_coco_annotations(coco_annotation=annotations)
    labels = [f"{id2label[class_id]}" for _, _, class_id, _ in detections]
    frame_ground_truth = box_annotator.annotate(scene=image.copy(), detections=detections, labels=labels)
    plt.imshow(cv2.cvtColor(frame_ground_truth, cv2.COLOR_BGR2RGB))
    plt.show()
    
def generage_graphs(path_csv):
    data = pd.read_csv(path_csv)
    metrics = data.columns
    
    train_metrics = [met for met in metrics if 'train' in met]
    val_metrics = [met for met in metrics if 'validation' in met]
    
    for metric in train_metrics:
        plt.figure(figsize=(10, 5))
        plt.plot(data[metric], label=metric, marker='o')
        plt.title(metric)
        plt.xlabel('Index')
        plt.ylabel(metric)
        plt.legend()
        plt.grid(True)
        plt.show()
        
    for metric in val_metrics:
        plt.figure(figsize=(10, 5))
        plt.plot(data[metric], label=metric, marker='o')
        plt.title(metric)
        plt.xlabel('Index')
        plt.ylabel(metric)
        plt.legend()
        plt.grid(True)
        plt.show()
        