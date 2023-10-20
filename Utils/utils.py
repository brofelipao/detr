import random
import os
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from transformers import DetrImageProcessor
from datasets import *

image_processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
categories = TEST_DATASET.coco.cats
id2label = {k: v['name'] for k,v in categories.items()}

def show_test_img(TRAIN_DATASET):
    image_ids = TRAIN_DATASET.coco.getImgIds()
    image_id = random.choice(image_ids)
    # load image and annotatons 
    image = TRAIN_DATASET.coco.loadImgs(image_id)[0]
    annotations = TRAIN_DATASET.coco.imgToAnns[image_id]
    image_path = os.path.join(TRAIN_DATASET.root, image['file_name'])
    image = Image.open(image_path)

    fig, ax = plt.subplots(1)
    
    annotations = TRAIN_DATASET.coco.imgToAnns[image_id]
    draw = ImageDraw.Draw(image, "RGBA")

    cats = TRAIN_DATASET.coco.cats
    id2label = {k: v['name'] for k,v in cats.items()}

    for annotation in annotations:
        box = annotation['bbox']
        class_idx = annotation['category_id']
        x,y,w,h = tuple(box)
        draw.rectangle((x,y,x+w,y+h), outline='red', width=1)
        
    plt.imshow(image)
    plt.show()
    