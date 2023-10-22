from transformers import DetrForObjectDetection, DetrConfig
import torch
import random
import cv2
import numpy as np
from Classes.CocoDetection import *
import matplotlib.pyplot as plt
import transformers
import supervision as sv
import pandas as pd
import matplotlib.pyplot as plt
from Utils.utils import *

TEST_DATASET = CocoDetection(image_directory_path=TEST_DIRECTORY, image_processor=image_processor, train=False)

def test_model(DATASET):
    detections_output = None
    categories = DATASET.coco.cats
    id2label = {k: v['name'] for k,v in categories.items()}
    box_annotator = sv.BoxAnnotator()

    model = DetrForObjectDetection.from_pretrained(MODEL_PATH)
    model.to(DEVICE)

    image_ids = DATASET.coco.getImgIds()
    for i in range(1):
        image_id = random.choice(image_ids)
        image_ids.remove(image_id)
        print('Image #{}'.format(image_id))
        # load image and annotatons 
        image = DATASET.coco.loadImgs(image_id)[0]
        annotations = DATASET.coco.imgToAnns[image_id]
        image_path = os.path.join(DATASET.root, image['file_name'])
        image = cv2.imread(image_path)
        # Annotate ground truth
        detections = sv.Detections.from_coco_annotations(coco_annotation=annotations)
        labels = [f"{id2label[class_id]}" for _, _, class_id, _ in detections]
        frame_ground_truth = box_annotator.annotate(scene=image.copy(), detections=detections, labels=labels)

        # Annotate detections
        with torch.no_grad():
            # load image and predict
            inputs = image_processor(images=image, return_tensors='pt').to(DEVICE)
            outputs = model(**inputs)
            # post-process
            target_sizes = torch.tensor([image.shape[:2]]).to(DEVICE)
            results = image_processor.post_process_object_detection(
                outputs=outputs, 
                threshold=0.7, 
                target_sizes=target_sizes
            )[0]
            detections_output = sv.Detections.from_transformers(transformers_results=results)
            labels = [f"{id2label[class_id]} {confidence:.2f}" for _, confidence, class_id, _ in detections_output]
            frame_detections = box_annotator.annotate(scene=image.copy(), detections=detections_output)
            
        # Combine both images side by side and display
        # fig, axs = plt.subplots(1, 2, figsize=(20, 10))
        # axs[0].imshow(cv2.cvtColor(frame_ground_truth, cv2.COLOR_BGR2RGB))
        # axs[0].axis('off')
        # axs[0].set_title(f'Ground Truth - {len(detections)}')

        # axs[1].imshow(cv2.cvtColor(frame_detections, cv2.COLOR_BGR2RGB))
        # axs[1].axis('off')
        # axs[1].set_title(f'Detections - {len(detections_output)}')
        
        plt.imshow(cv2.cvtColor(frame_detections, cv2.COLOR_BGR2RGB))
        plt.title(f'Detections - {len(detections_output)} - Real {len(detections)}')
        plt.show()

test_model(TEST_DATASET)
#generage_graphs('lightning_logs/version_0/metrics.csv')