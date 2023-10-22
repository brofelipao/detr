from Utils.utils import *
from torch.utils.data import DataLoader
from Classes.CocoDetection import *
from Classes.Detr import Detr
from Classes.DeformableDetr import DeformableDetr
from pytorch_lightning import Trainer


MAX_EPOCHS = 10
TRAIN_DATASET = CocoDetection(image_directory_path=TRAIN_DIRECTORY, image_processor=image_processor, train=True)
VAL_DATASET = CocoDetection(image_directory_path=VAL_DIRECTORY, image_processor=image_processor, train=False)
TEST_DATASET = CocoDetection(image_directory_path=TEST_DIRECTORY, image_processor=image_processor, train=False)

categories = TRAIN_DATASET.coco.cats
id2label = {k: v['name'] for k,v in categories.items()}

def collate_fn(batch):
    pixel_values = [item[0] for item in batch]
    encoding = image_processor.pad(pixel_values, return_tensors="pt")
    labels = [item[1] for item in batch]
    return {
        'pixel_values': encoding['pixel_values'],
        'pixel_mask': encoding['pixel_mask'],
        'labels': labels
    }

TRAIN_DATALOADER = DataLoader(dataset=TRAIN_DATASET, collate_fn=collate_fn, batch_size=4, shuffle=True)
VAL_DATALOADER = DataLoader(dataset=VAL_DATASET, collate_fn=collate_fn, batch_size=4)
TEST_DATALOADER = DataLoader(dataset=TEST_DATASET, collate_fn=collate_fn, batch_size=4)

#model = Detr(lr=1e-5, lr_backbone=1e-5, weight_decay=3e-4, id2label=id2label, train_dataloader=TRAIN_DATALOADER, val_dataloader=VAL_DATALOADER)
model = DeformableDetr(lr=1e-5, lr_backbone=1e-3, weight_decay=3e-4, id2label=id2label, train_dataloader=TRAIN_DATALOADER, val_dataloader=VAL_DATALOADER)

trainer = Trainer(devices=1, accelerator="gpu", max_epochs=MAX_EPOCHS, gradient_clip_val=0.1, accumulate_grad_batches=8, log_every_n_steps=5)
trainer.fit(model)
model.model.save_pretrained(MODEL_PATH)

