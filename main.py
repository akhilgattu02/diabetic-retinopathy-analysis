from generate_load_dataset import SegDataSet, IDRiDDatasetBuilder
from torch.utils.data import DataLoader, Dataset
import torch
import segmentation_models_pytorch as smp
import torch.nn
from tqdm import tqdm

import boto3
import io
import torch

image_paths = "/Users/akhilgattu/Desktop/VLM_project/Data/train/images/"
mask_paths = "/Users/akhilgattu/Desktop/VLM_project/Data/train/masks/"

image_test_paths = "/Users/akhilgattu/Desktop/VLM_project/Data/test/images/"
mask_test_paths = "/Users/akhilgattu/Desktop/VLM_project/Data/test/masks/"

DEVICE = 'mps' if torch.mps.is_available() else 'cpu'

dataset_builder_train = IDRiDDatasetBuilder("train")
dataset_builder_test = IDRiDDatasetBuilder("test")

class_to_id = dataset_builder_train.class_id_abnormality

dataset_builder_train.create_dataset("train")
dataset_builder_test.create_dataset("test")

seg_dataset = SegDataSet(image_paths, mask_paths)
seg_dataloader = DataLoader(
    dataset=seg_dataset,
    batch_size=3
)

seg_test_dataset = SegDataSet(image_paths, mask_paths)
seg_test_dataloader = DataLoader(
    dataset=seg_test_dataset,
    batch_size=3
)

#Use UNET++ model with timm-efficientnet-b3 encoder and imagenet encoder weights
model = smp.UnetPlusPlus(
        encoder_name="timm-efficientnet-b3",
        encoder_weights="imagenet",
        in_channels=3,
        classes=5,
        activation=None,
)


DiceLoss = smp.losses.DiceLoss
FocalLoss = smp.losses.FocalLoss

dice_loss = DiceLoss('multilabel', from_logits=True)
focal_loss = FocalLoss('multilabel', gamma=1.5)
bce = torch.nn.BCEWithLogitsLoss()

optimizer = torch.optim.AdamW(model.parameters())

EPOCHS = 10
model = model.to(DEVICE)
for ep in range(EPOCHS):
    model.train()
    total_loss = 0
    for img, mask in tqdm(seg_dataloader, desc=f"Train {ep}/{EPOCHS}"):
        img = img.to(DEVICE)
        mask = mask.to(DEVICE)
        output = model(img)
        optimizer.zero_grad()
        loss = 0.5 * dice_loss(output, mask) + 0.5 * bce(output, mask)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    print(total_loss/max(1, len(seg_dataloader)))

total_loss = 0
model = model.to('mps')
with torch.no_grad():
    for img, mask in tqdm(seg_test_dataloader):
            img = img.to(DEVICE)
            mask = mask.to(DEVICE)
            output = model(img)
            loss = 0.5 * dice_loss(output, mask) + 0.5 * bce(output, mask)
            total_loss += loss.item()
    print(total_loss/max(1, len(seg_test_dataloader)))


bucket_name = "diabetic-retinopathy-model"
s3_key = "models/unetplusplus/unet_plus_plus_idridd.pth"

# Move to CPU to avoid device serialization issues
model = model.to("cpu")

# Create in-memory buffer
buffer = io.BytesIO()

torch.save({
    "model_state": model.state_dict(),
    "optimizer_state": optimizer.state_dict(),
    "epochs": EPOCHS
}, buffer)

#Point buffer to start of file

buffer.seek(0) 

s3 = boto3.client("s3")

s3.upload_fileobj(
    buffer,
    bucket_name,
    s3_key
)

print("Model uploaded directly to S3.")
