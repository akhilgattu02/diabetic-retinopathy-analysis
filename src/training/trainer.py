from src.training.losses import dice_loss, ce_loss
from src.models.unet_plus_plus import model

import torch
import segmentation_models_pytorch as smp
import torch.nn
from tqdm import tqdm

import boto3
import io
import torch
import torch.nn.utils as utils
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("/Users/akhilgattu/Desktop/diabetic-retinopathy-analysis/runs/idridd_experiment/logs")

def train_model(model, seg_dataloader, seg_test_dataloader, optimizer, EPOCHS, DEVICE):
    global_step = 0
    for ep in range(EPOCHS):
        model.train()
        total_loss = 0
        for img, mask in tqdm(seg_dataloader, desc=f"Train {ep}/{EPOCHS}"):
            img = img.to(DEVICE)
            mask = mask.to(DEVICE)
            output = model(img)
            optimizer.zero_grad()
            loss = 0.5 * dice_loss(output, mask) + 0.5 * ce_loss(output, mask)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

            writer.add_scalar("Train/Loss", loss.item(), global_step)

            global_step += 1
            total_loss += loss.item()
        print(total_loss/max(1, len(seg_dataloader)))
        writer.add_scalar("Train/EpochLoss",
                        total_loss/len(seg_dataloader),
                        ep)
        evaluate_model(model, seg_test_dataloader, DEVICE, ep)
    return model

def evaluate_model(model, seg_test_dataloader, DEVICE, ep):
    total_loss = 0
    model = model.to('mps')
    model.eval()
    with torch.no_grad():
        for img, mask in tqdm(seg_test_dataloader):
                img = img.to(DEVICE)
                mask = mask.to(DEVICE)
                output = model(img)
                loss = 0.5 * dice_loss(output, mask) + 0.5 * ce_loss(output, mask)
                total_loss += loss.item()
        print(total_loss/max(1, len(seg_test_dataloader)))

    writer.add_scalar("Val/Loss", total_loss/len(seg_test_dataloader), ep)

def save_model_to_s3(model, optimizer, EPOCHS):
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

def save_model_locally(model):
    torch.save(model.state_dict(), "/Users/akhilgattu/Desktop/diabetic-retinopathy-analysis/checkpoints/unet_plus_plus_idridd.pth")