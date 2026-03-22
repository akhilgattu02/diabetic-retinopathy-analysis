from src.training.init_dataset import DEVICE, seg_dataloader, seg_test_dataloader
from src.training.trainer import train_model, evaluate_model, save_model_to_s3, save_model_locally
from src.models.unet_plus_plus import model

import torch


MODEL = model.to(DEVICE)

def run_training_pipeline():
    DEVICE = 'mps' if torch.mps.is_available() else 'cpu'
    optimizer = torch.optim.AdamW(MODEL.parameters())
    EPOCHS = 25
    trained_model = train_model(MODEL, seg_dataloader, seg_test_dataloader, optimizer, EPOCHS, DEVICE)
    #save_model_to_s3(trained_model, "unet_plus_plus_effnet_b3.pth")
    save_model_locally(trained_model)

