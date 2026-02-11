import segmentation_models_pytorch as smp
import torch

#Use UNET++ model with timm-efficientnet-b3 encoder and imagenet encoder weights
model = smp.UnetPlusPlus(
        encoder_name="timm-efficientnet-b3",
        encoder_weights="imagenet",
        in_channels=3,
        classes=6,
        activation=None,
)