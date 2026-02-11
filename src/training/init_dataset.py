from src.data_pipeline.generate_load_dataset import SegDataSet, IDRiDDatasetBuilder
from torch.utils.data import DataLoader, Dataset
import torch
import segmentation_models_pytorch as smp

image_paths = "/Users/akhilgattu/Desktop/diabetic-retinopathy-analysis/data/processed/train/images/"
mask_paths = "/Users/akhilgattu/Desktop/diabetic-retinopathy-analysis/data/processed/train/masks/"

image_test_paths = "/Users/akhilgattu/Desktop/diabetic-retinopathy-analysis/data/processed/test/images/"
mask_test_paths = "/Users/akhilgattu/Desktop/diabetic-retinopathy-analysis/data/processed/test/masks/"
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