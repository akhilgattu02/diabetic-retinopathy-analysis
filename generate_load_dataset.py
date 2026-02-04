'''
Author: Akhil Gattu
Date: 27th Jan 2026
Script to generate and load IDRiD dataset in proper format to be 
used by Dataset and DataLoader classes in torch.utils.data module 
'''

from torch.utils.data import Dataset, DataLoader
import cv2, os
import torch
import os
import numpy as np
import cv2
import albumentations as A

image_paths = "/Users/akhilgattu/Desktop/VLM_project/Data/train/images/"
mask_paths = "/Users/akhilgattu/Desktop/VLM_project/Data/train/masks/"

'''
Mapping abnormality to class id's
'''


'''
Class to generate final dataset in a
format that is readable by Dataset and DataLoader classes in torch.utils.data module 
'''
class IDRiDDatasetBuilder:
    #Initialize dataset type and map each abnormality to an id
    def __init__(self, dataset_type):
        self.dataset_type = dataset_type
        self.class_id_abnormality = {
            "OD": 5,
            "SE": 4,
            "EX": 3,
            "HE": 2,
            "MA": 1 
        }


    #Resize image to (512, 512) resolution
    def transform(self, img, is_mask=False):

        if is_mask:
            img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_NEAREST)
        else:
            img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)

        return img


    #Return masks of an image abnormality
    def return_image_abnormality(self, dataset_type, abnormal_path, ab, h, w):
        if dataset_type == "train":
            msk_root = "A. Segmentation/2. All Segmentation Groundtruths/a. Training Set/"
        
        else:
            msk_root = "A. Segmentation/2. All Segmentation Groundtruths/b. Testing Set/"

        if ab in os.listdir(os.path.join(msk_root, abnormal_path)):
            ab = os.path.join(msk_root, abnormal_path+ab)
            img_ab = cv2.imread(ab, cv2.IMREAD_GRAYSCALE)
            img_ab = self.transform(img_ab, True)

        else:
            img_ab = np.zeros((h, w), dtype=np.uint8)
        
        return img_ab

    '''
    Create training and testing final dataset, where each tif file consists of all the masks combined 
    and each pixel is a value in [0, 5] mapping class id to abnormailty.
    '''
    def create_dataset(self, dataset_type):

        if dataset_type == "train":
            root = "A. Segmentation/1. Original Images/a. Training Set/"
        else:
            root = "A. Segmentation/1. Original Images/b. Testing Set/"

        img_root = "Data"
        dir = os.listdir(root)
        dir.sort()

        if dataset_type not in os.listdir("/Users/akhilgattu/Desktop/VLM_project/"+img_root):
            os.makedirs(img_root+"/"+dataset_type+"/images")
            os.makedirs(img_root+"/"+dataset_type+"/masks")

        for fol in dir:
        
            img_path = os.path.join(root, fol)
            img_name = fol.split(".")[0]
            ma =  img_name + "_MA.tif"
            he = img_name + "_HE.tif"
            ex = img_name + "_EX.tif"
            se = img_name + "_SE.tif"
            od = img_name + "_OD.tif"

            img = cv2.imread(root+fol, cv2.IMREAD_COLOR)
            img = self.transform(img, False)

            msk = np.zeros((512, 512), dtype=np.uint8)
            h, w = 512, 512
            

            img_ma = self.return_image_abnormality(dataset_type, "1. Microaneurysms/", ma, h, w)
            img_he = self.return_image_abnormality(dataset_type, "2. Haemorrhages/", he, h, w)
            img_ex = self.return_image_abnormality(dataset_type, "3. Hard Exudates/", ex, h, w)
            img_se = self.return_image_abnormality(dataset_type, "4. Soft Exudates/", se, h, w)
            img_od = self.return_image_abnormality(dataset_type, "5. Optic Disc/", od, h, w)

            msk[img_od > 0] = self.class_id_abnormality["OD"]
            msk[img_se > 0] = self.class_id_abnormality["SE"]
            msk[img_ex > 0] = self.class_id_abnormality["EX"]
            msk[img_he > 0] = self.class_id_abnormality["HE"]
            msk[img_ma > 0] = self.class_id_abnormality["MA"]


            imgs = f"/Users/akhilgattu/Desktop/VLM_project/{img_root}/{dataset_type}/images/{fol}"
            masks = f"/Users/akhilgattu/Desktop/VLM_project/{img_root}/{dataset_type}/masks/{img_name}.tif"
            cv2.imwrite(imgs, img)
            cv2.imwrite(masks, msk)

'''
Class that uses torch.utils.data module to generate (img, mask) data 
batchwise with a batch size.
'''
class SegDataSet(Dataset):
    def __init__(self, image_paths, mask_paths):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.image_paths = [self.image_paths+image_path for image_path in sorted(os.listdir(self.image_paths))]
        self.mask_paths = [self.mask_paths+mask_path for mask_path in sorted(os.listdir(self.mask_paths))]
        self.aug = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
        ])

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_UNCHANGED)

        augmented = self.aug(image=image, mask=mask)
        image, mask = augmented["image"], augmented["mask"]

        image = image.astype(np.float32) / 255.0

        # ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406])
        std  = np.array([0.229, 0.224, 0.225])
        image = (image - mean) / std

        image = torch.from_numpy(image).permute(2,0,1).float()
        mask = torch.from_numpy(mask).long()

        return image, mask
