import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, DistributedSampler, SequentialSampler, random_split
from torchvision.transforms.functional import to_grayscale
from torchvision import transforms
import numpy as np

from load_dataset import KITTI360_dataset

# ROOT_FOLDER = '/home/rangga/rangga-prjs/ViT-pytorch'

class FishEye2D(Dataset):
    def __init__(self, data_folder):
        self.data_folder = data_folder
        self.image_files = [f for f in os.listdir(data_folder) if f.endswith(('png', 'jpg', 'jpeg'))]
        self.transform_rgb = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.transform_depth = transforms.Compose([
            transforms.RandomRotation(180),
            transforms.Grayscale(),
            transforms.ToTensor(),
        ])
        self.label_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.data_folder, self.image_files[idx])

        # Load image
        image = Image.open(img_path).convert('RGB')

        image_rgb = self.transform_rgb(image)
        image_depth = self.transform_depth(image)
        
        labels = torch.randn(1, 7)
        labels[:, :4] = 2 * (labels[:, :4].clamp(-1, 1))  # Ensure values are between -1 and 1
        labels[:, 4:] = 0.2 * (labels[:, 4:].clamp(-1, 1)) 

        return [image_rgb, image_depth], labels


def get_loaders():
    data_folder = os.path.join(ROOT_FOLDER, "fisheye-2d-200-images")
    dataset = FishEye2D(data_folder)

    # Split the dataset into training and testing sets
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Create data loaders for training and testing
    batch_size = 20
    train_sampler = RandomSampler(train_dataset)
    test_sampler = SequentialSampler(test_dataset)
    train_loader = DataLoader(dataset=train_dataset, sampler=train_sampler, batch_size=batch_size, collate_fn=list)
    test_loader = DataLoader(dataset=test_dataset, sampler=test_sampler, batch_size=batch_size, collate_fn=list)

    return train_loader, test_loader

def get_kitti_loaders(args):
    file_csv_path = "/home/indowicom/umam/ext_auto_calib_camlid/ETRI_Project_Auto_Calib/dataset_kitti360_edit.csv"
    dataset = KITTI360_dataset(csv_path=file_csv_path, sequences=[0], frame_step=3)
    
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    train_sampler = RandomSampler(train_dataset)
    test_sampler = SequentialSampler(test_dataset)
    train_loader = DataLoader(dataset=train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, collate_fn=list)
    test_loader = DataLoader(dataset=test_dataset, sampler=test_sampler, batch_size=args.eval_batch_size, collate_fn=list)
    
    return train_loader, test_loader

# TEST ONLY
if __name__ == "__main__":
    data_folder = os.path.join(ROOT_FOLDER, "fisheye-2d-200-images")
    dataset_rgb = FishEye2D(data_folder, image_type='rgb')
    dataset_depth = FishEye2D(data_folder, image_type='depth')

    batch_size = 20
    data_loader_rgb = DataLoader(dataset=dataset_rgb, batch_size=batch_size, shuffle=True, collate_fn=list)
    data_loader_depth = DataLoader(dataset=dataset_depth, batch_size=batch_size, shuffle=True, collate_fn=list)

    for batch_images_rgb, batch_images_depth in zip(data_loader_rgb, data_loader_depth):
        print("Batch Size RGB:", len(batch_images_rgb))
        print("Batch Size Depth:", len(batch_images_depth))
        # Individual images in the batch
        for i, image in enumerate(batch_images_rgb):
            print("RGB Image Shape:", batch_images_rgb[i].shape) 
            print("Depth Image Shape:", batch_images_depth[i].shape) 