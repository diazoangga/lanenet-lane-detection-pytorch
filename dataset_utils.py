import os
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from genericpath import exists
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split

def read_dataset(file_path, num_samples=4, split_ratio=0.2):
    assert os.path.exists(file_path), "Training file (txt) does not exist, please make sure the input file is right."
    
    num_files = sum(1 for _ in open(file_path))
    num_samples = min(num_samples, num_files)
    if num_files < num_samples:
        print('Number of samples is higher than the number of existing files. number of samples is setted to the number of existing files: ', num_files)
        num_samples = num_files
    skip_files = sorted(random.sample(range(1, num_files + 1), (num_files - num_samples)))
    text_file = pd.read_csv(file_path, header=None, sep=' ', skiprows=skip_files, names=['img', 'inst', '0', '1', '2', '3'])
    
    img_path = text_file['img'].values
    inst_path = text_file['inst'].values

    train_img_paths, val_img_paths, train_inst_paths, val_inst_paths = train_test_split(
                img_path, inst_path, test_size=split_ratio, random_state=42)
    
    return train_img_paths, train_inst_paths, val_img_paths, val_inst_paths

def img_preprocess(image_path, shape=(256, 512)):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize(shape),
        transforms.ToTensor()
    ])
    return transform(image).to(torch.device("cuda"))

def inst_preprocess(inst_path, shape=(256, 512)):
    image = Image.open(inst_path).convert("L")
    transform = transforms.Compose([
        transforms.Resize(shape),
        transforms.ToTensor()
    ])

    inst = transform(image).to(torch.device("cuda"))
   
    label_binary = torch.zeros_like(inst, dtype=torch.uint8, device="cuda")
    label_binary[inst != 0] = 1
    bin = torch.squeeze(label_binary)

    return bin.long(), inst

class SegmentationDataset(Dataset):
    def __init__(self, img_paths, inst_paths, shape=(256,512), transform=None):
        self.img_paths = img_paths
        self.inst_paths = inst_paths
        self.transform = transform
        self.shape = shape
    
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        img_path = './data/' + self.img_paths[idx]
        inst_path = './data/' + self.inst_paths[idx]
        assert exists(img_path), f'The dataset file does not exist: {img_path}'
        assert exists(inst_path), f'The dataset file does not exist: {inst_path}'

        img = img_preprocess(img_path, shape=self.shape)
        bin, inst = inst_preprocess(inst_path, shape=self.shape)
        
        # if self.transform:
        #     img = self.transform(img)
        
        return img, bin, inst
    
if __name__ == "__main__":
    train_img_paths, train_inst_paths, val_img_paths, val_inst_paths = read_dataset('./data/list/train_gt.txt', num_samples=10)

    train_dataset = SegmentationDataset(train_img_paths, train_inst_paths)
    val_dataset = SegmentationDataset(val_img_paths, val_inst_paths)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

    print(train_loader)
    for imgs, bins, insts in train_loader:
        print("Batch of Images Shape:", imgs.shape)
        print("Batch of Binary Labels Shape: ", bins.shape)
        print("Batch of Instance Labels Shape:", insts.shape)
        break