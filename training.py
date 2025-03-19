import os
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import datetime

from torch.utils.data import DataLoader
from torchvision import transforms
from dataset_utils import SegmentationDataset, read_dataset
from bisenetv2 import BiseNetV2
from loss_function import instance_loss, SpatialEmbLoss
from config_utils import Config
from metrics import CalculateMetrics
from tqdm import tqdm
from utils import save_plot

torch.manual_seed(120)
random.seed(120)
np.random.seed(120)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

date_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

CFG = Config(config_path='./lanenet.yml')
dataset_path = CFG.DATASET.TRAIN_FILE_LIST
num_samples = CFG.DATASET.MAX_NUM_SAMPLES
img_size = CFG.DATASET.IMAGE_SIZE
val_ratio = CFG.DATASET.VAL_RATIO
num_epochs = CFG.TRAIN.EPOCH_NUMS
batch_size = CFG.TRAIN.BATCH_SIZE
val_batch_size = CFG.TRAIN.VAL_BATCH_SIZE
learning_rate = CFG.SOLVER.LR
loss_type = CFG.TRAIN.LOSS
save_path = os.path.join(CFG.TRAIN.MODEL_SAVE_DIR, date_time)
continue_train = CFG.TRAIN.RESTORE_FROM_CHECKPOINT.ENABLE
model_path = CFG.TRAIN.RESTORE_FROM_CHECKPOINT.WEIGHT_PATH
init_epoch = CFG.TRAIN.RESTORE_FROM_CHECKPOINT.START_EPOCH

if not os.path.exists(save_path):
    os.makedirs(save_path)

## IMPORT ALL THE DATASETS
print('Importing the datasets with the following parameters...')
print('   Dataset path                    :', dataset_path)
print('   Max number of training data     :', num_samples)
print('   Val to all dataset ratio        :', val_ratio)
print('   Batch Size                      :', batch_size)

train_img_paths, train_inst_paths, val_img_paths, val_inst_paths = read_dataset(dataset_path, num_samples=num_samples, split_ratio=val_ratio)

train_dataset = SegmentationDataset(train_img_paths, train_inst_paths)
val_dataset = SegmentationDataset(val_img_paths, val_inst_paths)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

print(f'\nImporting the datasets is completed')

# Model
model = BiseNetV2().to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
if continue_train:
    checkpoint = torch.load(model_path, weight_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    criterion = checkpoint['loss']
    criterion_ce = criterion[0]
    criterion_disc = criterion[1]
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
else:
    if loss_type == 'SpatialEmbed':
        criterion_disc = SpatialEmbLoss()
    else:
        criterion_disc = instance_loss
    criterion_ce = nn.CrossEntropyLoss().cuda()
    criterion = [criterion_ce, criterion_disc]
metrics = CalculateMetrics(num_classes=2)

print('Model is built')

best_val_loss = float('inf')
history = {"train_loss": [], 
           "val_loss": [], 
           "train_ce_loss": [],
           "val_ce_loss": [], 
           "train_inst_loss": [],
           "val_inst_loss": [],
           "train_iou": [],
           "val_iou": [],
           "train_acc": [],
           "val_acc": [],
           "train_dice": [],
           "val_dice": [],}


# Training loop
print("Starting Training...")
count_skip_best_val = 0
for epoch in range(num_epochs):
    train_metrics = {"iou": 0, "dice": 0, "accuracy": 0}
    count = 0
    model.train()
    total_loss = 0
    total_ce_loss = 0
    total_inst_loss = 0
    train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
    for images, bin, inst in train_pbar:
        images, bin, inst = images.to(device), bin.to(device), inst.to(device)
        optimizer.zero_grad()
        bin_pred, inst_pred = model(images)
        # print(masks.shape, outputs.shape)
        # print(images.shape)
        # print(bin_pred.shape, inst_pred.shape)
        # print(bin.shape, inst.shape)
        ce_loss = criterion[0](bin_pred, bin)
        if loss_type == 'SpatialEmbed':
            inst_loss = criterion[1](inst_pred, inst, bin)
        else:
            inst_loss = criterion[1](inst, inst_pred)
        loss = ce_loss + inst_loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_ce_loss += ce_loss.item()
        total_inst_loss += inst_loss.item()
        # print(masks.shape, outputs.shape)
        batch_metrics = metrics(bin, bin_pred)
        for key in train_metrics:
            train_metrics[key] += batch_metrics[key]
        count += 1
        train_pbar.set_postfix(loss=loss.item(), ce_loss=ce_loss.item(), inst_loss=inst_loss.item(), iou=batch_metrics['iou'], dice=batch_metrics['dice'], acc=batch_metrics['accuracy'])
    
    for key in train_metrics:
        train_metrics[key] /= count
    avg_train_loss = total_loss/count
    avg_train_ce_loss = total_ce_loss/count
    avg_train_inst_loss = total_inst_loss/count

    model.eval()
    val_loss = 0
    val_ce_loss = 0
    val_inst_loss = 0
    val_metrics = {"iou": 0, "dice": 0, "accuracy": 0}
    count = 0

    with torch.no_grad():
        val_pbar = tqdm(val_loader, desc="Validating", leave=False)
        for images, bins, insts in val_pbar:
            images, bins, insts = images.to(device), bins.to(device), insts.to(device)
            bin_pred, inst_pred = model(images)
            ce_loss = criterion[0](bin_pred, bins)
            if loss_type == 'SpatialEmbed':
                inst_loss = criterion[1](inst_pred, inst, bin)
            else:
                inst_loss = criterion[1](inst, inst_pred)
            loss = ce_loss + inst_loss
            val_loss += loss.item()
            val_ce_loss += ce_loss.item()
            val_inst_loss += inst_loss.item()
            val_inst_loss += inst_loss.item()
            batch_val_metrics = metrics(bins, bin_pred)
            for key in val_metrics:
                val_metrics[key] += batch_metrics[key]
            count += 1
            val_pbar.set_postfix(validation_loss=loss.item(), 
                                val_ce_loss=ce_loss.item(),
                                val_inst_loss=inst_loss.item(),
                                iou=batch_metrics['iou'], 
                                dice=batch_metrics['dice'], 
                                acc=batch_metrics['accuracy'])

        for key in val_metrics:
            val_metrics[key] /= count
        avg_val_loss = val_loss/count
        avg_val_ce_loss = val_ce_loss/count
        avg_val_inst_loss = val_inst_loss/count

    
    history["train_loss"].append(avg_train_loss)
    history["val_loss"].append(avg_val_loss)
    history["train_ce_loss"].append(avg_train_ce_loss)
    history["val_ce_loss"].append(avg_val_ce_loss)
    history["train_inst_loss"].append(avg_train_inst_loss)
    history["val_inst_loss"].append(avg_val_inst_loss)
    history["train_acc"].append(train_metrics['accuracy'])
    history["val_acc"].append(val_metrics['accuracy'])
    history["train_iou"].append(train_metrics['iou'])
    history["val_iou"].append(val_metrics['iou'])
    history["train_dice"].append(train_metrics['dice'])
    history["val_dice"].append(val_metrics['dice'])

    print(f"Epoch {epoch+1}/{num_epochs}, \n"
      f"Train Loss: {avg_train_loss:.4f}, Train CE Loss: {avg_train_ce_loss:.4f}, Train Inst Loss: {avg_train_inst_loss:.4f}, "
      f"Train IoU: {train_metrics['iou']:.4f}, Train Dice: {train_metrics['dice']:.4f}, Train Acc: {train_metrics['accuracy']:.4f}\n"
      f"Val Loss: {avg_val_loss:.4f}, Val CE Loss: {avg_val_ce_loss:.4f}, Val Inst Loss: {avg_val_inst_loss:.4f}, "
      f"Val IoU: {val_metrics['iou']:.4f}, Val Dice: {val_metrics['dice']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}")
    
    save_plot(history, save_path)
    if avg_val_loss < best_val_loss:
        best_val_loss = val_loss
        model_save_path = os.path.join(save_path, f'epoch{epoch:03d}-loss{avg_val_loss:.2f}.pt')
        torch.save({'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict':optimizer.state_dict,
                     'loss': criterion},
                    model_save_path)
        print(f"Model improved and saved at {model_save_path}")
        count_skip_best_val = 0
    else:
        print('Model does not improve')
        count_skip_best_val += 1
    if count_skip_best_val >= 5:
        print(f'Early Stopping at Epoch: {epoch}')
        break
    
    # if early_stop_flag:
    #     early_stopping(val_loss, model)
    #     if early_stopping.early_stop:
    #         print(f'Early Stopping at Epoch: {epoch}')
    #         break
    
    # if decay_step is not None:
    #     schedule.step()
    
print("Training Complete.")
print(f"Best model saved at {model_save_path}")
