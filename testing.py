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
from inference import gen_color_img, gen_instance_mask, gen_mask
from loss_function import instance_loss, SpatialEmbLoss
from config_utils import Config
from metrics import CalculateMetrics, compute_batch_metrics
from tqdm import tqdm
from utils import save_plot
from sam import Sam

torch.manual_seed(120)
random.seed(120)
np.random.seed(120)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

date_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

CFG = Config(config_path='./lanenet.yml')
dataset_path = CFG.DATASET.TEST_FILE_LIST
num_samples = CFG.DATASET.MAX_NUM_SAMPLES
img_size = CFG.DATASET.IMAGE_SIZE
test_batch_size = CFG.TRAIN.VAL_BATCH_SIZE
loss_type = CFG.TRAIN.LOSS
model_path = CFG.MODEL.WEIGHT_PATH
model_arch = CFG.MODEL.MODEL_NAME

# if not os.path.exists(save_path):
#     os.makedirs(save_path)

## IMPORT ALL THE DATASETS
print('Importing the datasets with the following parameters...')
print('   Dataset path                    :', dataset_path)
print('   Max number of training data     :', num_samples)

_,_, val_img_paths, val_inst_paths = read_dataset(dataset_path, num_samples=num_samples, split_ratio=0.9)

print(val_img_paths)

test_dataset = SegmentationDataset(val_img_paths, val_inst_paths)

test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

print(f'\nImporting the datasets is completed')

# Model
if model_arch == 'BiSeNetV2':
    model = BiseNetV2(out_channels=5 if loss_type=='SpatialEmbed' else 4).to(device)
elif model_arch == 'SAM':
    model = Sam(num_classes=5 if loss_type=='SpatialEmbed' else 4).to(device)

checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint['model_state_dict'])
if loss_type == 'SpatialEmbed':
    criterion_disc = SpatialEmbLoss(n_sigma=2)
else:
    criterion_disc = instance_loss
criterion_ce = nn.CrossEntropyLoss().cuda()
criterion = [criterion_ce, criterion_disc]

metrics = CalculateMetrics(num_classes=2)
clustering_metrics = compute_batch_metrics


print('Model is built')

# history = {"val_loss": [], 
#            "val_ce_loss": [], 
#            "val_inst_loss": [],
#            "val_iou": [],
#            "val_acc": [],
#            "val_dice": [],}


# Testing loop
print("Starting Testing...")
count_skip_best_val = 0

test_bin_metrics_dict = {"iou": 0, "dice": 0, "accuracy": 0}
test_inst_metrics_dict = {"precision": 0,
            "recall": 0,
            "f1_score": 0,
            "mean_iou": 0,
            # "matched_pairs": 0,
            # "ious": 0
}
count = 0
count_inst = 0
model.eval()
total_loss = 0
total_ce_loss = 0
total_inst_loss = 0
test_pbar = tqdm(test_loader, leave=False)
for images, bin, inst in test_pbar:
    images, bin, inst = images.to(device), bin.to(device), inst.to(device)
    bin_pred, inst_pred = model(images)

    ce_loss = criterion[0](bin_pred, bin)
    if loss_type == 'SpatialEmbed':
        inst_loss = criterion[1](inst_pred, inst, bin)
    else:
        inst_loss = criterion[1](inst, inst_pred)

    loss = ce_loss + inst_loss

    test_bin_metrics = metrics(bin, bin_pred)

    total_loss += loss.item()
    total_ce_loss += ce_loss.item()
    total_inst_loss += inst_loss.item()
    # print(masks.shape, outputs.shape)
    for key in test_bin_metrics_dict:
        test_bin_metrics_dict[key] += test_bin_metrics[key]

    test_inst_metrics = clustering_metrics(inst_pred, inst, cluster_type=loss_type)
    if test_inst_metrics is None:
        pass
    else:
        for key in test_inst_metrics_dict:
            # print(key, test_inst_metrics[key])
            test_inst_metrics_dict[key] += test_inst_metrics[key]
        count_inst += 1
    count += 1
    # test_pbar.set_postfix(loss=loss.item(), ce_loss=ce_loss.item(), inst_loss=inst_loss.item(), iou=test_bin_metrics['iou'], dice=test_bin_metrics['dice'], acc=test_bin_metrics['accuracy'])

for key in test_bin_metrics_dict:
    test_bin_metrics_dict[key] /= count
for key in test_inst_metrics_dict:
    test_inst_metrics_dict[key] /= count_inst
avg_test_loss = total_loss/count
avg_test_ce_loss = total_ce_loss/count
avg_test_inst_loss = total_inst_loss/count


# history["train_loss"].append(avg_train_loss)
# history["val_loss"].append(avg_val_loss)
# history["train_ce_loss"].append(avg_train_ce_loss)
# history["val_ce_loss"].append(avg_val_ce_loss)
# history["train_inst_loss"].append(avg_train_inst_loss)
# history["val_inst_loss"].append(avg_val_inst_loss)
# history["train_acc"].append(train_metrics['accuracy'])
# history["val_acc"].append(val_metrics['accuracy'])
# history["train_iou"].append(train_metrics['iou'])
# history["val_iou"].append(val_metrics['iou'])
# history["train_dice"].append(train_metrics['dice'])
# history["val_dice"].append(val_metrics['dice'])

print(f"test loss: {avg_test_loss:.4f}, test ce loss: {avg_test_ce_loss:.4f}, test inst loss: {avg_test_inst_loss:.4f}")
print("test_bin_metrics_dict = {")
for k, v in test_bin_metrics_dict.items():
    print(f'    "{k}": {v:.4f},')
print("}")

print("test_inst_metrics_dict = {")
for k, v in test_inst_metrics_dict.items():
    if isinstance(v, float):
        print(f'    "{k}": {v:.4f},')
    else:
        print(f'    "{k}": {v},')
print("}")