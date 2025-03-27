import matplotlib.pyplot as plt
import numpy as np
import os
import yaml

def save_plot(history, save_dir):
    np.save(os.path.join(save_dir, 'history.npy'), history)

    train_loss_result = history['train_loss']
    val_loss_result = history['val_loss']
    train_ce_loss_result = history['train_ce_loss']
    val_ce_loss_result = history['val_ce_loss']
    train_inst_loss_result = history['train_inst_loss']
    val_inst_loss_result = history['val_inst_loss']
    train_acc_result = history['train_acc']
    val_acc_result = history['val_acc']
    train_iou_result = history['train_iou']
    val_iou_result = history['val_iou']
    train_dice_result = history['train_dice']
    val_dice_result = history['val_dice']

    x = list(range(len(train_loss_result)))

    save_fig_dir = os.path.join(save_dir, 'plot_fig')
    if not os.path.exists(save_fig_dir):
        os.makedirs(save_fig_dir)

    # Loss Plot
    plt.figure(figsize=(12, 4))
    plt.plot(x, train_loss_result, label='Train Loss')
    plt.plot(x, val_loss_result, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_fig_dir, 'loss.png'))
    plt.close()

    # CE Loss Plot
    plt.figure(figsize=(12, 4))
    plt.plot(x, train_ce_loss_result, label='Train CE Loss')
    plt.plot(x, val_ce_loss_result, label='Validation CE Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Cross-Entropy Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_fig_dir, 'ce_loss.png'))
    plt.close()

    # Instance Loss Plot
    plt.figure(figsize=(12, 4))
    plt.plot(x, train_inst_loss_result, label='Train Instance Loss')
    plt.plot(x, val_inst_loss_result, label='Validation Instance Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Instance Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_fig_dir, 'inst_loss.png'))
    plt.close()

    # Accuracy Plot
    plt.figure(figsize=(12, 4))
    plt.plot(x, train_acc_result, label='Train Accuracy')
    plt.plot(x, val_acc_result, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_fig_dir, 'accuracy.png'))
    plt.close()

    # IoU Plot
    plt.figure(figsize=(12, 4))
    plt.plot(x, train_iou_result, label='Train IoU')
    plt.plot(x, val_iou_result, label='Validation IoU')
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_fig_dir, 'iou.png'))
    plt.close()

    # Dice Plot
    plt.figure(figsize=(12, 4))
    plt.plot(x, train_dice_result, label='Train Dice')
    plt.plot(x, val_dice_result, label='Validation Dice')
    plt.xlabel('Epoch')
    plt.ylabel('Dice Score')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_fig_dir, 'dice.png'))
    plt.close()

def load_config(mode):
    config_files = {
        "train": "config_train.yaml",
        "test": "config_test.yaml",
        "inference": "config_inference.yaml",
    }

    if mode not in config_files:
        raise ValueError(f"Invalid mode: {mode}. Choose from {list(config_files.keys())}")

    config_path = config_files[mode]

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    return config

class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0
        self.best_model_state = None

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.best_model_state = model.state_dict()
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_model_state = model.state_dict()
            self.counter = 0

    def load_best_model(self, model):
        model.load_state_dict(self.best_model_state)

def process_seg_out(class_map):
    colored_img = np.zeros((128,128,3), dtype=np.uint8)

    colored_img[class_map == 0] = [0,0,0]
    colored_img[class_map == 1] = [128,0,0]
    colored_img[class_map == 2] = [0,128,0]

    return colored_img

def open_npy(npy_path):
    history = np.load(npy_path, allow_pickle=True).item()

    for key, values in history.items():
        print(f"{key}: {len(values)} entries")
        print(values[:5])

    return history

def convert_rtf_to_npy(rtf_path, ref_path, out_path):
    import re
    # import numpy as np

    # Load the keys from original history.npy
    original_history = np.load(ref_path, allow_pickle=True).item()
    keys = list(original_history.keys())

    # Read the RTF lines
    with open(rtf_path, 'r') as f:
        lines = f.readlines()

    # Regex patterns (same)
    train_pattern = re.compile(
        r"Train Loss: ([\d.]+), Train CE Loss: ([\d.]+), Train Inst Loss: ([\d.]+),Train IoU: ([\d.]+), Train Dice: ([\d.]+), Train Acc: ([\d.]+)"
    )
    val_pattern = re.compile(
        r"Val Loss: ([\d.]+), Val CE Loss: ([\d.]+), Val Inst Loss: ([\d.]+),Val IoU: ([\d.]+), Val Dice: ([\d.]+), Val Acc: ([\d.]+)"
    )

    # Initialize new history dictionary
    new_history = {key: [] for key in keys}

    # Iterate through lines and look for consecutive Train/Val pairs
    i = 0
    while i < len(lines) - 1:
        train_line = lines[i]
        val_line = lines[i + 1]

        train_match = train_pattern.search(train_line)
        val_match = val_pattern.search(val_line)

        if train_match and val_match:
            train_vals = list(map(float, train_match.groups()))
            val_vals = list(map(float, val_match.groups()))
            values = train_vals + val_vals
            for key, value in zip(keys, values):
                new_history[key].append(value)
            i += 2  # Skip to next pair
        else:
            i += 1  # Move to next line and try again

    # Preview a few entries
    for key in new_history:
        print(f"{key}: {new_history[key][:3]}")

    # Save it
    np.save(out_path, new_history)

def combine_npy(files, out_path):
    hist = []
    for file in files:
        temp = np.load(file, allow_pickle=True).item()
        hist.append(temp)
    merged = {}

    for key in hist[0].keys():
        print(key)
        merged[key] = []
    
    print(merged)
    for key in hist[0].keys():
        for i in range(len(hist)):
            print(hist[i][key])
            data_list = hist[i][key]
            for data in data_list:
                merged[key].append(data)
    

    np.save(out_path, merged)
    open_npy(out_path)

def manual_save_plot(history_file, save_dir):

    history = open_npy(history_file)


    train_loss_result = history['train_loss']
    val_loss_result = history['val_loss']
    train_ce_loss_result = history['train_ce_loss']
    val_ce_loss_result = history['val_ce_loss']
    train_inst_loss_result = history['train_inst_loss']
    val_inst_loss_result = history['val_inst_loss']
    train_acc_result = history['train_acc']
    val_acc_result = history['val_acc']
    train_iou_result = history['train_iou']
    val_iou_result = history['val_iou']
    train_dice_result = history['train_dice']
    val_dice_result = history['val_dice']

    x = list(range(len(train_loss_result)))

    save_fig_dir = os.path.join(save_dir, 'plot_fig')
    if not os.path.exists(save_fig_dir):
        os.makedirs(save_fig_dir)

    # Loss Plot
    plt.figure(figsize=(12, 4))
    plt.plot(x, train_loss_result, label='Train Loss')
    plt.plot(x, val_loss_result, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_fig_dir, 'loss.png'))
    plt.close()

    # CE Loss Plot
    plt.figure(figsize=(12, 4))
    plt.plot(x, train_ce_loss_result, label='Train CE Loss')
    plt.plot(x, val_ce_loss_result, label='Validation CE Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Cross-Entropy Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_fig_dir, 'ce_loss.png'))
    plt.close()

    # Instance Loss Plot
    plt.figure(figsize=(12, 4))
    plt.plot(x, train_inst_loss_result, label='Train Instance Loss')
    plt.plot(x, val_inst_loss_result, label='Validation Instance Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Instance Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_fig_dir, 'inst_loss.png'))
    plt.close()

    # Accuracy Plot
    plt.figure(figsize=(12, 4))
    plt.plot(x, train_acc_result, label='Train Accuracy')
    plt.plot(x, val_acc_result, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_fig_dir, 'accuracy.png'))
    plt.close()

    # IoU Plot
    plt.figure(figsize=(12, 4))
    plt.plot(x, train_iou_result, label='Train IoU')
    plt.plot(x, val_iou_result, label='Validation IoU')
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_fig_dir, 'iou.png'))
    plt.close()

    # Dice Plot
    plt.figure(figsize=(12, 4))
    plt.plot(x, train_dice_result, label='Train Dice')
    plt.plot(x, val_dice_result, label='Validation Dice')
    plt.xlabel('Epoch')
    plt.ylabel('Dice Score')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_fig_dir, 'dice.png'))
    plt.close()

def npy_to_txt(npy_path, out_path):
    history = np.load(npy_path, allow_pickle=True).item()

    # Save to a text file in the simple key: [list] format
    with open(out_path, "w") as f:
        for key, values in history.items():
            f.write(f"{key}: {values}\n")




if __name__ == '__main__':
    # open_npy('./train_out/UNetPP-Discloss/history.npy')
    # open_npy('./train_out/UNet-Discloss/history.npy')
    # convert_rtf_to_npy('./train_out/UNet-Discloss/history.rtf', './train_out/UNetPP-Discloss/history.npy', './train_out/UNet-Discloss/history.npy')
    # combine_npy(['./train_out/UNet-SpatialEmbed/1/history.npy',
    #              './train_out/UNet-SpatialEmbed/2/history.npy',
    #              './train_out/UNet-SpatialEmbed/3/history.npy',], './train_out/UNet-SpatialEmbed/history.npy',)

    manual_save_plot('./train_out/UNet-SpatialEmbed/history.npy', './train_out/UNet-SpatialEmbed')
    npy_to_txt('train_out/UNet-SpatialEmbed/history.npy',
               'train_out/UNet-SpatialEmbed/history.txt')