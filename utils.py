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