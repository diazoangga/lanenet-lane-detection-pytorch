import torch
import torch.nn as nn

class CalculateMetrics(nn.Module):
    def __init__(self, num_classes=3):
        super(CalculateMetrics, self).__init__()
        self.num_classes = num_classes
    
    def forward(self, y_true, y_pred, smooth=1e-7):
        y_true = y_true.squeeze(1)
        y_pred = torch.argmax(y_pred, dim=1)
        iou_per_class = []
        dice_per_class = []

        correct_pixels = torch.sum(y_pred == y_true).float()
        total_pixels = y_true.numel()
        accuracy = correct_pixels / total_pixels
        # print(y_pred.shape)

        for cls in range(self.num_classes):
            pred_cls = (y_pred == cls).float()
            target_cls = (y_true == cls).float()
            
            intersection = torch.sum(pred_cls * target_cls)
            union = torch.sum(pred_cls) + torch.sum(target_cls) - intersection
            
            dice = (2.0 * intersection + smooth) / (torch.sum(pred_cls) + torch.sum(target_cls) + smooth)
            iou = (intersection + smooth) / (union + smooth)
            
            iou_per_class.append(iou)
            dice_per_class.append(dice)
        
        mean_iou = torch.mean(torch.stack(iou_per_class))
        mean_dice = torch.mean(torch.stack(dice_per_class))
        
        return {"iou": mean_iou.item(), "dice": mean_dice.item(), "accuracy": accuracy.item()}

if __name__ == '__main__':
    metrics = CalculateMetrics(num_classes=2)
    y_true = torch.randint(0, 3, (4, 1, 128, 128))
    y_pred = torch.randn(4, 2, 128, 128)
    results = metrics(y_true, y_pred)
    print(results)