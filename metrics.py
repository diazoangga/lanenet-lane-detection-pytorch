import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix
import numpy as np
from inference import gen_instance_mask, gen_mask, img_preprocess, Cluster

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
    
def generate_instance_masks_batch(bin_preds, ins_preds, max_num_instances=4, cluster_type='DiscLoss'):
    """
    Applies gen_instance_mask() to a batch of predictions.

    Args:
        bin_preds: Tensor of shape [B, H, W] (semantic / binary prediction)
        ins_preds: Tensor of shape [B, 4, H, W] (instance embedding)
        max_num_instances: maximum number of clusters to generate

    Returns:
        instance_masks: List of np.ndarray of shape [H, W] for each sample
    """
    B = bin_preds.shape[0]
    instance_masks = []

    for i in range(B):
        sem = bin_preds[i]
        ins = ins_preds[i]
        if cluster_type =='DiscLoss':
            # print('here')
            mask = gen_instance_mask(sem, ins, max_num_instances)
            if mask is None:
                return None
        else:
            # ins = ins.cpu().detach()
            # sem = sem.cpu().detach()
            cluster = Cluster()
            mask, _ = cluster.cluster(ins, binary_mask=sem)
        instance_masks.append(mask)

    return instance_masks

def compute_batch_metrics(instance_preds, gt_labels, max_instances=4, iou_threshold=0.5, cluster_type='DiscLoss'):
    """
    Compute instance segmentation metrics over a batch.

    Args:
        instance_preds: Tensor [B, 4, H, W] (model output)
        gt_labels: Tensor [B, H, W] (ground truth instance labels)
        max_instances: max lanes to cluster per image
        iou_threshold: IoU threshold for matching

    Returns:
        metrics_per_sample: list of dicts for each sample
    """
    B = gt_labels.shape[0]
    gt_labels = torch.squeeze(gt_labels, dim=1)
    binary_preds = (gt_labels > 0).to(torch.uint8)  # or use model prediction if available
    # print(gt_labels.shape, instance_preds.shape)

    instance_masks = generate_instance_masks_batch(binary_preds, instance_preds, max_instances, cluster_type=cluster_type)
    if instance_masks is None:
        return None
    gt_masks = gt_labels.detach().cpu().numpy()

    all_metrics = {"precision": 0,
        "recall": 0,
        "f1_score": 0,
        "mean_iou": 0,
        # "matched_pairs": 0,
        # "ious": 0
    }
    count = 0
    for pred_mask, gt_mask in zip(instance_masks, gt_masks):
        metrics = compute_instance_segmentation_metrics(pred_mask, gt_mask, iou_threshold)
        count +=1
        for key in metrics:
            # print(key)
            all_metrics[key] += metrics[key]

    
    for key in all_metrics:
        all_metrics[key]/=count


    return all_metrics


def compute_instance_segmentation_metrics(pred_mask, gt_mask, iou_threshold=0.5, max_instances=4):
    """
    Computes instance segmentation metrics between predicted and GT masks.

    Args:
        pred_mask (np.ndarray): (H, W), instance IDs. 0 = background
        gt_mask (np.ndarray): (H, W), instance IDs. 0 = background
        iou_threshold (float): threshold to consider a prediction a match
        max_instances (int): max number of GT instances (e.g., 4 for lanes)

    Returns:
        dict: precision, recall, IoU, mIoU, matched pairs, etc.
    """
   

    assert pred_mask.shape == gt_mask.shape
    if isinstance(pred_mask, torch.Tensor):
        if pred_mask.is_cuda:
            pred_mask = pred_mask.cpu().detach()

    pred_ids = np.unique(pred_mask)
    pred_ids = pred_ids[pred_ids != 0]

    gt_ids = np.unique(gt_mask)
    gt_ids = gt_ids[gt_ids != 0]

    iou_matrix = np.zeros((len(gt_ids), len(pred_ids)))
    # print(gt_ids, pred_ids)

    for i, gt_id in enumerate(gt_ids):
        gt_region = gt_mask == gt_id
        for j, pred_id in enumerate(pred_ids):
            pred_region = pred_mask == pred_id
            intersection = np.logical_and(gt_region, pred_region).sum()
            union = np.logical_or(gt_region, pred_region).sum()
            iou_matrix[i, j] = intersection / union if union > 0 else 0.0

    # Match predicted and GT instances using Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(-iou_matrix)  # maximize IoU

    matches = []
    ious = []
    tp = 0
    for r, c in zip(row_ind, col_ind):
        iou = iou_matrix[r, c]
        if iou >= iou_threshold:
            matches.append((gt_ids[r], pred_ids[c]))
            ious.append(iou)
            tp += 1

    fp = len(pred_ids) - tp
    fn = len(gt_ids) - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    mean_iou = np.mean(ious) if ious else 0

    return {
        "precision": precision,
        "recall": recall,
        "f1_score": 2 * precision * recall / (precision + recall + 1e-8),
        "mean_iou": mean_iou,
        # "true_positives": tp,
        # "false_positives": fp,
        # "false_negatives": fn,
        # "matched_pairs": matches,
        # "ious": ious
    }


if __name__ == '__main__':
    metrics = CalculateMetrics(num_classes=2)
    y_true = torch.randint(0, 3, (4, 1, 128, 128))
    y_pred = torch.randn(4, 2, 128, 128)
    results = metrics(y_true, y_pred)
    print(results)