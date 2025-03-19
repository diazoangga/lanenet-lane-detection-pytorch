import torch
import torch.nn as nn
import torch.nn.functional as F
from lovasz_losses import lovasz_hinge

def instance_loss(correct_label, prediction, feature_dim=4, delta_v=0.5, delta_d=3.0, param_var=1.0, param_dist=1.0, param_reg=0.001):
    """
    Computes the discriminative loss for instance segmentation based on the paper:
    "Semantic Instance Segmentation with a Discriminative Loss Function" (https://arxiv.org/abs/1708.02551)
    
    Arguments:
    - prediction: Tensor of shape (N, C, H, W) representing predicted embeddings (feature space)
    - correct_label: Tensor of shape (N, H, W) representing ground truth instance labels
    - feature_dim: Integer representing the number of feature dimensions (C)
    - delta_v: Distance threshold for intra-cluster compactness
    - delta_d: Distance threshold for inter-cluster separation
    - param_var: Weight for variance loss (pulling pixels together)
    - param_dist: Weight for distance loss (pushing clusters apart)
    - param_reg: Weight for regularization loss (stability of clusters)
    
    Returns:
    - Total loss combining variance, distance, and regularization terms
    """
    batch_size = prediction.shape[0]
    total_loss, total_var, total_dist, total_reg = 0.0, 0.0, 0.0, 0.0
    
    for i in range(batch_size):
        pred = prediction[i].permute(1, 2, 0).reshape(-1, feature_dim)
        labels = correct_label[i].reshape(-1)
        unique_labels = torch.unique(labels, sorted=True)
        
        if len(unique_labels) == 0:
            continue
        
        means = torch.stack([pred[labels == label].mean(dim=0) for label in unique_labels])
        
        # Variance Loss (L1): Intra-cluster compactness
        var_loss = torch.stack([
            torch.max(torch.norm(pred[labels == label] - mean, dim=1) - delta_v, torch.tensor(0.0, device=prediction.device)).pow(2).mean()
            for mean, label in zip(means, unique_labels)
        ]).mean()
        
        # Distance Loss (L2): Inter-cluster separation
        dist_loss = 0.0
        M = len(means)
        if M > 1:
            dist_matrix = torch.cdist(means, means, p=2)
            mask = torch.ones_like(dist_matrix, dtype=torch.bool)
            mask.fill_diagonal_(0)
            dist_loss = torch.sum(torch.max(delta_d - dist_matrix[mask], torch.tensor(0.0, device=prediction.device)).pow(2)) / (M * (M - 1) / 2)
        
        # Regularization Loss: Keeps cluster centers near the origin
        reg_loss = means.norm(dim=1).mean()
        
        total_loss += param_var * var_loss + param_dist * dist_loss + param_reg * reg_loss
        total_var += var_loss
        total_dist += dist_loss
        total_reg += reg_loss
    
    return total_loss / batch_size

def instance_loss_1(instance_label, net_out, delta_v=0.6, delta_d=6.0, param_var=1.0, param_dist=1.0, param_reg=0.001):
    num_instances = instance_label.max().item() + 1
    feature_dim = net_out.shape[1]
    
    if num_instances == 0:
        return torch.tensor(0.0, device=net_out.device)
    
    instance_labels = instance_label.view(-1)
    net_out = net_out.permute(0, 2, 3, 1).reshape(-1, feature_dim)
    
    unique_labels = torch.unique(instance_labels, sorted=True)
    
    means = torch.stack([net_out[instance_labels == label].mean(dim=0) for label in unique_labels])
    
    # L1 Loss: Intra-cluster compactness
    l1_loss = torch.stack([
        torch.max(torch.norm(net_out[instance_labels == label] - mean, dim=1) - delta_v, torch.tensor(0.0, device=net_out.device)).pow(2).mean()
        for mean, label in zip(means, unique_labels)
    ]).mean()
    
    # L2 Loss: Inter-cluster separation
    l2_loss = 0.0
    M = len(means)
    if M > 1:
        dist_matrix = torch.cdist(means, means, p=2)  # Compute Euclidean distance matrix
        mask = torch.ones_like(dist_matrix, dtype=torch.bool)
        mask.fill_diagonal_(0)
        l2_loss = torch.sum(torch.max(delta_d - dist_matrix[mask], torch.tensor(0.0, device=net_out.device)).pow(2)) / (M * (M - 1) / 2)
    
    # L_reg: Regularization to keep means compact
    l_reg = means.norm(dim=1).mean()
    
    return param_var * l1_loss + param_dist * l2_loss + param_reg * l_reg

class SpatialEmbLoss(nn.Module):

    def __init__(self, to_center=True, n_sigma=1, foreground_weight=1,):
        super().__init__()

        print('Created spatial emb loss function with: to_center: {}, n_sigma: {}, foreground_weight: {}'.format(
            to_center, n_sigma, foreground_weight))

        self.to_center = to_center
        self.n_sigma = n_sigma
        self.foreground_weight = foreground_weight

        # coordinate map
        xm = torch.linspace(0, 2, 2048).view(
            1, 1, -1).expand(1, 1024, 2048)
        ym = torch.linspace(0, 1, 1024).view(
            1, -1, 1).expand(1, 1024, 2048)
        xym = torch.cat((xm, ym), 0)

        self.register_buffer("xym", xym)

    def forward(self, prediction, instances, labels, w_inst=1, w_var=10, w_seed=1, iou=False, iou_meter=None):

        batch_size, height, width = prediction.size(
            0), prediction.size(2), prediction.size(3)
        
        instances = instances.squeeze(1)

        xym_s = self.xym[:, 0:height, 0:width].contiguous().to("cuda")  # 2 x h x w

        loss = 0

        for b in range(0, batch_size):

            spatial_emb = torch.tanh(prediction[b, 0:2]) + xym_s  # 2 x h x w
            sigma = prediction[b, 2:2+self.n_sigma]  # n_sigma x h x w
            seed_map = torch.sigmoid(
                prediction[b, 2+self.n_sigma:2+self.n_sigma + 1])  # 1 x h x w

            # loss accumulators
            var_loss = 0
            instance_loss = 0
            seed_loss = 0
            obj_count = 0

            instance = instances[b].unsqueeze(0)  # 1 x h x w
            label = labels[b].unsqueeze(0)  # 1 x h x w

            instance_ids = instance.unique()
            instance_ids = instance_ids[instance_ids != 0]
            

            # regress bg to zero
            bg_mask = label == 0
            if bg_mask.sum() > 0:
                seed_loss += torch.sum(
                    torch.pow(seed_map[bg_mask] - 0, 2))

            for id in instance_ids:

                in_mask = instance.eq(id)   # 1 x h x w

                # calculate center of attraction
                if self.to_center:
                    xy_in = xym_s[in_mask.expand_as(xym_s)].view(2, -1)
                    center = xy_in.mean(1).view(2, 1, 1)  # 2 x 1 x 1
                else:
                    center = spatial_emb[in_mask.expand_as(spatial_emb)].view(
                        2, -1).mean(1).view(2, 1, 1)  # 2 x 1 x 1

                # calculate sigma
                sigma_in = sigma[in_mask.expand_as(
                    sigma)].view(self.n_sigma, -1)

                s = sigma_in.mean(1).view(
                    self.n_sigma, 1, 1)   # n_sigma x 1 x 1

                # calculate var loss before exp
                var_loss = var_loss + \
                    torch.mean(
                        torch.pow(sigma_in - s[..., 0].detach(), 2))

                s = torch.exp(s*10)

                # calculate gaussian
                dist = torch.exp(-1*torch.sum(
                    torch.pow(spatial_emb - center, 2)*s, 0, keepdim=True))

                # apply lovasz-hinge loss
                instance_loss = instance_loss + \
                    lovasz_hinge(dist*2-1, in_mask)

                # seed loss
                seed_loss += self.foreground_weight * torch.sum(
                    torch.pow(seed_map[in_mask] - dist[in_mask].detach(), 2))

                obj_count += 1

            if obj_count > 0:
                instance_loss /= obj_count
                var_loss /= obj_count

            seed_loss = seed_loss / (height * width)

            loss += w_inst * instance_loss + w_var * var_loss + w_seed * seed_loss

        loss = loss / (b+1)

        return loss + prediction.sum()*0

if __name__ == "__main__":
    net_out = torch.randn(8, 4, 256, 512).to("cuda")  # Random tensor simulating network output
    instance_label = torch.randint(0,4, (8, 1, 256, 512)).to("cuda")/4  # Random labels simulating instance segmentation
    bin_label = torch.randn(8, 256,512).to("cuda")
    # loss_value = instance_loss(instance_label, net_out)
    # loss_value_1 = instance_loss_1(instance_label, net_out)
    loss = SpatialEmbLoss()
    l = loss(net_out, instance_label, bin_label)
    print("Instance Loss:", l.item())