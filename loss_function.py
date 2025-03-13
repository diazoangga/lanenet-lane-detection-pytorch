import torch
import torch.nn as nn
import torch.nn.functional as F

def instance_loss(instance_label, net_out, delta_v=0.5, delta_d=3.0, param_var=1.0, param_dist=1.0, param_reg=0.001):
    num_instances = instance_label.max() + 1
    feature_dim = net_out.shape[1]
    
    if num_instances == 0:
        return torch.tensor(0.0, device=net_out.device)
    
    instance_labels = instance_label.view(-1)
    net_out = net_out.permute(0, 2, 3, 1).reshape(-1, feature_dim)
    
    unique_labels = torch.unique(instance_labels, sorted=True)
    
    means = torch.stack([net_out[instance_labels == label].mean(dim=0) for label in unique_labels])
    
    l_var = torch.stack([F.relu(torch.norm(net_out[instance_labels == label] - mean, dim=1) - delta_v).pow(2).mean() 
                         for mean, label in zip(means, unique_labels)]).mean()
    
    l_dist = 0.0
    if len(means) > 1:
        dist_matrix = torch.cdist(means, means)
        mask = torch.ones_like(dist_matrix, dtype=torch.bool)
        mask.fill_diagonal_(0)
        l_dist = torch.mean(F.relu(2.0 * delta_d - dist_matrix[mask]).pow(2))
    
    l_reg = means.norm(dim=1).mean()
    
    return param_var * l_var + param_dist * l_dist + param_reg * l_reg

if __name__ == "__main__":
    net_out = torch.randn(8, 3, 256, 512).to("cuda")  # Random tensor simulating network output
    instance_label = torch.randint(0, 5, (8, 256, 512)).to("cuda")  # Random labels simulating instance segmentation
    loss_value = instance_loss(instance_label, net_out)
    print("Instance Loss:", loss_value.item())