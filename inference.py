import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import torch
import torch.nn.functional as F
from torchvision import transforms
from bisenetv2 import BiseNetV2
from PIL import Image
from sam import Sam
from datetime import datetime
# from scipy import ndimage as ndi


def img_preprocess(image_path, shape=(256, 512)):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize(shape),
        transforms.ToTensor()
    ])
    img = transform(image).to(torch.device("cuda"))
    img = torch.unsqueeze(img, dim=0)
    return img

def gen_mask(ins_img):
    mask = []
    for i, mask_i in enumerate(ins_img):
        binarized = mask_i * (i + 1)
        mask.append(binarized)
    mask = np.sum(np.stack(mask, axis=0), axis=0).astype(np.uint8)
    return mask

def coloring(mask):
    ins_color_img = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    n_ins = len(np.unique(mask)) - 1
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, n_ins)]
    for i in range(n_ins):
        ins_color_img[mask == i + 1] =\
            (np.array(colors[i][:3]) * 255).astype(np.uint8)
    return ins_color_img


# def gen_instance_mask(sem_pred, ins_pred, n_obj):
#     print(ins_pred.shape)
#     print(sem_pred.shape)
#     print(ins_pred[:, sem_pred].shape)
#     embeddings = ins_pred[:, sem_pred].transpose(1, 0).detach().cpu().numpy()
#     print(embeddings.shape)
#     clustering = KMeans(n_obj).fit(embeddings)
#     labels = clustering.labels_

#     instance_mask = np.zeros_like(sem_pred, dtype=np.uint8)
#     for i in range(n_obj):
#         lbl = np.zeros_like(labels, dtype=np.uint8)
#         lbl[labels == i] = i + 1
#         instance_mask[sem_pred] += lbl

#     return instance_mask

def gen_instance_mask(sem_pred, ins_pred, n_obj):

    sem_pred = sem_pred.to(torch.bool).detach().cpu().numpy()
    ins_pred = ins_pred.detach().cpu().numpy()

    # print("Instance Prediction Shape:", ins_pred.shape)  # (4, 512, 256)
    # print("Semantic Prediction Shape:", sem_pred.shape)  # (512, 256)


    C, H, W = ins_pred.shape
    ins_pred = ins_pred.reshape(C, -1)  # (4, 131072)

    embeddings = ins_pred[:, sem_pred.reshape(-1)].T  

    # print("Extracted Embeddings Shape:", embeddings.shape) 

    if embeddings.shape[0] < n_obj:
        return None
        raise ValueError(f"Not enough valid pixels ({embeddings.shape[0]}) for {n_obj} clusters.")

    clustering = KMeans(n_clusters=n_obj, random_state=42).fit(embeddings)
    labels = clustering.labels_

    instance_mask = np.zeros((H, W), dtype=np.uint8)

    # print(instance_mask.shape, sem_pred.shape)

    instance_mask[sem_pred] = labels + 1  
    return instance_mask


def gen_color_img(sem_pred, ins_pred, n_obj):
    return coloring(gen_instance_mask(sem_pred, ins_pred, n_obj))


def inference_bisenetv2(model, img_path, max_num_lanes=4):
    input_img = img_preprocess(img_path)
    t0 = datetime.now()
    
    with torch.no_grad():
        bin_pred, inst_pred = model(input_img)
    t1 = datetime.now()

    print(1000000/(t1.microsecond-t0.microsecond))
    
    bin_pred = bin_pred.detach().cpu()
    bin_pred = torch.argmax(bin_pred, dim=1, keepdim=True).squeeze()
    inst_pred = inst_pred.squeeze()
    # print(inst_pred.shape)
    

    lane_img = gen_color_img(bin_pred, inst_pred, max_num_lanes)
    input_img = input_img.squeeze().permute(1, 2, 0).cpu().numpy()
    
    return input_img, bin_pred, lane_img


class Cluster:
    def __init__(self):
        xm = torch.linspace(0, 2, 2048).view(1, 1, -1).expand(1, 1024, 2048)
        ym = torch.linspace(0, 1, 1024).view(1, -1, 1).expand(1, 1024, 2048)
        xym = torch.cat((xm, ym), 0)

        self.xym = xym.cuda()

    def cluster(self, prediction, binary_mask=None, n_sigma=1, dist_th=0.98):
        def show_heatmap(dist, title="Gaussian Distance"):
            H, W = mask.shape[-2], mask.shape[-1]
            dist_2d = torch.zeros(H, W, dtype=torch.float32)

            # Flatten everything and apply dist to where mask is True
            mask_flat = mask.view(-1)
            dist_flat = dist.view(-1)
            dist_2d.view(-1)[mask_flat.bool().cpu()] = dist_flat.cpu()

            # Convert to numpy
            dist_np = dist_2d.numpy()

            # Plot
            # import matplotlib.pyplot as plt
            plt.figure(figsize=(6, 4))
            plt.imshow(dist_np, cmap='hot')
            plt.colorbar()
            plt.title(title)
            plt.axis("off")
            plt.show()

        height, width = prediction.shape[1], prediction.shape[2]
        xym_s = self.xym[:, 0:height, 0:width]
        
        spatial_emb = torch.tanh(prediction[0:2]) + xym_s  # 2 x h x w
        sigma = prediction[2:2+n_sigma]  # n_sigma x h x w
        seed_map = torch.sigmoid(prediction[2+n_sigma:2+n_sigma + 1])  # 1 x h x w
        # print(seed_map.shape)
       
        instance_map = torch.zeros(height, width, dtype=torch.uint8, device='cuda')
        instances = []

        seed_thresh_mask = seed_map.bool()

        if binary_mask is not None:
            binary_mask = binary_mask.bool().unsqueeze(0)  # (1, H, W)
            mask = seed_thresh_mask & binary_mask  # combine
            # print(binary_mask.shape)
        else:
            mask = seed_thresh_mask

        # print(mask.sum())

        if mask.sum() > 128:
            spatial_emb_masked = spatial_emb[mask.expand_as(spatial_emb)].view(2, -1)
            sigma_masked = sigma[mask.expand_as(sigma)].view(n_sigma, -1)
            seed_map_masked = seed_map[mask].view(1, -1)

            unclustered = torch.ones(mask.sum(), dtype=torch.uint8, device=prediction.device)
            instance_map_masked = torch.zeros(mask.sum(), dtype=torch.uint8, device=prediction.device)

            count = 1
            while unclustered.sum() > 0:
                weighted_seed_map = seed_map_masked * unclustered.float()
                seed = weighted_seed_map.argmax().item()
                seed_score = weighted_seed_map.max().item()
                # print(seed)

                if seed_score < 0:
                    break

                center = spatial_emb_masked[:, seed:seed+1]
                s = torch.exp(sigma_masked[:, seed:seed+1] * 10)
                diff = spatial_emb_masked - center
                dist = torch.exp(-1 * torch.sum((diff**2) * s, dim=0, keepdim=True))  # (1, N)
                # print(torch.unique(dist))
                
                dist_threshold = torch.quantile(dist, 0.80)
                # dist_threshold = dist.mean() + 0.3 * dist.std()
                # print(dist.mean(), dist.std(), dist_threshold)
                # show_heatmap(dist, title=f"Gaussian Distance (Seed {count})")

                proposal = (dist >= dist_threshold).squeeze()

                if proposal.sum() > 128:
                    if unclustered[proposal].sum().float() / proposal.sum().float() > 0.5:
                        instance_map_masked[proposal] = count
                        instance_mask = torch.zeros(height, width, dtype=torch.uint8)
                        instance_mask[mask.squeeze().cpu()] = proposal.cpu().byte()
                        instances.append({'mask': instance_mask * 255, 'score': seed_score})
                        count += 1

                unclustered[proposal] = 0

            instance_map[mask.squeeze()] = instance_map_masked

        return instance_map, instances







        # count = 1
        # mask = seed_map > 0.5

        # if mask.sum() > 128:

        #     spatial_emb_masked = spatial_emb[mask.expand_as(spatial_emb)].view(2, -1)
        #     sigma_masked = sigma[mask.expand_as(sigma)].view(n_sigma, -1)
        #     seed_map_masked = seed_map[mask].view(1, -1)

        #     unclustered = torch.ones(mask.sum()).byte().cuda()
        #     instance_map_masked = torch.zeros(mask.sum()).byte().cuda()

        #     while(unclustered.sum() > 128):

        #         seed = (seed_map_masked * unclustered.float()).argmax().item()
        #         seed_score = (seed_map_masked * unclustered.float()).max().item()
        #         if seed_score < threshold:
        #             break
        #         center = spatial_emb_masked[:, seed:seed+1]
        #         unclustered[seed] = 0
        #         s = torch.exp(sigma_masked[:, seed:seed+1]*10)
        #         dist = torch.exp(-1*torch.sum(torch.pow(spatial_emb_masked -
        #                                                 center, 2)*s, 0, keepdim=True))

        #         proposal = (dist > 0.5).squeeze()

        #         if proposal.sum() > 128:
        #             if unclustered[proposal].sum().float()/proposal.sum().float() > 0.5:
        #                 instance_map_masked[proposal.squeeze()] = count
        #                 instance_mask = torch.zeros(height, width).byte()
        #                 instance_mask[mask.squeeze().cpu()] = proposal.cpu()
        #                 instances.append(
        #                     {'mask': instance_mask.squeeze()*255, 'score': seed_score})
        #                 count += 1

        #         unclustered[proposal] = 0

        #     instance_map[mask.squeeze().cpu()] = instance_map_masked.cpu()

        # return instance_map, instances


def inference_spatial_embed(model, img_path, n_sigma=2, threshold=0.5, dist_th=0.98):
    input_img = img_preprocess(img_path)
    with torch.no_grad():
        bin_pred, inst_pred = model(input_img)
    
    # bin_pred = bin_pred.detach().cpu()
    bin_pred = torch.argmax(bin_pred, dim=1, keepdim=True).squeeze()
    inst_pred = inst_pred.squeeze()

    cluster = Cluster()
    # print(inst_pred.shape)
    instance_map, instance = cluster.cluster(inst_pred, binary_mask=bin_pred, n_sigma=2, dist_th=dist_th)
    instance_map = instance_map
    # instance_mask = instance[1]['mask']
    # print(instance_map.shape, torch.unique(instance_map))
    # print(predictions)
    # sigma_x = inst_pred[0
    # seed_map = torch.sigmoid(inst_pred[4])

    input_img = input_img.squeeze().permute(1, 2, 0).cpu().numpy()
    bin_pred = bin_pred.detach().cpu()

    return input_img, bin_pred, coloring(instance_map.detach().cpu())











    # spatial_emb = inst_pred[0:2]
    # sigma = inst_pred[2:4]
    # seed_map = torch.sigmoid(inst_pred[4])
    # print(np.unique(seed_map))

    # seed_points = (seed_map>threshold) & bin_pred

    # H, W = spatial_emb.shape[1], spatial_emb.shape[2]
    # embedding = spatial_emb.view(2, -1).T  # (H*W, 2)
    # sigma = torch.exp(sigma * 10).view(2, -1).T  # (H*W, 2)

    # seed_coords = seed_points.nonzero(as_tuple=False)    

    # instance_map = torch.zeros((H, W), dtype=torch.int32)

    # instance_id = 1
    # for coord in seed_coords:
    #     y, x = coord
    #     center = spatial_emb[:, y, x].view(1, 2)  # (1, 2)
    #     s = sigma[y * W + x].view(1, 2)  # (1, 2)

    #     # Gaussian score
    #     diff = embedding - center  # (H*W, 2)
    #     dist = torch.exp(-torch.sum((diff ** 2) * s, dim=1))  # (H*W,)

    #     mask = (dist > 0.2).view(H, W) & bin_pred  # You can adjust 0.5
    #     instance_map[mask] = instance_id
    #     instance_id += 1

    # # print(instance_map.shape)
    # print(np.unique(instance_map))

    
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model_path = "./train_out/UNet-Discloss/epoch032-loss6.30.pt"
    # model = BiseNetV2(out_channels=4).to(device)
    model = Sam(num_classes=4).to(device)
    checkpoint = torch.load(model_path, weights_only=False)
    print(checkpoint.keys())
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    image_path = "./data/driver_23_30frame/05151649_0422.MP4/00180.jpg"

    # t0 = datetime.now()
    input_img, bin_pred, lane_img = inference_bisenetv2(model, image_path)
    # t1 = datetime.now()

    # print(1000000/(t1.microsecond-t0.microsecond))
    # input_img, bin_pred, lane_img = inference_spatial_embed(model, image_path, n_sigma=2, dist_th=0.9)

    fig, axes = plt.subplots(1,3, figsize=(15,5))
    axes[0].imshow(input_img)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    axes[1].imshow(bin_pred, cmap='gray')
    axes[1].set_title('Binary Lane Detection')
    axes[1].axis('off')

    axes[2].imshow(lane_img)
    axes[2].set_title('Semantic Lane Detection')
    axes[2].axis('off')

    plt.show()