import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import torch
import torch.nn.functional as F
from torchvision import transforms
from bisenetv2 import BiseNetV2
from PIL import Image
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

    print("Instance Prediction Shape:", ins_pred.shape)  # (4, 512, 256)
    print("Semantic Prediction Shape:", sem_pred.shape)  # (512, 256)


    C, H, W = ins_pred.shape
    ins_pred = ins_pred.reshape(C, -1)  # (4, 131072)

    embeddings = ins_pred[:, sem_pred.reshape(-1)].T  

    print("Extracted Embeddings Shape:", embeddings.shape) 

    if embeddings.shape[0] < n_obj:
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
    with torch.no_grad():
        bin_pred, inst_pred = model(input_img)
    
    bin_pred = bin_pred.detach().cpu()
    bin_pred = torch.argmax(bin_pred, dim=1, keepdim=True).squeeze()
    inst_pred = inst_pred.squeeze()
    # print(inst_pred.shape)
    

    lane_img = gen_color_img(bin_pred, inst_pred, max_num_lanes)
    input_img = input_img.squeeze().permute(1, 2, 0).cpu().numpy()
    
    return input_img, bin_pred, lane_img

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model_path = "./train_out/20250312-224917/epoch059-loss5.40.pt"
model = BiseNetV2().to(device)
checkpoint = torch.load(model_path, weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

image_path = "./data/driver_23_30frame/05151649_0422.MP4/00180.jpg"
input_img, bin_pred, lane_img = inference_bisenetv2(model, image_path)

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