import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, List, Tuple

# Import the new UNet from the lane detection codebase.
# Make sure that the file (e.g., unet.py) containing the improved UNet is in the PYTHONPATH.
from unet import UNet

class Sam(nn.Module):
    """
    SAM module re-implemented for lane detection using the improved UNet segmentation head.
    It takes an input image and returns a binary lane mask.
    """
    def __init__(
        self,
        num_classes: int = 1,
        img_size: int = 256,
        pixel_mean: List[float] = [123.675, 116.28, 103.53],
        pixel_std: List[float] = [58.395, 57.12, 57.375],
    ) -> None:
        super().__init__()
        self.img_size = img_size
        # Initialize the UNet segmentation backbone.
        # in_channels=3 for RGB input, out_channels=num_classes for binary segmentation.
        self.unet = UNet(in_channels=3, out_channels=num_classes)
        # Register normalization buffers.
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), persistent=False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), persistent=False)

    # def preprocess(self, x: torch.Tensor) -> torch.Tensor:
    #     """
    #     Normalize and pad the image to a square of size self.img_size.
    #     Assumes input x is a torch tensor with shape (3, H, W).
    #     """
    #     x = (x - self.pixel_mean) / self.pixel_std
    #     h, w = x.shape[-2:]
    #     padh = self.img_size - h
    #     padw = self.img_size - w
    #     # Pad the image on the right and bottom.
    #     x = F.pad(x, (0, padw, 0, padh))
    #     return x

    def forward(self, input_images) -> List[Dict[str, torch.Tensor]]:
        """
        Expects a list of input dictionaries. Each dictionary must contain:
          - "image": a tensor of shape (3, H, W)
          - "original_size": a tuple (H_orig, W_orig)
        Returns a list of dictionaries with keys:
          - "masks": the binary segmentation mask (bool tensor) of shape (1, H_orig, W_orig)
          - "logits": the raw output logits (float tensor) from the segmentation head
        """
        # Preprocess all images: normalize and pad.
        # input_images = torch.stack([self.preprocess(record["image"]) for record in batched_input], dim=0)
        # Forward pass through UNet; expected logits shape: (B, num_classes, self.img_size, self.img_size)
        logits_bin, logits_inst = self.unet(input_images)
        
        # outputs = []
        # for i, record in enumerate(batched_input):
        #     orig_size = record["original_size"]  # tuple (H_orig, W_orig)
        #     # Upsample logits to the original image size.
        #     upsampled_logits = F.interpolate(logits[i:i+1], size=orig_size, mode="bilinear", align_corners=False)
        #     # Apply sigmoid to convert logits to probabilities and threshold at 0.5 to get binary mask.
        #     mask = torch.sigmoid(upsampled_logits) > 0.5
        #     outputs.append({
        #         "masks": mask,           # binary mask, shape (1, H_orig, W_orig)
        #         "logits": upsampled_logits,
        #     })
        return logits_bin, logits_inst
    
if __name__ == "__main__":   
    model = Sam()
    input_tensor = torch.randn(8, 3, 512, 256)
    output = model(input_tensor)
    print(output[0].shape, output[1].shape)