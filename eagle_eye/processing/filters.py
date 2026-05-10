import numpy as np
import cv2
import torch
import torch.nn as nn
import os

class ESPCN(nn.Module):
    """
    Lightweight Super-Resolution model (ESPCN) using PyTorch.
    """
    def __init__(self, scale_factor=2):
        super(ESPCN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5, padding=2)
        self.tanh = nn.Tanh()
        self.conv2 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 3 * (scale_factor ** 2), kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)

    def forward(self, x):
        x = self.tanh(self.conv1(x))
        x = self.tanh(self.conv2(x))
        x = self.pixel_shuffle(self.conv3(x))
        return x

# Instantiate the model globally to avoid re-initialization overhead per frame/call
_sr_model = ESPCN(scale_factor=2)
weights_path = os.path.join(os.path.dirname(__file__), 'espcn_dummy_weights.pth')
if os.path.exists(weights_path):
    _sr_model.load_state_dict(torch.load(weights_path))
else:
    print(f"Warning: weights file {weights_path} not found.")

_sr_model.eval()

def super_resolution_filter(image: np.ndarray) -> np.ndarray:
    """
    Runs the ESPCN model to double the resolution while sharpening details.
    """
    print("  -> Running Super-Resolution (ESPCN) Pipeline...")

    # The input image from the demosaic pipeline is an 8-bit RGB image.
    # Convert it to a float tensor and normalize to [0, 1].
    img_tensor = torch.from_numpy(image).float().permute(2, 0, 1).unsqueeze(0) / 255.0

    with torch.no_grad():
        out_tensor = _sr_model(img_tensor)

    # Convert back to a numpy array.
    out_img = out_tensor.squeeze().permute(1, 2, 0).numpy()
    out_img = np.clip(out_img * 255, 0, 255).astype(np.uint8)

    return out_img

class AIProcessor:
    """
    Modular structure for AI processing filters.
    """
    def __init__(self):
        self.filters = []

    def add_filter(self, filter_func):
        """
        Adds a processing filter to the pipeline.
        :param filter_func: A callable that takes a numpy array and returns a numpy array.
        """
        self.filters.append(filter_func)

    def process(self, raw_image: np.ndarray) -> np.ndarray:
        """
        Applies all registered filters to the image sequentially.
        """
        if raw_image is None:
            return None

        print("Starting processing pipeline...")
        processed_image = raw_image.copy()

        for idx, filter_func in enumerate(self.filters):
            print(f"Applying filter {idx + 1} / {len(self.filters)}")
            processed_image = filter_func(processed_image)

        return processed_image


def demosaic_pipeline(image: np.ndarray) -> np.ndarray:
    """
    Demosaicing processing pipeline:
    1. Demosaic Bayer to RGB
    2. White Balance based on the brightest neutral pixel
    3. Gamma Correction (1/2.2)
    4. Fast Non-Local Means Denoising
    """
    print("  -> Running Demosaic Pipeline...")

    # Ensure input is 2D for Bayer demosaicing
    if len(image.shape) > 2:
        image = image.squeeze()

    # Demosaic: convert 32-bit float Bayer to 16-bit uint for OpenCV demosaicing
    bayer_16 = np.clip(image * 65535, 0, 65535).astype(np.uint16)
    rgb_16 = cv2.cvtColor(bayer_16, cv2.COLOR_BayerBG2RGB)

    # Convert back to 32-bit float for processing
    rgb_float = rgb_16.astype(np.float32) / 65535.0

    # 1. White Balance based on brightest neutral pixel
    std_dev = np.std(rgb_float, axis=2)
    mean_val = np.mean(rgb_float, axis=2)

    # Mask to find neutral pixels (where color channels have low variance)
    neutral_mask = std_dev < 0.05
    if not np.any(neutral_mask):
        neutral_mask = std_dev < 0.1 # Relax threshold if none found

    if np.any(neutral_mask):
        # Among neutral pixels, find the brightest one
        brightest_idx = np.argmax(mean_val[neutral_mask])
        brightest_pixel = rgb_float[neutral_mask][brightest_idx]
    else:
        # Fallback to absolute brightest pixel if no neutral pixels found
        brightest_idx = np.argmax(mean_val)
        brightest_pixel = rgb_float.reshape(-1, 3)[brightest_idx]

    # Scale all channels based on the brightest pixel to perform White Balance
    scale_factors = np.max(brightest_pixel) / (brightest_pixel + 1e-6)
    rgb_wb = np.clip(rgb_float * scale_factors, 0, 1.0)

    # 2. Gamma Correction (1/2.2)
    rgb_gamma = np.power(rgb_wb, 1.0 / 2.2)

    # 3. Fast Non-Local Means Denoising
    # Convert to 8-bit for OpenCV denoising
    rgb_8bit = np.clip(rgb_gamma * 255, 0, 255).astype(np.uint8)

    # Apply fastNlMeansDenoisingColored
    rgb_denoised = cv2.fastNlMeansDenoisingColored(rgb_8bit, None, 10, 10, 7, 21)

    return rgb_denoised
