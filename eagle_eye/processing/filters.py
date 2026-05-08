import numpy as np
import cv2

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
