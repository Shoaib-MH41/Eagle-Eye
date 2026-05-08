import cv2
import numpy as np

def mertens_exposure_fusion(frames):
    """
    Combines aligned frames using Mertens Exposure Fusion to create a high dynamic range image.

    Args:
        frames (list of np.ndarray): A list of aligned frames.

    Returns:
        np.ndarray: A 32-bit floating-point HDR image.
    """
    if not frames:
        return None

    print("Running Mertens Exposure Fusion...")

    # Initialize Mertens fusion algorithm
    merge_mertens = cv2.createMergeMertens()

    # Apply fusion
    fused_image = merge_mertens.process(frames)

    return fused_image
