import cv2
import numpy as np

def capture_aligned_burst(camera_controller, base_iso, base_exposure, focus_distance):
    """
    Captures a burst of 5 RAW frames with exposure bracketing and aligns them using OpenCV ECC.
    """
    multipliers = [0.25, 0.5, 1.0, 2.0, 4.0]
    frames = []

    for mult in multipliers:
        exposure = int(base_exposure * mult)
        camera_controller.set_manual_controls(iso=base_iso, exposure_time_ns=exposure, focus_distance=focus_distance)

        # Create and dispatch a new request with the updated controls
        camera_controller.create_capture_request()

        frame = camera_controller.capture_raw_image()
        if frame is not None:
            frames.append(frame)

    if not frames:
        return []

    # Use the 3rd frame (multiplier 1.0) as the reference frame
    reference_idx = min(2, len(frames) - 1)
    ref_frame = frames[reference_idx]

    aligned_frames = []

    # Preprocess reference frame for ECC
    def preprocess_for_ecc(img):
        # Normalize to 8-bit for OpenCV ECC
        # Handling the case where image might be all zeros (like in mock)
        min_val = img.min()
        max_val = img.max()
        if max_val == min_val:
            return np.zeros_like(img, dtype=np.uint8)
        normalized = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        return normalized.astype(np.uint8)

    ref_frame_8u = preprocess_for_ecc(ref_frame)

    # ECC Criteria
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 0.001)

    for i, frame in enumerate(frames):
        if i == reference_idx:
            aligned_frames.append(frame)
            continue

        frame_8u = preprocess_for_ecc(frame)
        warp_matrix = np.eye(2, 3, dtype=np.float32)

        try:
            # We use MOTION_TRANSLATION as hand shake is mostly translation
            _, warp_matrix = cv2.findTransformECC(
                ref_frame_8u,
                frame_8u,
                warp_matrix,
                cv2.MOTION_TRANSLATION,
                criteria,
                None,
                1
            )

            # Align the original 16-bit RAW frame to the reference frame using the inverse map
            aligned = cv2.warpAffine(
                frame,
                warp_matrix,
                (ref_frame.shape[1], ref_frame.shape[0]),
                flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP
            )
            aligned_frames.append(aligned)
        except cv2.error as e:
            print(f"ECC alignment failed for frame {i}: {e}. Keeping unaligned.")
            aligned_frames.append(frame)

    return aligned_frames
