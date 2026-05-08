import os
import sys

# Ensure modules can be imported if script is run directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from eagle_eye.camera.camera2_api import Camera2Controller
from eagle_eye.camera.burst import capture_aligned_burst
from eagle_eye.processing.filters import AIProcessor, example_denoise_filter, example_demosaic_filter, example_color_correction_filter

def main():
    print("Initializing Eagle Eye Camera System...")

    # 1. Initialize Camera Controller
    camera = Camera2Controller()

    # 2. Open Camera
    camera.open_camera(camera_id="0")

    # 3. Configure Output for RAW_SENSOR format
    # Example dimensions: 1920x1080 (Usually sensor dependent)
    camera.setup_image_reader(width=1920, height=1080)

    # 4. Capture Aligned Burst
    # ISO 400, 1/100s base exposure (10,000,000 ns), infinity focus
    aligned_frames = capture_aligned_burst(camera, base_iso=400, base_exposure=10000000, focus_distance=0.0)

    if aligned_frames and len(aligned_frames) > 0:
        print(f"Successfully captured {len(aligned_frames)} aligned RAW frames.")
        print(f"Shape of first frame: {aligned_frames[0].shape}, Dtype: {aligned_frames[0].dtype}")
    else:
        print("Failed to capture burst images.")
        return

    # 7. Initialize and Configure Processing Pipeline
    print("\nSetting up AI Processing Pipeline...")
    processor = AIProcessor()

    # Add filters in desired order
    processor.add_filter(example_denoise_filter)
    processor.add_filter(example_demosaic_filter)
    processor.add_filter(example_color_correction_filter)

    # 8. Process Image (using the reference frame for example)
    reference_frame = aligned_frames[2] if len(aligned_frames) > 2 else aligned_frames[0]
    final_image = processor.process(reference_frame)

    print("\nProcessing complete!")
    if final_image is not None:
        print(f"Final Image Shape: {final_image.shape}, Dtype: {final_image.dtype}")

if __name__ == "__main__":
    main()
