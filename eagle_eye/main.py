import os
import sys

# Ensure modules can be imported if script is run directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from eagle_eye.camera.camera2_api import Camera2Controller
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

    # 4. Set Manual Controls
    # ISO 400, 1/100s exposure (10,000,000 ns), infinity focus
    camera.set_manual_controls(iso=400, exposure_time_ns=10000000, focus_distance=0.0)

    # 5. Create Capture Request
    request = camera.create_capture_request()

    # 6. Capture Image
    # This will return a NumPy array of the RAW data
    raw_array = camera.capture_raw_image()

    if raw_array is not None:
        print(f"Captured RAW Image Shape: {raw_array.shape}, Dtype: {raw_array.dtype}")
    else:
        print("Failed to capture image.")
        return

    # 7. Initialize and Configure Processing Pipeline
    print("\nSetting up AI Processing Pipeline...")
    processor = AIProcessor()

    # Add filters in desired order
    processor.add_filter(example_denoise_filter)
    processor.add_filter(example_demosaic_filter)
    processor.add_filter(example_color_correction_filter)

    # 8. Process Image
    final_image = processor.process(raw_array)

    print("\nProcessing complete!")
    if final_image is not None:
        print(f"Final Image Shape: {final_image.shape}, Dtype: {final_image.dtype}")

if __name__ == "__main__":
    main()
