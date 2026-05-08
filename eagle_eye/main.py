import os
import sys

# Ensure modules can be imported if script is run directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from eagle_eye.camera.camera2_api import Camera2Controller
from eagle_eye.camera.burst import capture_aligned_burst
from eagle_eye.processing.filters import AIProcessor, example_denoise_filter, example_demosaic_filter, example_color_correction_filter, example_super_resolution_filter
from eagle_eye.processing.fusion import mertens_exposure_fusion
import cv2
from PIL import Image
import piexif

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
    base_iso = 400
    base_exposure_ns = 10000000
    aligned_frames = capture_aligned_burst(camera, base_iso=base_iso, base_exposure=base_exposure_ns, focus_distance=0.0)

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
    processor.add_filter(example_super_resolution_filter)

    # 8. HDR Fusion and Processing
    print("\nFusing aligned frames...")
    fused_frame = mertens_exposure_fusion(aligned_frames)
    final_image = processor.process(fused_frame)

    print("\nProcessing complete!")
    if final_image is not None:
        print(f"Final Image Shape: {final_image.shape}, Dtype: {final_image.dtype}")

        # Save as Lossless PNG
        png_path = "output.png"
        cv2.imwrite(png_path, cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR))
        print(f"Saved lossless output to {png_path}")

        # Save as 100% Quality JPEG with EXIF
        jpg_path = "output.jpg"

        # Create EXIF data
        exif_dict = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}, "thumbnail": None}

        # EXIF tags (ISO, Exposure Time)
        # Exposure time is saved as a rational (numerator, denominator)
        exposure_sec = base_exposure_ns / 1e9
        # Approximate rational for 1/100s -> (1, 100)
        num, den = int(exposure_sec * 1000000), 1000000

        exif_dict["Exif"][piexif.ExifIFD.ISOSpeedRatings] = base_iso
        exif_dict["Exif"][piexif.ExifIFD.ExposureTime] = (num, den)

        exif_bytes = piexif.dump(exif_dict)

        # Save JPEG using PIL to inject EXIF
        pil_image = Image.fromarray(final_image)
        pil_image.save(jpg_path, "JPEG", quality=100, exif=exif_bytes)
        print(f"Saved JPEG output with EXIF to {jpg_path}")

if __name__ == "__main__":
    main()
