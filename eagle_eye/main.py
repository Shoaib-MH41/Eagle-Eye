import os
import sys

# Ensure modules can be imported if script is run directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from eagle_eye.camera.camera2_api import Camera2Controller
from eagle_eye.camera.burst import capture_aligned_burst
from eagle_eye.processing.filters import AIProcessor, demosaic_pipeline, super_resolution_filter
from eagle_eye.processing.fusion import mertens_exposure_fusion
from PIL import Image

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
    processor.add_filter(demosaic_pipeline)

    # 8. HDR Fusion and Processing
    print("\nFusing aligned frames...")
    fused_frame = mertens_exposure_fusion(aligned_frames)

    # Run the standard demosaic pipeline on the fused frame
    processed_image = processor.process(fused_frame)

    # Apply Super-Resolution on the final processed RGB image
    print("\nApplying Super-Resolution...")
    final_image = super_resolution_filter(processed_image)

    print("\nProcessing complete!")
    if final_image is not None:
        print(f"Final Image Shape: {final_image.shape}, Dtype: {final_image.dtype}")

        print("\nSaving final output...")
        img_pil = Image.fromarray(final_image)

        # Extract EXIF from original frames if available (mock arrays won't have it)
        exif_bytes = None
        if aligned_frames and hasattr(aligned_frames[0], 'info') and 'exif' in aligned_frames[0].info:
            exif_bytes = aligned_frames[0].info['exif']
        elif aligned_frames and hasattr(aligned_frames[0], 'getexif'):
            exif_bytes = aligned_frames[0].getexif().tobytes()
            if not exif_bytes:
                exif_bytes = None

        save_kwargs = {}
        if exif_bytes:
            save_kwargs['exif'] = exif_bytes

        # Save as Lossless PNG
        png_path = "output.png"
        img_pil.save(png_path, "PNG", **save_kwargs)
        print(f"Saved Lossless PNG to: {png_path}")

        # Save as 100% Quality JPEG
        jpg_path = "output.jpg"
        img_pil.save(jpg_path, "JPEG", quality=100, **save_kwargs)
        print(f"Saved 100% Quality JPEG to: {jpg_path}")

if __name__ == "__main__":
    main()
