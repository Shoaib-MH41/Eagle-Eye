import numpy as np

try:
    from jnius import autoclass, cast
    # Core Android imports for Camera2 API
    Context = autoclass('android.content.Context')
    CameraManager = autoclass('android.hardware.camera2.CameraManager')
    CameraCharacteristics = autoclass('android.hardware.camera2.CameraCharacteristics')
    CaptureRequest = autoclass('android.hardware.camera2.CaptureRequest')
    CameraDevice = autoclass('android.hardware.camera2.CameraDevice')
    ImageReader = autoclass('android.media.ImageReader')
    ImageFormat = autoclass('android.graphics.ImageFormat')
    Handler = autoclass('android.os.Handler')
    Looper = autoclass('android.os.Looper')
    JNIUS_AVAILABLE = True
except ImportError:
    print("jnius not found. Running with mocked Android classes for testing.")
    JNIUS_AVAILABLE = False
    class MockClass:
        def __getattr__(self, item):
            return MockClass()
        def __call__(self, *args, **kwargs):
            return MockClass()

    Context = MockClass()
    CameraManager = MockClass()
    CameraCharacteristics = MockClass()
    CaptureRequest = MockClass()
    CameraDevice = MockClass()
    ImageReader = MockClass()
    ImageFormat = MockClass()
    ImageFormat.RAW_SENSOR = 32
    Handler = MockClass()
    Looper = MockClass()

class Camera2Controller:
    """
    Interfaces with the Android Camera2 API to provide manual control and RAW capture.
    """
    def __init__(self, context=None):
        self.context = context
        self.camera_manager = None
        self.camera_device = None
        self.image_reader = None

        # Manual control parameters
        self.iso = 100
        self.exposure_time_ns = 10000000  # 10ms in nanoseconds
        self.focus_distance = 0.0 # Infinity focus

        if self.context and JNIUS_AVAILABLE:
            self.camera_manager = self.context.getSystemService(Context.CAMERA_SERVICE)

    def open_camera(self, camera_id="0"):
        """Opens the specified camera."""
        if not JNIUS_AVAILABLE:
            print(f"Mock: Opening camera {camera_id}")
            self.camera_device = True
            return

        print(f"Opening camera {camera_id} via CameraManager...")

        from jnius import PythonJavaClass, java_method

        class CameraStateCallback(PythonJavaClass):
            __javainterfaces__ = ['android/hardware/camera2/CameraDevice$StateCallback']

            def __init__(self, controller):
                super(CameraStateCallback, self).__init__()
                self.controller = controller

            @java_method('(Landroid/hardware/camera2/CameraDevice;)V')
            def onOpened(self, camera):
                print("Camera opened successfully.")
                self.controller.camera_device = camera

            @java_method('(Landroid/hardware/camera2/CameraDevice;)V')
            def onDisconnected(self, camera):
                print("Camera disconnected.")
                camera.close()
                self.controller.camera_device = None

            @java_method('(Landroid/hardware/camera2/CameraDevice;I)V')
            def onError(self, camera, error):
                print(f"Camera error: {error}")
                camera.close()
                self.controller.camera_device = None

        self.state_callback = CameraStateCallback(self)
        # Note: In a real app with permissions handled, we pass the callback
        self.camera_manager.openCamera(camera_id, self.state_callback, None)

    def setup_image_reader(self, width, height):
        """Sets up the ImageReader to receive RAW_SENSOR output."""
        print(f"Setting up ImageReader for {width}x{height} RAW_SENSOR format")
        if JNIUS_AVAILABLE:
            # Maximum 2 images in the queue
            self.image_reader = ImageReader.newInstance(width, height, ImageFormat.RAW_SENSOR, 2)

            from jnius import PythonJavaClass, java_method
            class ImageAvailableListener(PythonJavaClass):
                __javainterfaces__ = ['android/media/ImageReader$OnImageAvailableListener']

                @java_method('(Landroid/media/ImageReader;)V')
                def onImageAvailable(self, reader):
                    print("New image available in ImageReader.")

            self.image_listener = ImageAvailableListener()
            self.image_reader.setOnImageAvailableListener(self.image_listener, None)

            # Setup capture session
            if self.camera_device:
                surface_list = autoclass('java.util.ArrayList')()
                surface_list.add(self.image_reader.getSurface())

                class SessionStateCallback(PythonJavaClass):
                    __javainterfaces__ = ['android/hardware/camera2/CameraCaptureSession$StateCallback']

                    def __init__(self, controller):
                        super(SessionStateCallback, self).__init__()
                        self.controller = controller

                    @java_method('(Landroid/hardware/camera2/CameraCaptureSession;)V')
                    def onConfigured(self, session):
                        print("Capture session configured.")
                        self.controller.capture_session = session

                    @java_method('(Landroid/hardware/camera2/CameraCaptureSession;)V')
                    def onConfigureFailed(self, session):
                        print("Capture session configuration failed.")

                self.session_callback = SessionStateCallback(self)
                self.camera_device.createCaptureSession(surface_list, self.session_callback, None)

    def set_manual_controls(self, iso: int, exposure_time_ns: int, focus_distance: float):
        """
        Set manual parameters for the next capture request.
        :param iso: Sensor sensitivity (e.g., 100, 200, 400, 800)
        :param exposure_time_ns: Exposure time in nanoseconds
        :param focus_distance: Lens focus distance in diopters (0.0 is infinity)
        """
        self.iso = iso
        self.exposure_time_ns = exposure_time_ns
        self.focus_distance = focus_distance
        print(f"Manual controls configured -> ISO: {self.iso}, Exposure: {self.exposure_time_ns} ns, Focus: {self.focus_distance}")

    def create_capture_request(self):
        """Builds a CaptureRequest with manual control overrides."""
        if not self.camera_device or not JNIUS_AVAILABLE:
            print("Mock: Creating manual capture request")
            return None

        # Create a still capture request
        request_builder = self.camera_device.createCaptureRequest(CameraDevice.TEMPLATE_STILL_CAPTURE)

        # Disable automatic modes
        request_builder.set(CaptureRequest.CONTROL_AE_MODE, CaptureRequest.CONTROL_AE_MODE_OFF)
        request_builder.set(CaptureRequest.CONTROL_AF_MODE, CaptureRequest.CONTROL_AF_MODE_OFF)
        request_builder.set(CaptureRequest.CONTROL_AWB_MODE, CaptureRequest.CONTROL_AWB_MODE_OFF)

        # Apply manual settings
        request_builder.set(CaptureRequest.SENSOR_SENSITIVITY, self.iso)
        request_builder.set(CaptureRequest.SENSOR_EXPOSURE_TIME, self.exposure_time_ns)
        request_builder.set(CaptureRequest.LENS_FOCUS_DISTANCE, self.focus_distance)

        if self.image_reader:
            request_builder.addTarget(self.image_reader.getSurface())

        return request_builder.build()

    def capture_raw_image(self) -> np.ndarray:
        """
        Captures a frame and converts the RAW_SENSOR Android Image to a NumPy array.
        """
        print("Capturing RAW image...")

        if not JNIUS_AVAILABLE:
            print("Mock: Returning dummy RAW numpy array (1080p, 16-bit)")
            # Create a simple image with some features so ECC can converge during tests
            dummy_image = np.zeros((1080, 1920), dtype=np.uint16)
            # Add a white square in the middle
            dummy_image[400:600, 800:1100] = 60000
            # Add some random noise
            dummy_image += np.random.randint(0, 500, (1080, 1920), dtype=np.uint16)
            return dummy_image

        if not hasattr(self, 'capture_session') or self.capture_session is None:
            print("Capture session not configured yet.")
            return None

        # Dispatch capture request
        request = self.create_capture_request()
        if request:
            self.capture_session.capture(request, None, None)

        # In a real environment, we would acquire the latest image from the ImageReader
        # This normally happens asynchronously in the onImageAvailable callback,
        # but for demonstration we show the buffer extraction logic here.
        image = self.image_reader.acquireLatestImage()
        if image is None:
            return None

        plane = image.getPlanes()[0]
        buffer = plane.getBuffer()
        capacity = buffer.capacity()

        # Copy the ByteBuffer content into a bytearray
        buffer_array = bytearray(capacity)
        buffer.get(buffer_array)

        # Assuming 16-bit depth for RAW_SENSOR data
        raw_data = np.frombuffer(buffer_array, dtype=np.uint16)

        # Reshape array based on image dimensions
        width = image.getWidth()
        height = image.getHeight()
        raw_data = raw_data.reshape((height, width))

        image.close()
        return raw_data
