import os
import cv2
import time
import subprocess
import spdlog

logger = spdlog.ConsoleLogger('Camera')

CAMERA_PARAMS_DEFAULT = {
    'CAMERA_INDEX': 0,
    'WIDTH': 1280,
    'HEIGHT': 960,
    'EXPOSURE_VALUE': -7.0,
    'TEMPERATURE_VALUE': 6500.0,
    'FOCUS_VALUE': 398.0,
    'BRIGHTNESS_VALUE': 0.0,
    'CONTRAST_VALUE': 33.0,
}

class Camera:    
    def __init__(
        self, 
        image_save_dir: str = None,
        camera_params: dict = None,
        buffer_clear_frames: int = 3,
    ):
        assert image_save_dir is not None, "[Camera] image_save_dir is required"
        self.image_save_dir = image_save_dir
        os.makedirs(self.image_save_dir, exist_ok=True)

        self.cap = None
        self.camera_params = camera_params if camera_params is not None else CAMERA_PARAMS_DEFAULT
        self.video_device_path = f"/dev/video{self.camera_params['CAMERA_INDEX']}"
        self._setup_camera()

        self.buffer_clear_frames = buffer_clear_frames

    def _setup_camera(self):
        # driver setup
        self._set_camera_params_v4l2ctl()
        
        # open camera using V4L2 backend
        self.cap = cv2.VideoCapture(self.camera_params['CAMERA_INDEX'], cv2.CAP_V4L2)
        if not self.cap.isOpened():
            raise RuntimeError(f"[Warning] Failed to open camera index {self.camera_params['CAMERA_INDEX']}")
        
        try:
            # set buffer size (must be set before other settings)
            try:
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except:
                pass
            
            # try to set pixel format to MJPG (more efficient, reduce timeout)
            try:
                fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                self.cap.set(cv2.CAP_PROP_FOURCC, fourcc)
            except:
                # if MJPG is not supported, try YUYV
                try:
                    fourcc = cv2.VideoWriter_fourcc(*'YUYV')
                    self.cap.set(cv2.CAP_PROP_FOURCC, fourcc)
                except:
                    print("[Warning] Failed to set pixel format")
            
            # set resolution
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_params['WIDTH'])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_params['HEIGHT'])
            
            # verify resolution
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if actual_width != self.camera_params['WIDTH'] or actual_height != self.camera_params['HEIGHT']:
                print(f"[Warning] Resolution setting may have failed. Expected: {self.camera_params['WIDTH']}x{self.camera_params['HEIGHT']}, Actual: {actual_width}x{actual_height}")
            
            # clear buffer and warm up camera (critical step, solves timeout problem)
            for i in range(20):  # read and discard first 20 frames
                ret, _ = self.cap.read()
                if not ret:
                    time.sleep(0.05)
            
            logger.info("Camera parameters set successfully")
            
        except Exception as e:
            if self.cap:
                self.cap.release()
            raise RuntimeError(f"[Warning] Failed to setup camera: {e}")
    
    def _set_camera_params_v4l2ctl(self):
        """ set camera parameters using v4l2-ctl command line tool """
        if not os.path.exists(self.video_device_path):
            print(f"[Warning] Camera device {self.video_device_path} does not exist, skip v4l2-ctl setting")
            return
        try:
            # check if v4l2-ctl is available
            result = subprocess.run(['which', 'v4l2-ctl'], capture_output=True, text=True)
            if result.returncode != 0:
                print("[Warning] v4l2-ctl not found, skip parameter setting")
                return
                        
            def run_v4l2ctl(command_args, max_retries=3, retry_delay=1.0, param_name=""):
                """ run v4l2-ctl command and return if successful """
                for attempt in range(max_retries):
                    result = subprocess.run(
                        ['v4l2-ctl', '-d', self.video_device_path] + command_args,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        check=False
                    )
                    if result.returncode == 0:
                        if attempt > 0 and param_name:
                            print(f"[Info] {param_name}: {attempt + 1}th attempt successful")
                        return True
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                return False
            
            # set auto_exposure to manual mode (must be set before exposure_time_absolute)
            # auto_exposure: 0=Auto Mode, 1=Manual Mode, 2=Shutter Priority Mode, 3=Aperture Priority Mode
            success_count = 0
            if run_v4l2ctl(['-c', 'auto_exposure=1'], param_name="auto_exposure"):
                success_count += 1
            else:
                print(f"[Warning] Failed to set auto_exposure to manual mode (1) after retries")
            
            exposure_value = self.camera_params.get('EXPOSURE_VALUE')
            temp_value = self.camera_params.get('TEMPERATURE_VALUE')
            focus_value = self.camera_params.get('FOCUS_VALUE')
            brightness_value = self.camera_params.get('BRIGHTNESS_VALUE')
            contrast_value = self.camera_params.get('CONTRAST_VALUE')
            
            expected_params_count = 1  # auto_exposure is always set
            if exposure_value is not None: expected_params_count += 1
            if temp_value is not None: expected_params_count += 2  # white_balance_temperature + white_balance_automatic
            if focus_value is not None: expected_params_count += 2  # focus_absolute + focus_automatic_continuous
            if brightness_value is not None: expected_params_count += 1
            if contrast_value is not None: expected_params_count += 1
            
            if exposure_value is not None:
                exposure_abs = max(2, int(100 * (10 ** (exposure_value / 10))))
                exposure_abs = min(max(2, exposure_abs), 10000)
                if run_v4l2ctl(['-c', f'exposure_time_absolute={exposure_abs}'], param_name="exposure_time_absolute"): success_count += 1
                else: print(f"[Warning] Failed to set exposure_time_absolute: {exposure_abs} (from EXPOSURE_VALUE={exposure_value}) after retries")
            
            if temp_value is not None:
                if run_v4l2ctl(['-c', f'white_balance_temperature={int(temp_value)}'], param_name="white_balance_temperature"): success_count += 1
                else: print(f"[Warning] Failed to set white balance temperature: {temp_value} after retries")
                if run_v4l2ctl(['-c', 'white_balance_automatic=0'], param_name="white_balance_automatic"): success_count += 1
                else: print(f"[Warning] Failed to set white balance automatic: 0 after retries")
            
            if focus_value is not None:
                if run_v4l2ctl(['-c', f'focus_absolute={int(focus_value)}'], param_name="focus_absolute"): success_count += 1
                else: print(f"[Warning] Failed to set focus: {focus_value} after retries")
                if run_v4l2ctl(['-c', 'focus_automatic_continuous=0'], param_name="focus_automatic_continuous"): success_count += 1
                else: print(f"[Warning] Failed to set focus automatic continuous: 0 after retries")
            
            if brightness_value is not None:
                if run_v4l2ctl(['-c', f'brightness={int(brightness_value)}'], param_name="brightness"): success_count += 1
                else: print(f"[Warning] Failed to set brightness: {brightness_value} after retries")
            
            if contrast_value is not None:
                if run_v4l2ctl(['-c', f'contrast={int(contrast_value)}'], param_name="contrast"): success_count += 1
                else: print(f"[Warning] Failed to set contrast: {contrast_value} after retries")
            
            print(f"[Info] v4l2-ctl parameters set successfully ({success_count}/{expected_params_count} parameters set successfully)")
            
        except Exception as e:
            raise RuntimeError(f"[Warning] Failed to set parameters using v4l2-ctl: {e}")

    def get_frame(self):
        # Clear buffer before getting frame
        for _ in range(self.buffer_clear_frames):
            ret, _ = self.cap.read()
            if not ret:
                break

        if not self.cap.grab():
            # if grab fails, try multiple times
            for _ in range(5):
                if self.cap.grab():
                    break
                time.sleep(0.05)
            else:
                # if grab still fails, try read directly
                ret, frame = self.cap.read()
                if not ret:
                    raise RuntimeError("[Warning] Failed to grab frame")
                else:
                    timestamp = time.time()
                    return frame, timestamp
        
        # retrieve frame
        ret, frame = self.cap.retrieve()
        if not ret or frame is None:
            # if retrieve fails, try read directly
            ret, frame = self.cap.read()
            if not ret:
                for _ in range(5):
                    ret, frame = self.cap.read()
                    if ret and frame is not None:
                        break
                    time.sleep(0.1)
                if not ret or frame is None:
                    raise RuntimeError("[Warning] Failed to get frame")
        
        timestamp = time.time()
        return frame, timestamp
    
    def clear_buffer(self):
        """Clear camera buffer by reading and discarding frames"""
        for _ in range(self.buffer_clear_frames):
            ret, _ = self.cap.read()
            if not ret:
                break

    def release(self):
        if self.cap:
            self.cap.release()
            self.cap = None
        cv2.destroyAllWindows()