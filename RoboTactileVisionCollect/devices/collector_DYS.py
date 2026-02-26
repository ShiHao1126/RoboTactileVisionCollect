import os
import cv2
import time
import spdlog
import numpy as np
from typing import Union, List
from .camera import Camera
from .ftsensor import FTSensor

logger = spdlog.ConsoleLogger('Collector_DYS')


class Collector_DYS:
    def __init__(
        self,
        enable_camera: bool = True,
        enable_ft: bool = True,
        image_save_dir: str = None,
        image_format: str = 'png',
        ft_save_path: str = None,
        camera_params: dict = None,
        num_frames: int = 1,
        ft_data_header: list = [
            'image_name',  # image file name
            'timestamp',
            'fx', 'fy', 'fz',  # force (N)
            'mx', 'my', 'mz',  # torque (Nm)
            'x', 'y', 'z',     # position in contact center frame (mm)
            'depth',           # depth relative to contact face, keep positive (mm)
        ],
    ):
        self.enable_camera = enable_camera
        self.image_format = image_format
        self.num_frames = num_frames
        if self.enable_camera:
            self.camera = Camera(
                image_save_dir=image_save_dir,
                camera_params=camera_params,
            )
        else:
            self.camera = None

        self.enable_ft = enable_ft
        if self.enable_ft:
            self.ft_sensor = FTSensor(
                csv_save_path=ft_save_path,
                data_header=ft_data_header,
            )
        else:
            self.ft_sensor = None

        logger.info(f"Initialized | camera: {self.enable_camera}, ft: {self.enable_ft}")
    
    def save(
        self,
        image_name: str = None,
        ft_data: Union[List, np.ndarray] = None,
        xyz_mm: Union[List, np.ndarray] = None,
        depth_mm: float = None,
        delay: float = 0.3,
    ):
        if not self.enable_camera and not self.enable_ft:
            return

        if not self.enable_camera:
            image_name = 'None'

        if self.enable_camera:
            assert image_name is not None, "[Collector] image_name is required when camera is enabled"
        if self.enable_ft:
            assert ft_data is not None, "[Collector] ft_data is required when FT sensor is enabled"
            assert xyz_mm is not None, "[Collector] xyz_mm is required when FT sensor is enabled"
            assert depth_mm is not None, "[Collector] depth_mm is required when FT sensor is enabled"

        # for logging
        save_info_parts = []
        if self.enable_camera:
            save_info_parts.append(f"image: {image_name}")
        if self.enable_ft:
            ft_data_array = np.array(ft_data) if isinstance(ft_data, list) else ft_data
            save_info_parts.append(f"ft: [{ft_data_array[0]:.2f}, {ft_data_array[1]:.2f}, {ft_data_array[2]:.2f}]N")
            xyz_mm_array = np.array(xyz_mm) if isinstance(xyz_mm, list) else xyz_mm
            save_info_parts.append(f"point: [{xyz_mm_array[0]:.2f}, {xyz_mm_array[1]:.2f}, {xyz_mm_array[2]:.2f}]mm")
            save_info_parts.append(f"depth: {depth_mm:.2f}mm")
        
        logger.info(f"Save | {', '.join(save_info_parts)} | {self.num_frames} frames")

        def _save_one_frame(
                image_name: str = None, 
                ft_data: Union[List, np.ndarray] = None,
                xyz_mm: Union[List, np.ndarray] = None,
                depth_mm: float = None,
            ):
            timestamp = time.time()

            # save camera image
            if self.enable_camera:
                assert image_name is not None, "[Collector] image_name is required"
                frame, timestamp = self.camera.get_frame()
                image_path = os.path.join(self.camera.image_save_dir, f'{image_name}.{self.image_format}')
                cv2.imwrite(image_path, frame)
                logger.debug(f"Saved image: {image_path}")

            # save ft data
            if self.enable_ft:
                assert ft_data is not None, "[Collector] ft_data is required"
                assert xyz_mm is not None, "[Collector] xyz_mm is required"
                assert depth_mm is not None, "[Collector] depth_mm is required"
                
                ft_data_array = np.array(ft_data) if isinstance(ft_data, list) else ft_data
                xyz_mm_array = np.array(xyz_mm) if isinstance(xyz_mm, list) else xyz_mm
                
                assert ft_data_array.shape[0] == 6, "[Collector] ft_data must be 6D array"
                assert xyz_mm_array.shape[0] == 3, "[Collector] xyz_mm must be 3D array"
                
                ft_save_data = {
                    'image_name': image_name,
                    'timestamp': timestamp,
                    'fx': float(ft_data_array[0]),
                    'fy': float(ft_data_array[1]),
                    'fz': float(ft_data_array[2]),
                    'mx': float(ft_data_array[3]),
                    'my': float(ft_data_array[4]),
                    'mz': float(ft_data_array[5]),
                    'x': float(xyz_mm_array[0]),
                    'y': float(xyz_mm_array[1]),
                    'z': float(xyz_mm_array[2]),
                    'depth': float(depth_mm),
                }

                self.ft_sensor.write_in(save_data=ft_save_data)
                logger.debug(f"Saved FT data: {self.ft_sensor.csv_save_path}")

        for i in range(self.num_frames):
            current_image_name = f'{image_name}_{i}' if self.num_frames > 1 else image_name
            _save_one_frame(
                image_name=current_image_name, 
                ft_data=ft_data, 
                xyz_mm=xyz_mm,
                depth_mm=depth_mm)
            if i < self.num_frames - 1:
                time.sleep(delay)
    
    def save_reference_image(
        self,
        ref_save_dir: str = None,
        image_name: str = 'ref',
        num_images: int = 3,
        delay: float = 0.5,
    ):
        if not self.enable_camera:
            return

        assert ref_save_dir is not None, "[Collector] ref_save_dir is required"
        os.makedirs(ref_save_dir, exist_ok=True)
        
        logger.info(f"Save reference | name: {image_name} | {num_images} images | dir: {ref_save_dir}")
        
        for i in range(num_images):
            frame, timestamp = self.camera.get_frame()
            image_path = os.path.join(ref_save_dir, f'{image_name}_{i}.{self.image_format}')
            cv2.imwrite(image_path, frame)
            logger.debug(f"Saved reference image: {image_path}")
            if i < num_images - 1:
                time.sleep(delay)
    
    def release(self):
        logger.debug("Releasing resources")
        if self.enable_camera:
            self.camera.release()
        if self.enable_ft:
            self.ft_sensor.release()
        logger.info("Collector released")
    