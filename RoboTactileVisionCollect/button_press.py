import os
import time
import flexivrdk
import numpy as np
from rizon import Rizon
from scipy.spatial.transform import Rotation as Rot
from utils import pose_matrix_to_quaternion, generate_sampling_points, Camera, FTSensor

BUTTON_CENTER_XY = [0.61364, -0.00740]

class ButtonPress:   

    def __init__(self, 
            tool_name: str = 'soft_eef',
            data_dir: str = "/media/ubuntu/B0A8C06FA8C0361E/Data/SoftButton_Data",
            record_name: str = None,
            sampling_points_config: dict = None,
            camera_params: dict = None,
            begin_position: list = None,
            high_position: list = None,
            ):

        self.tool_name = tool_name
        self.sampling_points_config = sampling_points_config

        self.data_dir = data_dir
        if record_name is None:
            record_name = f"record_{time.strftime('%Y%m%d_%H%M%S')}"
        self.record_name = record_name
        self.record_dir = os.path.join(self.data_dir, self.record_name)
        os.makedirs(self.record_dir, exist_ok=True)

        self.ftsensor = FTSensor(record_dir=self.record_dir, precision=9)
        self.camera = Camera(record_dir=self.record_dir, camera_params=camera_params)

        self.robot = Rizon(tool_name=self.tool_name)
        self.robot.motion_mode('tcp')

        self.begin_pose = np.eye(4)
        self.begin_pose[:3, :3] = Rot.from_euler('xyz', [180, 0, 180], degrees=True).as_matrix()
        self.begin_pose[:3, 3] = BUTTON_CENTER_XY + [0.15]

        self.high_pose = np.eye(4)
        self.high_pose[:3, :3] = Rot.from_euler('xyz', [180, 0, 180], degrees=True).as_matrix()
        if high_position is not None:
            self.high_pose[:3, 3] = high_position
        else:
            self.high_pose[:3, 3] = self.HIGH_POSITION

    def close(self):
        if hasattr(self, "camera") and self.camera is not None:
            self.camera.release()
        if hasattr(self, "robot") and self.robot is not None:
            self.robot.close()

    def save_reference_image(self, img_name, save_num=3, delay=0.5):
        for _ in range(3):
            self.camera.get_frame()
        for i in range(save_num):
            frame, timestamp = self.camera.get_frame()
            img_dir = self.camera.depth_dir
            self.camera.save_frame(frame, f'{img_name}_{i}.png', img_dir)
            time.sleep(delay)
        print(f"Reference image saved: {img_name}")

    def save_one_frame(self, img_prefix, idx, error_mm=None, sampling_point=None):
        frame, timestamp = self.camera.get_frame()
        image_name = f'{img_prefix}_{idx}frame_{timestamp}.png'
        self.camera.save_frame(frame, image_name)
        self.ftsensor.write(
            self.robot.ft_read(tcp=True, filtered=True, raw=False),
            image_name=image_name,
            timestamp=timestamp,
            error_mm=error_mm,
            sampling_point=sampling_point
        )

    def save_data(self, img_prefix, error_mm=None, sampling_point=None, delay=0.5):
        # clear camera buffer
        for _ in range(3):
            self.camera.get_frame()
        
        for i in range(5):
            self.save_one_frame(img_prefix, i, error_mm=error_mm, sampling_point=sampling_point)
            time.sleep(delay)

    def move_down_with_force(self, target_force=10.0, target_pose=None, period=0.02, duration=3.0):
        if target_pose is None:
            raise ValueError("target_pose cannot be None")       
        try:
            self.check_connection()
            self.robot.robot.SetForceControlAxis([False, False, True, False, False, False])
            
            target_wrench = [0.0, 0.0, target_force, 0.0, 0.0, 0.0]
            target_pose_quat = pose_matrix_to_quaternion(target_pose)
            
            start_time = time.time()        
            while time.time() - start_time < duration:
                self.robot.robot.SendCartesianMotionForce(target_pose_quat, target_wrench)
                time.sleep(period)
            
            final_pose = self.robot.tcp_read()
            error_m = final_pose[2, 3] - target_pose[2, 3]
            error_mm = error_m * 1000
            if self.robot.robot.fault():
                if not self.robot.robot.ClearFault():
                    raise RuntimeError(f"Failed to clear fault for robot")
            if error_m > 0.0002:
                print(f"Force too small, cannot compress button, error = {error_mm:.4f}mm, retry")
                return False, error_mm
            elif error_m < -0.0002:
                print(f"Force too large, compress button over, error = {error_mm:.4f}mm, retry")
                return False, error_mm
            else:
                print(f"Force control completed: error within 0.2mm, error = {error_mm:.4f}mm, force = {target_force:.3f}N")
                return True, error_mm

        except RuntimeError as e:
            print(f"Error: {e}")
            # 修复：异常时返回失败状态
            return False, None
        except Exception as e:
            print(f"Unexpected error: {e}")
            return False, None

        finally:
            try:
                if self.robot.robot.connected():
                    self.robot.robot.SetForceControlAxis([False, False, False, False, False, False])
            except Exception as e:
                print(f"Warning: Failed to restore force control axis in finally: {e}")

    def move_down(self, step_size=0.0005, use_force_control=False, init_force=10.0, max_retries=99):
        self.check_connection()
        begin_pose = self.robot.tcp_read()
        begin_pos = begin_pose[:3, 3]
        down_pos = begin_pos - [0, 0, step_size]
        down_pose = begin_pose.copy()
        down_pose[:3, 3] = down_pos
        if not use_force_control:
            self.robot.tcp_move(down_pose)
            return None
        else:
            current_force = init_force
            retry_count = 0
            error_mm = None
            while retry_count < max_retries:
                success, error_mm = self.move_down_with_force(
                    target_force=current_force,
                    duration=3.0,
                    target_pose=down_pose.copy()
                )
                if success:
                    break  # 成功，退出循环
                if error_mm is None:
                    print(f"[Retry {retry_count:03d}/{max_retries:03d}][Warning] error_mm is None, skipping force adjustment")
                    retry_count += 1
                    continue
                # 类型和值检查
                if not isinstance(error_mm, (int, float)):
                    print(f"[Retry {retry_count:03d}/{max_retries:03d}][Error] error_mm is not a number: {error_mm} (type: {type(error_mm)})")
                    retry_count += 1
                    continue
                # 调试信息：显示调整前的状态
                old_force = current_force
                abs_error = abs(error_mm)
                if old_force >= 3:
                    coeff_small = 0.6
                    coeff_large = 1.5
                else:
                    coeff_small = 0.2
                    coeff_large = 0.6
                if abs_error < 0.3:
                    adjustment = coeff_small * error_mm
                    current_force = current_force + adjustment
                    coeff_used = coeff_small
                else:
                    adjustment = coeff_large * error_mm
                    current_force = current_force + adjustment
                    coeff_used = coeff_large
                current_force_before_clamp = current_force
                current_force = max(0.05, min(current_force, 45.0))
                print(f"[Retry {retry_count:03d}/{max_retries:03d}] Force adjustment: {old_force:.4f}N -> {current_force_before_clamp:.4f}N")
                self.check_connection()
                self.robot.tcp_move(begin_pose)
                self.robot.sleep(1.0)
                retry_count += 1
            return error_mm, current_force

    def move_up(self, step_size=0.0005):
        self.check_connection()
        
        begin_pose = self.robot.tcp_read()
        begin_pos = begin_pose[:3, 3]
        up_pos = begin_pos + [0, 0, step_size]
        up_pose = begin_pose.copy()
        up_pose[:3, 3] = up_pos

        self.robot.tcp_move(up_pose)
    
    def rotate_around_z(self, angle_degrees):
        self.check_connection()
        
        current_pose = self.robot.tcp_read()
        current_pos = current_pose[:3, 3].copy()
        current_rotation = current_pose[:3, :3]
        current_rot = Rot.from_matrix(current_rotation)
        
        z_axis_rotation = Rot.from_euler('z', angle_degrees, degrees=True)
        new_rot = current_rot * z_axis_rotation
        new_rotation_matrix = new_rot.as_matrix()
        
        new_pose = current_pose.copy()
        new_pose[:3, :3] = new_rotation_matrix
        new_pose[:3, 3] = current_pos
        
        self.robot.tcp_move(new_pose)
    
    def get_sphere_trajectory_poses(self):
        '''
        generate trajectory poses from current pose
        '''
        sampling_points_2d = generate_sampling_points(origin=(0., 0.),
                                                                grid_step=self.sampling_points_config['grid_step'],
                                                                square_half_width=self.sampling_points_config['square_half_width'],
                                                                radius_limit=self.sampling_points_config['radius_limit'])
        current_pose = self.robot.tcp_read()
        trajectory_poses = []
        
        for idx, point in enumerate(sampling_points_2d):
            begin_xy = current_pose[:2, 3]
            target_xy = begin_xy + point
            target_pose = current_pose.copy()
            target_pose[:2, 3] = target_xy
            trajectory_poses.append(target_pose)
        
        return trajectory_poses, sampling_points_2d
    
    def excute_fixed_depth_sequence(self, 
            depth=0.0005, 
            init_force=10.0,
            force_mix_ratio=0.5):
        """
        执行固定深度的采样序列
        
        Args:
            depth: 下压深度（米）
            init_force: 初始力（N），第一个采样点使用此值
            force_mix_ratio: 力混合比例（0-1），用于计算后续点的初始力
                            new_init_force = force_mix_ratio * init_force + (1 - force_mix_ratio) * last_point_force
                            默认 0.5 表示各占50%
        """
        self.force_torque_logger.create_depth_dir(f'{(depth*1000):.2f}mm')
        self.camera_logger.create_depth_dir(f'{(depth*1000):.2f}mm')
        self.check_connection()
        self.robot.tcp_move(self.begin_pose)

        # get reference image
        self.save_reference_image(f"reference_{(depth*1000):.2f}mm")

        self.robot.sleep(1.0)
        
        trajectory_poses, sampling_points_2d = self.get_sphere_trajectory_poses()
        
        user_init_force = init_force
        current_init_force = init_force
        last_point_force = None
        
        for idx, (pose, sampling_point) in enumerate(zip(trajectory_poses, sampling_points_2d)):
            if idx == 0:
                current_init_force = user_init_force
                print(f"[{idx:03d}/{len(trajectory_poses):03d}][Depth: {depth*1000:.2f}mm] Init force: {current_init_force:.2f}N (user set)")
            else:
                if last_point_force is not None:
                    current_init_force = force_mix_ratio * user_init_force + (1 - force_mix_ratio) * last_point_force
                    print(f"[{idx:03d}/{len(trajectory_poses):03d}][Depth: {depth*1000:.2f}mm] Init force: {current_init_force:.2f}N (mixed)")
                else:
                    current_init_force = user_init_force
                    print(f"[{idx:03d}/{len(trajectory_poses):03d}][Depth: {depth*1000:.2f}mm] Init force: {current_init_force:.2f}N (fallback to user set)")

            self.check_connection()
            self.robot.tcp_move(pose)
            self.robot.sleep(1.0)
            error_mm, last_point_force = self.move_down(step_size=float(depth + self.DELTA_DISTANCE_BEGIN_TO_BUTTON), use_force_control=True, init_force=current_init_force)
            self.robot.sleep(1.0)
            self.save_data(f'{(depth*1000):.2f}mm_{idx:03d}point', error_mm=error_mm, sampling_point=sampling_point)
            self.move_up(step_size=float(depth + self.DELTA_DISTANCE_BEGIN_TO_BUTTON))
            self.robot.sleep(6.0)
            print(f"------------------------------------------------------------------------\n")
        self.robot.tcp_move(self.begin_pose)

        self.robot_reconstruction()

    def translate_xy(self, dx=0.0, dy=0.0):
        """
        在XY平面平移TCP位置
        
        Args:
            dx: x方向平移距离（米）
            dy: y方向平移距离（米）
        """
        self.check_connection()
        
        current_pose = self.robot.tcp_read()
        current_pos = current_pose[:3, 3].copy()
        new_pos = current_pos + [dx, dy, 0]
        
        new_pose = current_pose.copy()
        new_pose[:3, 3] = new_pos
        
        self.robot.tcp_move(new_pose)
    
    def lift_and_press_down(self, initial_pose, depth=0.0005):
        """
        抬起并重新落下的可复用函数
        
        Args:
            initial_height: 初始高度（米）
            depth: 下压深度（米）
            action: 可选的回调函数，在抬起后、下压前执行。函数签名: action() -> None
        """
        self.check_connection()
        
        # 抬起
        begin_pose = self.robot.tcp_read()
        current_xy = begin_pose[:2, 3].copy()
        up_pose = initial_pose.copy()
        up_pose[:2, 3] = current_xy
        self.robot.tcp_move(up_pose)
        
        self.robot.sleep(0.5)

        # 下压
        down_pose = up_pose.copy()
        down_pose[2, 3] = down_pose[2, 3] - depth
        self.robot.tcp_move(down_pose)
        self.robot.sleep(0.5)

    def excute_move_sequence(self, depth=0.0005, random_range=0.0000):
        self.force_torque_logger.create_depth_dir(f'{(depth*1000):.2f}mm')
        self.camera_logger.create_depth_dir(f'{(depth*1000):.2f}mm')
        self.check_connection()
        self.robot.tcp_move(self.begin_pose)

        self.save_reference_image(f"reference_{(depth*1000):.2f}mm", delay=0.5)
        self.robot.sleep(0.5)

        origin_depth = depth
        
        trajectory_poses, sampling_points_2d = self.get_sphere_trajectory_poses()
        for idx, (pose, sampling_point) in enumerate(zip(trajectory_poses, sampling_points_2d)):
            print(f"\n{'='*80}")
            print(f"Processing trajectory point {idx+1}/{len(trajectory_poses)}")
            print(f"{'='*80}\n")

            # random offset the depth
            depth_offset = np.random.uniform(-random_range, random_range)
            current_depth = origin_depth + depth_offset
            self.check_connection()
            self.robot.tcp_move(pose)
            self.robot.sleep(0.5)
            
            # 记录初始位置（用于后续返回）
            initial_pose = pose
            
            # ========== 步骤1: 向下移动指定depth，记录数据 ==========
            print("Step 1: Moving down and recording data...")
            self.move_down(step_size=current_depth, use_force_control=False)
            self.robot.sleep(0.5)
            print(f"Saving data: {(current_depth*1000):.2f}mm_{idx:03d}point_press1")
            print(f"Current pose: {self.robot.tcp_read()}")
            self.save_data(f'{(current_depth*1000):.2f}mm_{idx:03d}point_press1', 
                          sampling_point=sampling_point, delay=0.5)
            
            # ========== 步骤2: 顺时针旋转45°（3次，每次15°），每次旋转后采集数据 ==========
            print("Step 2: Clockwise rotation (45° total, 15° per step)...")
            for rot_idx in range(3):
                self.rotate_around_z(10.0)  # 顺时针旋转10°
                self.robot.sleep(0.5)
                rot_end_pose = self.robot.tcp_read()
                rot_end_pose[2, 3] = initial_pose[2, 3] - float(current_depth)
                self.robot.tcp_move(rot_end_pose)
                self.robot.sleep(0.5)
                print(f"Saving data: {(current_depth*1000):.2f}mm_{idx:03d}point_rotate_cw{rot_idx+1}")
                print(f"Current pose: {self.robot.tcp_read()}")
                self.save_data(f'{(current_depth*1000):.2f}mm_{idx:03d}point_rotate_cw{rot_idx+1}', 
                              sampling_point=sampling_point, delay=0.5)
            
            # ========== 步骤3: 抬高并再次下压 ==========
            print("Step 3: Moving up to initial height...")
            self.lift_and_press_down(initial_pose=initial_pose, depth=current_depth)
            print(f"Saving data: {(current_depth*1000):.2f}mm_{idx:03d}point_press2")
            print(f"Current pose: {self.robot.tcp_read()}")
            self.save_data(f'{(current_depth*1000):.2f}mm_{idx:03d}point_press2', 
                          sampling_point=sampling_point, delay=0.5)
            
            # ========== 步骤4: 逆时针旋转45°（3次，每次15°），每次旋转后采集数据 ==========
            print("Step 4: Counter-clockwise rotation (45° total, 15° per step)...")
            for rot_idx in range(3):
                self.rotate_around_z(-10.0)  # 逆时针旋转10°
                self.robot.sleep(0.5)
                rot_end_pose = self.robot.tcp_read()
                rot_end_pose[2, 3] = initial_pose[2, 3] - float(current_depth)
                self.robot.tcp_move(rot_end_pose)
                self.robot.sleep(0.5)
                print(f"Saving data: {(current_depth*1000):.2f}mm_{idx:03d}point_rotate_ccw_{rot_idx+1}")
                print(f"Current pose: {self.robot.tcp_read()}")
                self.save_data(f'{(current_depth*1000):.2f}mm_{idx:03d}point_rotate_ccw_{rot_idx+1}', 
                              sampling_point=sampling_point, delay=0.5)
            
            # ========== 步骤5: 抬起到初始高度并再次下压，采集数据 ==========
            print("Step 5: Moving up, then down again, and recording...")
            self.lift_and_press_down(initial_pose=initial_pose, depth=current_depth)
            
            # ========== 步骤6: X方向平移采集 ==========
            print("Step 6: X-direction translation...")
            # 记录X方向的初始位置
            initial_xy = initial_pose[:2, 3].copy()
            
            # X正方向平移（每次1mm，共3次，总共3mm）
            for x_step in range(1, 4):  # 1mm, 2mm, 3mm
                target_xy = initial_xy + [0.001 * x_step, 0.0]
                target_pose = initial_pose.copy()
                target_pose[:2, 3] = target_xy
                target_pose[2, 3] -= float(current_depth)
                self.robot.tcp_move(target_pose)
                self.robot.sleep(0.5)
                print(f"Current pose: {self.robot.tcp_read()}")
                print(f"Saving data: {(current_depth*1000):.2f}mm_{idx:03d}point_xforward_{x_step}mm")
                self.save_data(f'{(current_depth*1000):.2f}mm_{idx:03d}point_xforward_{x_step}mm', 
                              sampling_point=sampling_point, delay=0.5)

            self.lift_and_press_down(initial_pose=initial_pose, depth=current_depth)
            self.robot.sleep(0.5)
            
            # X负方向平移（回到初始位置，每次1mm，共3次）
            for x_step in range(2, -1, -1):  # 2mm, 1mm, 0mm (回到初始)
                target_xy = initial_xy + [0.001 * x_step, 0.0]
                target_pose = initial_pose.copy()
                target_pose[:2, 3] = target_xy
                target_pose[2, 3] -= float(current_depth)
                self.robot.tcp_move(target_pose)
                self.robot.sleep(0.5)
                print(f"Current pose: {self.robot.tcp_read()}")
                print(f"Saving data: {(current_depth*1000):.2f}mm_{idx:03d}point_xbackward_{x_step}mm")
                self.save_data(f'{(current_depth*1000):.2f}mm_{idx:03d}point_xbackward_{x_step}mm', 
                              sampling_point=sampling_point, delay=0.5)
            
            # 确保回到X初始位置（轨迹点的初始位置）
            self.robot.tcp_move(initial_pose)
            self.robot.sleep(0.5)
            self.move_down(step_size=current_depth, use_force_control=False)
            self.robot.sleep(0.5)
            
            # ========== 步骤7: Y方向平移采集（与X方向相同的步骤） ==========
            print("Step 7: Y-direction translation...")
            # 记录Y方向的初始位置（应该是轨迹点的初始位置）
            initial_xy = initial_pose[:2, 3].copy()
            
            # Y正方向平移（每次1mm，共3次，总共3mm）
            for y_step in range(1, 4):  # 1mm, 2mm, 3mm
                target_xy = initial_xy + [0.0, 0.001 * y_step]
                target_pose = initial_pose.copy()
                target_pose[:2, 3] = target_xy
                target_pose[2, 3] -= float(current_depth)
                self.robot.tcp_move(target_pose)
                self.robot.sleep(0.5)
                print(f"Current pose: {self.robot.tcp_read()}")
                print(f"Saving data: {(current_depth*1000):.2f}mm_{idx:03d}point_yforward_{y_step}mm")
                self.save_data(f'{(current_depth*1000):.2f}mm_{idx:03d}point_yforward_{y_step}mm', 
                              sampling_point=sampling_point, delay=0.5)
            
            self.lift_and_press_down(initial_pose=initial_pose, depth=current_depth)
            self.robot.sleep(0.5)
            
            # Y负方向平移（回到初始位置，每次1mm，共3次）
            for y_step in range(2, -1, -1):  # 2mm, 1mm, 0mm (回到初始)
                target_xy = initial_xy + [0.0, 0.001 * y_step]
                target_pose = initial_pose.copy()
                target_pose[:2, 3] = target_xy
                target_pose[2, 3] -= float(current_depth)
                self.robot.tcp_move(target_pose)
                self.robot.sleep(0.5)
                print(f"Current pose: {self.robot.tcp_read()}")
                print(f"Saving data: {(current_depth*1000):.2f}mm_{idx:03d}point_ybackward_{y_step}mm")
                self.save_data(f'{(current_depth*1000):.2f}mm_{idx:03d}point_ybackward_{y_step}mm', 
                              sampling_point=sampling_point, delay=0.5)
            
            # 确保回到X初始位置（轨迹点的初始位置）
            self.robot.tcp_move(initial_pose)
            self.robot.sleep(0.5)
            self.move_down(step_size=current_depth, use_force_control=False)
            self.robot.sleep(0.5)
            
            # ========== 步骤8: X与Y的45°方向平移采集 ==========
            print("Step 8: 45° direction (X+Y) translation...")
            initial_xy = initial_pose[:2, 3].copy()
            
            # XY正方向平移（每次1mm，共3次，总共3mm）
            for diag_step in range(1, 4):  # 1mm, 2mm, 3mm
                target_xy = initial_xy + [0.001 * diag_step, 0.001 * diag_step]
                target_pose = initial_pose.copy()
                target_pose[:2, 3] = target_xy
                target_pose[2, 3] -= float(current_depth)
                self.robot.tcp_move(target_pose)
                self.robot.sleep(0.5)
                print(f"Current pose: {self.robot.tcp_read()}")
                print(f"Saving data: {(current_depth*1000):.2f}mm_{idx:03d}point_diagforward_{diag_step}mm")
                self.save_data(f'{(current_depth*1000):.2f}mm_{idx:03d}point_diagforward_{diag_step}mm', 
                              sampling_point=sampling_point, delay=0.5)
            
            self.lift_and_press_down(initial_pose=initial_pose, depth=current_depth)
            self.robot.sleep(0.5)
            
            # XY负方向平移（回到初始位置，每次1mm，共3次）
            for diag_step in range(2, -1, -1):  # 2mm, 1mm, 0mm (回到初始)
                target_xy = initial_xy + [0.001 * diag_step, 0.001 * diag_step]
                target_pose = initial_pose.copy()
                target_pose[:2, 3] = target_xy
                target_pose[2, 3] -= float(current_depth)
                self.robot.tcp_move(target_pose)
                self.robot.sleep(0.5)
                print(f"Current pose: {self.robot.tcp_read()}")
                print(f"Saving data: {(current_depth*1000):.2f}mm_{idx:03d}point_diagbackward_{diag_step}mm")
                self.save_data(f'{(current_depth*1000):.2f}mm_{idx:03d}point_diagbackward_{diag_step}mm', 
                              sampling_point=sampling_point, delay=0.5)
            
            # 确保回到X初始位置（轨迹点的初始位置）
            self.robot.tcp_move(initial_pose)
            self.robot.sleep(0.5)
            self.move_down(step_size=current_depth, use_force_control=False)
            self.robot.sleep(0.5)

        # high position
        self.robot.tcp_move(self.high_pose)
        self.robot.sleep(0.5)


if __name__ == "__main__":
    button_press = ButtonPress(
        robot_id="Rizon4s-063231",
        tool_name='soft_eef',
        data_dir="/media/ubuntu/B0A8C06FA8C0361E/Data/SoftButton_Data",
    )
