import time
import spdlog
import numpy as np
from typing import List
from ..devices.collector_DYS import Collector_DYS
from ..rizon.rizon import Rizon
from ..utils import Colors
logger = spdlog.ConsoleLogger('Trajectory_DYS')

def execute_press_sequence_DYS(
    robot: Rizon,
    collector: Collector_DYS,
    begin_pose: np.ndarray,     # m
    contact_z: float,           # m
    depth: float = 0.001,       # m
    trajectory_xyz_list: List[np.ndarray] = None,  # m (center [0, 0, 0])
):
    assert depth is not None, "depth must be provided"
    assert begin_pose is not None, "begin_pose must be provided"
    assert trajectory_xyz_list is not None, "trajectory_xyz_list must be provided"

    # begin when eef is at the begin pose
    contact_face_center = begin_pose[:3].copy()
    contact_face_center[2] = contact_z

    for k, point_xyz in enumerate(trajectory_xyz_list):
        logger.info(Colors.colorize(f"Point{k+1}/{len(trajectory_xyz_list)} | depth: {depth*1000:.1f}mm | point: {point_xyz}", Colors.BLUE))
        # start from begin pose
        sample_pose_base = begin_pose.copy()
        sample_pose_base[:3] += point_xyz

        down_pose_base = sample_pose_base.copy()
        down_pose_base[2] = contact_z - depth

        robot.move_l(target_pose=sample_pose_base, vel=0.005)

        for j in range(3):
            # zero ft sensor
            robot.robot.ExecutePrimitive("ZeroFTSensor", dict())
            while not robot.robot.primitive_states()["terminated"]:
                time.sleep(0.1)

            # press down and save
            robot.move_l(target_pose=down_pose_base, vel=0.0015)
            robot.hold(
                duration_before=1.0, 
                duration_after=1.0,
                callback=lambda r: collector.save(
                    image_name=f'{depth*1000:.1f}mm_press{k+1}-{j+1}', 
                    ft_data=r.robot.states().ext_wrench_in_world,
                    xyz_mm=(r.robot.states().tcp_pose[:3] - contact_face_center) * 1000.0,
                    depth_mm= - (r.robot.states().tcp_pose[2] - contact_z) * 1000.0
                )
            )
            robot.move_l(target_pose=sample_pose_base, vel=0.002)
            if j < 2:
                time.sleep(1.0)

def execute_rotate_sequence_DYS(
    robot: Rizon,
    collector: Collector_DYS,
    begin_pose: np.ndarray,     # m
    contact_z: float,           # m
    depth: float = 0.001,       # m
    trajectory_xyz_list: List[np.ndarray] = None,  # m (center [0, 0, 0])
):
    assert depth is not None, "depth must be provided"
    assert begin_pose is not None, "begin_pose must be provided"
    assert trajectory_xyz_list is not None, "trajectory_xyz_list must be provided"

    contact_face_center = begin_pose[:3].copy()
    contact_face_center[2] = contact_z

    for k, point_xyz in enumerate(trajectory_xyz_list):
        logger.info(Colors.colorize(f"Point{k+1}/{len(trajectory_xyz_list)} | depth: {depth*1000:.1f}mm | point: {point_xyz}", Colors.BLUE))
        # start from begin pose
        sample_pose_base = begin_pose.copy()
        sample_pose_base[:3] += point_xyz

        down_pose_base = sample_pose_base.copy()
        down_pose_base[2] = contact_z - depth

        robot.move_l(target_pose=sample_pose_base, vel=0.005)

        # zero ft sensor
        robot.robot.ExecutePrimitive("ZeroFTSensor", dict())
        while not robot.robot.primitive_states()["terminated"]:
            time.sleep(0.1)

        # press down and rotate cw
        robot.move_l(target_pose=down_pose_base, vel=0.0015)
        for i in range(3):
            robot.move_l(target_pose=robot.get_rot_pose(angle=5.0), vel=0.002, angVel=5.0)
            robot.hold(
                duration_before=1.0, 
                duration_after=1.0,
                callback=lambda r: collector.save(
                    image_name=f'{depth*1000:.1f}mm_rotate{k+1}-{i+1}cw', 
                    ft_data=r.robot.states().ext_wrench_in_world,
                    xyz_mm=(r.robot.states().tcp_pose[:3] - contact_face_center) * 1000.0,
                    depth_mm= - (r.robot.states().tcp_pose[2] - contact_z) * 1000.0
                )
            )
        robot.move_l(target_pose=sample_pose_base, vel=0.002, angVel=5.0)

        time.sleep(2.0)

        # zero ft sensor
        robot.robot.ExecutePrimitive("ZeroFTSensor", dict())
        while not robot.robot.primitive_states()["terminated"]:
            time.sleep(0.1)

        # press down and rotate ccw
        robot.move_l(target_pose=down_pose_base, vel=0.0015)
        for i in range(3):
            robot.move_l(target_pose=robot.get_rot_pose(angle=-5.0), vel=0.002, angVel=5.0)
            robot.hold(
                duration_before=1.0, 
                duration_after=1.0,
                callback=lambda r: collector.save(
                    image_name=f'{depth*1000:.1f}mm_rotate{k+1}-{i+1}ccw', 
                    ft_data=r.robot.states().ext_wrench_in_world,
                    xyz_mm=(r.robot.states().tcp_pose[:3] - contact_face_center) * 1000.0,
                    depth_mm= - (r.robot.states().tcp_pose[2] - contact_z) * 1000.0
                )
            )
        robot.move_l(target_pose=sample_pose_base, vel=0.002, angVel=5.0)

def execute_move_sequence_DYS(
    robot: Rizon,
    collector: Collector_DYS,
    begin_pose: np.ndarray,     # m
    contact_z: float,           # m
    depth: float = 0.001,       # m
    trajectory_xyz_list: List[np.ndarray] = None,  # m (center [0, 0, 0])
):
    assert depth is not None, "depth must be provided"
    assert begin_pose is not None, "begin_pose must be provided"
    assert trajectory_xyz_list is not None, "trajectory_xyz_list must be provided"
    
    contact_face_center = begin_pose[:3].copy()
    contact_face_center[2] = contact_z

    step_size = 0.0005
    trajectory = [
        [         0, step_size , 0, 0, 0, 0, 0],
        [         0, -step_size, 0, 0, 0, 0, 0],
        [-step_size, 0         , 0, 0, 0, 0, 0],
        [step_size , 0         , 0, 0, 0, 0, 0],
        [-step_size, step_size , 0, 0, 0, 0, 0],
        [step_size , -step_size, 0, 0, 0, 0, 0],
        [step_size , step_size , 0, 0, 0, 0, 0],
        [-step_size, -step_size, 0, 0, 0, 0, 0],
    ]

    for k, point_xyz in enumerate(trajectory_xyz_list):
        logger.info(Colors.colorize(f"Point {k+1}/{len(trajectory_xyz_list)} | depth: {depth*1000:.1f}mm | point: {point_xyz}", Colors.BLUE))
        # start from begin pose
        sample_pose_base = begin_pose.copy()
        sample_pose_base[:3] += point_xyz

        down_pose_base = sample_pose_base.copy()
        down_pose_base[2] = contact_z - depth

        for i, step in enumerate(trajectory):
            # move to sample point
            robot.move_l(target_pose=sample_pose_base, vel=0.005)

            # zero ft sensor
            robot.robot.ExecutePrimitive("ZeroFTSensor", dict())
            while not robot.robot.primitive_states()["terminated"]:
                time.sleep(0.1)
            
            # press down
            robot.move_l(target_pose=down_pose_base, vel=0.0015)
            time.sleep(0.3)

            # move 3 times
            for j in range(3):
                logger.info(Colors.colorize(f"Move forward {i+1}/{len(trajectory)} | depth: {depth*1000:.1f}mm | step: {step} | time: {j+1}/3", Colors.CYAN))
                robot.move_l(target_pose=down_pose_base + np.array(step, dtype=float) * (j+1), vel=0.002)
                robot.hold(
                    duration_before=1.0, 
                    duration_after=1.0,
                    callback=lambda r: collector.save(
                        image_name=f'{depth*1000:.1f}mm_move{k+1}-{i+1}-{j+1}', 
                        ft_data=r.robot.states().ext_wrench_in_world,
                        xyz_mm=(r.robot.states().tcp_pose[:3] - contact_face_center) * 1000.0,
                        depth_mm= - (r.robot.states().tcp_pose[2] - contact_z) * 1000.0
                    )
                )
            
            # up and down
            robot.move_l(target_pose=sample_pose_base + np.array(step, dtype=float) * 3, vel=0.002)
            time.sleep(2.0)
            # zero ft sensor
            robot.robot.ExecutePrimitive("ZeroFTSensor", dict())
            while not robot.robot.primitive_states()["terminated"]:
                time.sleep(0.1)
            robot.move_l(target_pose=down_pose_base + np.array(step, dtype=float) * 3, vel=0.0015)
            time.sleep(0.3)

            # back
            for j in range(3):
                logger.info(Colors.colorize(f"Move backward {i+1}/{len(trajectory)} | depth: {depth*1000:.1f}mm | step: {step} | time: {j+1}/3", Colors.CYAN))
                robot.move_l(target_pose=down_pose_base + np.array(step, dtype=float) * (3-(j+1)), vel=0.002)
                robot.hold(
                    duration_before=1.0, 
                    duration_after=1.0,
                    callback=lambda r: collector.save(
                        image_name=f'{depth*1000:.1f}mm_move{k+1}-{i+1}-{j+1}back', 
                        ft_data=r.robot.states().ext_wrench_in_world,
                        xyz_mm=(r.robot.states().tcp_pose[:3] - contact_face_center) * 1000.0,
                        depth_mm= - (r.robot.states().tcp_pose[2] - contact_z) * 1000.0
                    )
                )
            
            # move to sample pose
            robot.move_l(target_pose=sample_pose_base, vel=0.002)
            time.sleep(2.0)