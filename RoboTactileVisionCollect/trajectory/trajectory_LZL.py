import time
import spdlog
import numpy as np
from typing import List
from ..devices.collector_LZL import Collector_LZL
from ..rizon.rizon import Rizon

logger = spdlog.ConsoleLogger('Trajectory_LZL')

def execute_single_point_LZL(
    robot: Rizon,
    collector: Collector_LZL,
    begin_pose: np.ndarray,
    force: float = 1.0,
    step_size: float = 0.001,
):
    assert force is not None, "force must be provided"
    assert step_size is not None, "step_size must be provided"
    assert begin_pose is not None, "begin_pose must be provided"

    # the trajectory begin when eef is at the begin pose
    logger.info(f"Executing trajectory point | force: {force}N")

    trajectory = [
        [0, step_size, 0],                       # move 1
        [0, -step_size, 0],                      # move 3
        [-step_size, 0, 0],                      # move 5
        [step_size, 0, 0],                       # move 7
        [-step_size, step_size, 0],              # move 9
        [step_size, -step_size, 0],              # move 11
        [step_size, step_size, 0],               # move 13
        [-step_size, -step_size, 0],             # move 15
    ]

    # 固定姿态：避免从 tcp_pose 四元数->欧拉角时产生不连续跳变导致快速旋转
    fixed_quat_wxyz = begin_pose[3:7].copy()

    for k in range(2):

        for i, step in enumerate(trajectory):
            # start from begin pose
            # press down and save
            robot.force_hybrid(
                target_pose=begin_pose,
                target_wrench=np.array([0, 0, force, 0, 0, 0], dtype=float),
            )
            robot.hold(
                duration_before=1.0, 
                duration_after=1.0,
                callback=lambda r: collector.save(
                    image_name=f'{force:.1f}N_{k+1}_press{i+1}', 
                    ft_data=r.robot.states().ext_wrench_in_world,
                    point_xy_mm=np.array([0, 0])
                )
            )
            
            for j in range(2):
                # move
                target_pose = np.array(robot.robot.states().tcp_pose, dtype=float)
                target_pose[:3] += np.array(step, dtype=float)
                target_pose[3:7] = fixed_quat_wxyz
                robot.move_l(target_pose=target_pose)
                time.sleep(0.3)
                # save
                robot.hold(
                    duration_before=1.0, 
                    duration_after=1.0,
                    callback=lambda r: collector.save(
                        image_name=f'{force:.1f}N_{k+1}_move{i+1}-{j+1}', 
                        ft_data=r.robot.states().ext_wrench_in_world,
                        point_xy_mm=np.array([0, 0])
                    )
                )

            for j in range(2):
                # move
                target_pose = np.array(robot.robot.states().tcp_pose, dtype=float)
                target_pose[:3] -= np.array(step, dtype=float)
                target_pose[3:7] = fixed_quat_wxyz
                robot.move_l(target_pose=target_pose)
                time.sleep(0.3)
                # save
                robot.hold(
                    duration_before=1.0, 
                    duration_after=1.0,
                    callback=lambda r: collector.save(
                        image_name=f'{force:.1f}N_{k+1}_move{i+1}-{j+1}back', 
                        ft_data=r.robot.states().ext_wrench_in_world,
                        point_xy_mm=np.array([0, 0])
                    )
                )
            
            # move to begin pose
            robot.move_l(target_pose=begin_pose)
            time.sleep(0.3)

def execute_multi_point_LZL(
    robot: Rizon,
    collector: Collector_LZL,
    force: float = 1.0,
    begin_pose: np.ndarray = None,
):
    assert force is not None, "force must be provided"
    assert begin_pose is not None, "begin_pose must be provided"

    x_max = 0.62472; x_min = 0.61562; x_middle = (x_max + x_min) / 2
    y_max = -0.00469; y_min = -0.01117
    # x_max = 0.61327+0.0015; x_min = 0.61327-0.0015; x_middle = (x_max + x_min) / 2
    # y_max = 0.19331+0.001; y_min = 0.19331-0.001
    z = begin_pose[2]
    trajectory = [
        [x_max, y_max, z],
        [x_max, y_min, z],
        [x_middle, y_max, z],
        [x_middle, y_min, z],
        [x_min, y_max, z],
        [x_min, y_min, z],
    ]

    # 固定姿态：避免从 tcp_pose 四元数->欧拉角时产生不连续跳变导致快速旋转
    fixed_quat_wxyz = begin_pose[3:7].copy()

    for i, point in enumerate(trajectory):
        # move to trajectory point
        target_pose = np.zeros(7)
        target_pose[:3] = np.array(point, dtype=float)
        target_pose[3:7] = fixed_quat_wxyz
        robot.move_l(target_pose=target_pose)
        time.sleep(0.3)

        for j in range(3):
            # press down and save
            robot.force_hybrid(
                target_pose=target_pose,
                target_wrench=np.array([0, 0, force, 0, 0, 0], dtype=float),
            )
            robot.hold(
                duration_before=1.0, 
                duration_after=1.0,
                callback=lambda r: collector.save(
                    image_name=f'{force:.1f}N_point{i+1}_press{j+1}', 
                    ft_data=r.robot.states().ext_wrench_in_world,
                    point_xy_mm=np.array(point[:2], dtype=float)
                )
            )

            # move up
            robot.move_l(target_pose=target_pose)
            time.sleep(0.3)
