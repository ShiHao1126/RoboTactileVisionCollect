import os
import sys
import time
import shutil
import spdlog
import numpy as np
from tap import Tap
from scipy.spatial.transform import Rotation as R
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
from RoboTactileVisionCollect.rizon import Rizon
from RoboTactileVisionCollect.devices import Collector_DYS
from RoboTactileVisionCollect.trajectory import *
from RoboTactileVisionCollect.utils import generate_sampling_points

logger = spdlog.ConsoleLogger('Collect_DYS')

# 网格 步长2.5mm 半径23mm 方形半长25mm 硅胶总半径26.5mm
# 抬起后2-3s缓冲
# 1. 下压  每个点下压三次，每次保存一张
# 2. 旋转  一次转10，分3次，顺时针+逆时针 每转一次保存一张
# 3. 米字  平移3次，一次1mm，到终点抬起再下压，返回中心 每平移一次保存一张

class Args(Tap):
    depth: float = 0.0004                            # m
    begin_xyz: list = [0.61502, -0.00808, 0.07720]    # m

    contact_z: float = 0.07096                   # m

    grid_step: float = 6                      # mm 2.5 -> "move" is unique 6mm
    square_half_width: float = 20.0             # mm
    radius_limit: float = 18.0                  # mm  

    # record_dir: str = "/home/ubuntu/data/DYSData/record_20260201/press"
    # record_dir: str = "/home/ubuntu/data/DYSData/record_20260201/rotate"
    record_dir: str = "/home/ubuntu/data/DYSData/record_20260201/move"

    tool_name: str = "Flange"
    force_overwrite: bool = True
    enable_camera: bool = True
    enable_ft: bool = True
    num_frames: int = 1

    camera_params: dict = {
                'CAMERA_INDEX': 0,
                'WIDTH': 1280,
                'HEIGHT': 960,
                'EXPOSURE_VALUE': 1.0,
                'TEMPERATURE_VALUE': 6500.0,
                'FOCUS_VALUE': 398.0,
                'BRIGHTNESS_VALUE': 0.0,
                'CONTRAST_VALUE': 33.0,
                }


def main(args: Args):
    if args.force_overwrite:
        if os.path.exists(os.path.join(args.record_dir, f'{args.depth * 1000:.1f}mm')):
            shutil.rmtree(os.path.join(args.record_dir, f'{args.depth * 1000:.1f}mm'))
        os.makedirs(os.path.join(args.record_dir, f'{args.depth * 1000:.1f}mm'), exist_ok=True)
    else:
        if os.path.exists(os.path.join(args.record_dir, f'{args.depth * 1000:.1f}mm')):
            raise FileExistsError(f"Record directory {os.path.join(args.record_dir, f'{args.depth * 1000:.1f}mm')} already exists")

    collector = Collector_DYS(
        enable_camera=args.enable_camera,
        enable_ft=args.enable_ft,
        image_save_dir=os.path.join(args.record_dir, f'{args.depth * 1000:.1f}mm', 'images'),
        ft_save_path=os.path.join(args.record_dir, f'{args.depth * 1000:.1f}mm', 'ft.csv'),
        num_frames=args.num_frames,
        camera_params=args.camera_params,
    )

    robot = Rizon(tool_name=args.tool_name)

    # move to begin pose
    begin_xyz = args.begin_xyz   # m
    begin_rot = R.from_euler('xyz', [180, 0, 180], degrees=True).as_matrix()
    begin_quat_xyzw = R.from_matrix(begin_rot).as_quat()
    begin_quat_wxyz = np.array([begin_quat_xyzw[3], begin_quat_xyzw[0], begin_quat_xyzw[1], begin_quat_xyzw[2]], dtype=float)
    if begin_quat_wxyz[0] < 0:
        begin_quat_wxyz = -begin_quat_wxyz
    begin_pose = np.concatenate([begin_xyz, begin_quat_wxyz])
    robot.move_l(
        target_pose=begin_pose,
        vel=0.03,
        zoneRadius="Z50",
        targetTolerLevel=1,
        )
    time.sleep(0.3)

    # generate points   
    points = generate_sampling_points(  # in mm
        grid_step=args.grid_step, 
        square_half_width=args.square_half_width, 
        radius_limit=args.radius_limit
        )
    trajectory_xyz = [np.array([point[0] / 1000.0, point[1] / 1000.0, 0.0]) for point in points]    # relative xyz in m (center [0, 0, 0])
    logger.info(f"Points number: {len(trajectory_xyz)}")

    collector.save_reference_image(
        ref_save_dir=os.path.join(args.record_dir, f'{args.depth * 1000:.1f}mm', 'ref'), 
        image_name=f'ref'
    )
    logger.info(f"Reference image saved")
    
    logger.info(f"Sequence started")

    # execute_press_sequence_DYS(
    #     robot=robot,
    #     collector=collector,
    #     begin_pose=begin_pose,      # m
    #     contact_z=args.contact_z,   # m
    #     depth=args.depth,
    #     trajectory_xyz_list=trajectory_xyz,   # relative xyz in m (center [0, 0, 0])
    # # )
    # execute_rotate_sequence_DYS(
    #     robot=robot,
    #     collector=collector,
    #     begin_pose=begin_pose,      # m
    #     contact_z=args.contact_z,   # m
    #     depth=args.depth,
    #     trajectory_xyz_list=trajectory_xyz,   # relative xyz in m (center [0, 0, 0])
    # )
    execute_move_sequence_DYS(
        robot=robot,
        collector=collector,
        begin_pose=begin_pose,      # m
        contact_z=args.contact_z,   # m
        depth=args.depth,
        trajectory_xyz_list=trajectory_xyz,   # relative xyz in m (center [0, 0, 0])
    )

    collector.release()
    robot.close()

if __name__ == "__main__":
    args = Args().parse_args()
    main(args)

