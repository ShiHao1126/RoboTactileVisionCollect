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
from RoboTactileVisionCollect.devices import Collector_LZL
from RoboTactileVisionCollect.trajectory import *

logger = spdlog.ConsoleLogger('Collect_LZL')

# 右上：[0.62472, -0.01117]
# 左下：[0.61562, -0.00469]
# x_step = 0.00455 y_step = 0.00216

# 垂直下压：0-15N，0.1N为间隔，六个点，每个点压三次，每次间隔1s
# 水平移动：5-15N，1N为间隔，一个中心点，按标号顺序走米字，每个力走两次
# 5-10N：1mm, 10-15N:1.5mm, 每个来回要抬起再压下，


class Args(Tap):
    force: float = 1.0
    step_size: float = 0.0005
    num_frames: int = 1

    record_dir: str = "/home/ubuntu/data/LZLData/record_20260129/press"
    # record_dir: str = "/home/ubuntu/data/LZLData/record_20260129/move"

    tool_name: str = "zjn"
    force_overwrite: bool = True
    enable_camera: bool = True
    enable_ft: bool = True

    camera_params: dict = {
                'CAMERA_INDEX': 0,
                'WIDTH': 1280,
                'HEIGHT': 960,
                'EXPOSURE_VALUE': 1.0,
                'TEMPERATURE_VALUE': 6500.0,
                'FOCUS_VALUE': 398.0,
                'BRIGHTNESS_VALUE': 0.0,
                'CONTRAST_VALUE': 33.0,
        },

def main(args: Args):
    if args.force_overwrite:
        if os.path.exists(os.path.join(args.record_dir, f'{args.force:.1f}N')):
            shutil.rmtree(os.path.join(args.record_dir, f'{args.force:.1f}N'))
        os.makedirs(os.path.join(args.record_dir, f'{args.force:.1f}N'), exist_ok=True)
    else:
        if os.path.exists(os.path.join(args.record_dir, f'{args.force:.1f}N')):
            raise FileExistsError(f"Record directory {os.path.join(args.record_dir, f'{args.force:.1f}N')} already exists")

    collector = Collector_LZL(
        enable_camera=args.enable_camera,
        enable_ft=args.enable_ft,
        image_save_dir=os.path.join(args.record_dir, f'{args.force:.1f}N', 'images'),
        ft_save_path=os.path.join(args.record_dir, f'{args.force:.1f}N', 'ft.csv'),
        camera_params=args.camera_params,
        num_frames=args.num_frames,
    )

    robot = Rizon(tool_name=args.tool_name)


    # move to begin pose
    begin_xyz = [0.62214, -0.00814, 0.021]
    # begin_xyz = [0.61214, 0.1926, 0.0413]   # test
    # begin_xyz = [0.62254, -0.00642, 0.03481]
    # begin_xyz = [0.61327, 0.19331, 0.05046]
    begin_rot = R.from_euler('xyz', [180, 0, 180], degrees=True).as_matrix()
    begin_quat_xyzw = R.from_matrix(begin_rot).as_quat()
    begin_quat_wxyz = np.array([begin_quat_xyzw[3], begin_quat_xyzw[0], begin_quat_xyzw[1], begin_quat_xyzw[2]], dtype=float)
    if begin_quat_wxyz[0] < 0:
        begin_quat_wxyz = -begin_quat_wxyz
    begin_pose = np.concatenate([begin_xyz, begin_quat_wxyz])
    robot.move_l(
        target_pose=begin_pose,
        vel=0.02,
        zoneRadius="Z50",
        targetTolerLevel=1,
        )
    time.sleep(0.3)

    collector.save_reference_image(ref_save_dir=os.path.join(args.record_dir, f'{args.force:.1f}N', 'ref'), image_name=f'ref')

    # execute_single_point_LZL(robot, collector, begin_pose, args.force, args.step_size)
    
    execute_multi_point_LZL(robot, collector, args.force, begin_pose)

    collector.release()
    robot.close()

if __name__ == "__main__":
    args = Args().parse_args()
    main(args)

