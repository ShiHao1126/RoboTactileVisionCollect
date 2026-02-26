import os
import sys
import time
import spdlog
import numpy as np
from scipy.spatial.transform import Rotation as R
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from RoboTactileVisionCollect.rizon import Rizon
logger = spdlog.ConsoleLogger("ForceComp")


def main():
    robot = Rizon(tool_name="Flange")
    begin_xyz = np.array([0.65, 0.15, 0.3])
    begin_rot = R.from_euler('xyz', [180, 0, 180], degrees=True).as_matrix()
    begin_quat_xyzw = R.from_matrix(begin_rot).as_quat()
    begin_quat_wxyz = np.array([begin_quat_xyzw[3], begin_quat_xyzw[0], begin_quat_xyzw[1], begin_quat_xyzw[2]], dtype=float)
    if begin_quat_wxyz[0] < 0:
        begin_quat_wxyz = -begin_quat_wxyz
    begin_pose = np.concatenate([begin_xyz, begin_quat_wxyz])

    end_pose = begin_pose.copy()
    end_pose[1] = -0.15
    logger.info("moving to begin pose")
    robot.force_comp(
        target_pose=begin_pose,
        stiff_scale=[0.3, 0.3, 0.3, 0.3, 0.3, 0.3],
    )
    time.sleep(10.0)
    logger.info("moving to end pose")

    robot.force_comp(
        target_pose=end_pose,
        stiff_scale=[0.3, 0.3, 0.3, 0.3, 0.3, 0.3],
    )
    time.sleep(10.0)

if __name__ == "__main__":
    main()