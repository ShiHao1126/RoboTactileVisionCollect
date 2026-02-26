import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
import spdlog
import flexivrdk

from RoboTactileVisionCollect.rizon.rizon import Rizon

def main():
    logger = spdlog.ConsoleLogger("Test")

    robot = Rizon(tool_name="Flange")

    # rotate 30 degrees around z axis
    logger.info("Rotating 30 degrees around z axis...")
    for i in range(3):
        robot.move_l(target_pose=robot.get_rot_pose(angle=-10.0), vel=0.002, angVel=10.0)
        time.sleep(1.0)
    logger.info("Rotation complete")

if __name__ == "__main__":
    main()