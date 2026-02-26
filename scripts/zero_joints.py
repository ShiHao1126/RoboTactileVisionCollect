import os
import sys
import time
import spdlog
import numpy as np
from tap import Tap
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)
from RoboTactileVisionCollect.rizon import Rizon

logger = spdlog.ConsoleLogger("ZeroJoints")

def main():
    robot = Rizon()
    
    robot.robot.ExecutePrimitive("MoveJ", {"target": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]})
    while not robot.robot.primitive_states()["terminated"]:
        time.sleep(0.1)
    logger.info("All joints moved to zero position")

if __name__ == "__main__":
    main()

