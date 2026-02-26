import time
import spdlog
import flexivrdk
import sys
import os

# Add the example_py directory to path to import utility functions
sys.path.append(
    os.path.join(os.path.dirname(__file__), "..", "flexiv_rdk", "example_py")
)
from utility import quat2eulerZYX

# def main():
#     logger = spdlog.ConsoleLogger("Test")
#     frequency = 100
#     period = 1.0 / frequency
#     mode = flexivrdk.Mode

#     robot = flexivrdk.Robot("Rizon4s-063231")
#     if robot.fault():
#         logger.warn("Fault occurred on the connected robot, trying to clear ...")
#         if not robot.ClearFault():
#             logger.error("Fault cannot be cleared, exiting ...")
#             return 1
#         logger.info("Fault on the connected robot is cleared")
    
#     robot.Enable()
#     while not robot.operational():
#         time.sleep(1)
#     logger.info("Robot is now operational")

#     robot.SwitchMode(mode.NRT_PRIMITIVE_EXECUTION)
#     robot.ExecutePrimitive("ZeroFTSensor", dict())
#     robot.ExecutePrimitive("MoveL", dict({"target": [0.65, -0.3, 0.2]}))
#     while not robot.primitive_states()["terminated"]:
#         time.sleep(1)
#     logger.info("Sensor zeroing complete")

#     robot.SwitchMode(mode.NRT_CARTESIAN_MOTION_FORCE)
#     robot.SetForceControlAxis([False, False, False, False, False, False])
    
#     target_pose = robot.states().tcp_pose.copy()
#     # Z axis down 5mm
#     target_pose[2] = robot.states().tcp_pose[2] - 0.001

#     while True:
#         time.sleep(period)
#         if robot.fault():
#             raise Exception("Fault occurred on the connected robot, exiting ...")
        
#         robot.SendCartesianMotionForce(target_pose)

#         if abs(robot.states().tcp_pose[2] - target_pose[2]) < 0.0001:        
#             logger.info(f"FT: {robot.states().ext_wrench_in_world}")
#             logger.info(f"delta z: {target_pose[2] - robot.states().tcp_pose[2]}")
#             break

#     logger.info("Press test completed")


def main_1():
    logger = spdlog.ConsoleLogger("Test")
    mode = flexivrdk.Mode

    robot = flexivrdk.Robot("Rizon4s-063231")
    if robot.fault():
        logger.warn("Fault occurred on the connected robot, trying to clear ...")
        if not robot.ClearFault():
            logger.error("Fault cannot be cleared, exiting ...")
            return 1
        logger.info("Fault on the connected robot is cleared")
    
    robot.Enable()
    while not robot.operational():
        time.sleep(1)
    logger.info("Robot is now operational")

    robot.SwitchMode(mode.NRT_PRIMITIVE_EXECUTION)
    
    # Zero force/torque sensor
    logger.info("Zeroing force/torque sensors...")
    robot.ExecutePrimitive("ZeroFTSensor", dict())
    # Wait for ZeroFTSensor to complete (it uses terminated, not reachedTarget)
    while not robot.primitive_states()["terminated"]:
        time.sleep(0.1)
    logger.info("Sensor zeroing complete")

    # Get current TCP pose [x, y, z, q_w, q_x, q_y, q_z]
    current_pose = robot.states().tcp_pose.copy()

    # Position: keep current x,y, raise z by 1 mm
    target_position = [
        current_pose[0],
        current_pose[1],
        current_pose[2] - 0.006,
    ]

    # Orientation: convert current quaternion [w,x,y,z] to Euler ZYX (deg)
    quat = [current_pose[3], current_pose[4], current_pose[5], current_pose[6]]
    euler_deg = quat2eulerZYX(quat, degree=True)

    logger.info(
        f"Moving TCP from z={current_pose[2]:.4f} to z={target_position[2]:.4f} "
        f"with orientation (deg)={euler_deg}"
    )

    # Execute MoveL to target TCP pose in WORLD frame
    robot.ExecutePrimitive(
        "MoveL",
        {
            "target": flexivrdk.Coord(
                target_position,
                euler_deg,
                ["WORLD", "WORLD_ORIGIN"],
            ),
            "acc": 0.1,
            "vel": 0.05,
            "zoneRadius": "Z50",
            "targetTolerLevel": 1,
        },
    )

    # Wait for MoveL to reach target
    logger.info("Waiting for MoveL to reach target...")
    while True:
        state = robot.primitive_states()
        reached = state.get("reachedTarget", 0)
        if reached:
            break
        if robot.fault():
            logger.error("Fault occurred during MoveL, aborting")
            return 1
        time.sleep(0.1)

    logger.info("MoveL complete")
    final_pose = robot.states().tcp_pose
    logger.info(f"Final z position: {final_pose[2]:.4f}")
    logger.info(f"Delta z: {target_position[2] - final_pose[2]:.6f}")

    HOLD_TIME = 10.0  # 想要保持 10 秒

    robot.SwitchMode(mode.NRT_PRIMITIVE_EXECUTION)
    robot.ExecutePrimitive("Hold", dict())
    start = time.time()

    while time.time() - start < HOLD_TIME:
        state = robot.primitive_states()
        if state.get("terminated", 0):
            # Hold 内部超过 1s 自动结束，就再发一次
            robot.ExecutePrimitive("Hold", dict())
        if robot.fault():
            raise RuntimeError("Fault during Hold")
        time.sleep(0.1)
    
    logger.info("Hold complete")


def main_2():
    logger = spdlog.ConsoleLogger("Test")
    mode = flexivrdk.Mode

    robot = flexivrdk.Robot("Rizon4s-063231")
    if robot.fault():
        logger.warn("Fault occurred on the connected robot, trying to clear ...")
        if not robot.ClearFault():
            logger.error("Fault cannot be cleared, exiting ...")
            return 1
        logger.info("Fault on the connected robot is cleared")
    
    robot.Enable()
    while not robot.operational():
        time.sleep(1)
    logger.info("Robot is now operational")

    robot.SwitchMode(mode.NRT_PRIMITIVE_EXECUTION)
    
    # Zero force/torque sensor
    logger.info("Zeroing force/torque sensors...")
    robot.ExecutePrimitive("ZeroFTSensor", dict())
    # Wait for ZeroFTSensor to complete (it uses terminated, not reachedTarget)
    while not robot.primitive_states()["terminated"]:
        time.sleep(0.1)
    logger.info("Sensor zeroing complete")

    logger.info("Moving down...")

    robot.ExecutePrimitive("Contact", {
        "contactCoord": "world",              # 参考坐标系
        "contactDir": [0.0, 0.0, -1.0],      # 接触方向（向下）
        "contactVel": 0.001,                   # 接触速度 (m/s)
        "maxContactForce": 1.0,               # 最大接触力 (N)
        "enableFineContact": 1                # 启用精细接触模式
    })
    while not robot.primitive_states()["terminated"]:
        time.sleep(0.1)
    logger.info("Contact complete")

if __name__ == "__main__":
    main_2()